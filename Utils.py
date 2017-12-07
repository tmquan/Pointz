#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, argparse, glob

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.transform
import skimage.segmentation


# Tensorpack toolbox
import tensorpack.tfutils.symbolic_functions as symbf

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

# Tensorflow 
import tensorflow as tf
from tensorflow import layers
# from tensorflow.contrib.layers.python import layers
###############################################################################
SHAPE = 256
BATCH = 1
TEST_BATCH = 100
EPOCH_SIZE = 100
NB_FILTERS = 64  # channel size

DIMX  = 1024
DIMY  = 1024
DIMZ  = 2
DIMC  = 1
###############################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return LeakyReLU(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return LeakyReLU(x, name=name)
###############################################################################
# Utility function for scaling 
def convert_to_range_tanh(x, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / 255.0 - 0.5) * 2.0
###############################################################################
def convert_to_range_imag(x, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * 255.0
###############################################################################		
def convert_to_range_sigm(x, name='ToRangeSigm'):
	with tf.variable_scope(name):
		return (x / 1.0 + 1.0) / 2.0


###############################################################################
# FusionNet
@layer_register(log_shape=True)
def residual(x, chan, first=False):
	with argscope([Conv2D], nl=INLReLU, stride=1, kernel_shape=3):
		input = x
		return (LinearWrap(x)
				.Conv2D('conv0', chan, padding='SAME')
				.Conv2D('conv1', chan/2, padding='SAME')
				.Conv2D('conv2', chan, padding='SAME', nl=tf.identity)
				.InstanceNorm('inorm')()) + input

###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=1, stride=1):
	with argscope([Conv2D], nl=INLReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		old_shape = inputs.get_shape().as_list()
		results = tf.reshape(results, [-1, chan, old_shape[2]*scale, old_shape[3]*scale])
		return results

###############################################################################
@layer_register(log_shape=True)
def residual_enc(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
		x = (LinearWrap(x)
			# .Dropout('drop', 0.75)
			.Conv2D('conv_i', chan, stride=2) 
			.residual('res_', chan, first=True)
			.Conv2D('conv_o', chan, stride=1) 
			())
		return x

###############################################################################
@layer_register(log_shape=True)
def residual_dec(x, chan, first=False):
	with argscope([Conv2D, Deconv2D], nl=INLReLU, stride=1, kernel_shape=3):
				
		x = (LinearWrap(x)
			.Deconv2D('deconv_i', chan, stride=1) 
			.residual('res2_', chan, first=True)
			.Deconv2D('deconv_o', chan, stride=2) 
			# .Dropout('drop', 0.75)
			())
		return x

###############################################################################
@auto_reuse_variable_scope
def arch_generator(img, last_dim=2):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)
		# e3 = Dropout('dr', e3, 0.75)

		d3 = residual_dec('d3',    e3, NB_FILTERS*4)
		d2 = residual_dec('d2', d3+e2, NB_FILTERS*2)
		d1 = residual_dec('d1', d2+e1, NB_FILTERS*1)
		d0 = residual_dec('d0', d1+e0, NB_FILTERS*1) 
		dd =  (LinearWrap(d0)
				.Conv2D('convlast', last_dim, kernel_shape=3, stride=1, padding='SAME', nl=tf.tanh, use_bias=True) ())
		return dd

@auto_reuse_variable_scope
def arch_discriminator(img):
	assert img is not None
	with argscope([Conv2D, Deconv2D], nl=INLReLU, kernel_shape=3, stride=2, padding='SAME'):
		img = Conv2D('conv0', img, NB_FILTERS, nl=LeakyReLU)
		e0 = residual_enc('e0', img, NB_FILTERS*1)
		# e0 = Dropout('dr', e0, 0.75)
		e1 = residual_enc('e1',  e0, NB_FILTERS*2)
		e2 = residual_enc('e2',  e1, NB_FILTERS*4)

		e3 = residual_enc('e3',  e2, NB_FILTERS*8)

		ret = Conv2D('convlast', e3, 1, stride=1, padding='SAME', nl=tf.identity, use_bias=True)
		return ret


###############################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, imageDir, labelDir, size, dtype='float32', isTrain=True):
		self.dtype      = dtype
		self.imageDir   = imageDir
		self.labelDir   = labelDir
		self._size      = size
		self.isTrain    = isTrain

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.imageDir + '/*.tif')
		labels = glob.glob(self.labelDir + '/*.tif')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		labels = natsorted(labels)


		#
		# Pick randomly a pair of training instance
		#
		# seed = 2015
		# np.random.seed(seed)
		for k in range(self._size):
			rand_index = np.random.randint(0, len(images))
			rand_image = np.random.randint(0, len(images))
			rand_label = np.random.randint(0, len(labels))



			image = skimage.io.imread(images[rand_index])
			label = skimage.io.imread(labels[rand_index])

			# Crop the num image if greater than 50
			# Random crop 50 1024 1024 from 150
			assert image.shape == label.shape
			# numSections = image.shape[0]
			# if numSections > DIMN:
			# 	randz = np.random.randint(0, numSections - DIMN + 1) # DIMN is minimum
			# 	image = image[randz:randz+DIMN,...]
			# 	label = label[randz:randz+DIMN,...]
			dimz, dimy, dimx = image.shape
			# if self.isTrain:
			
			randz = np.random.randint(0, dimz-DIMZ+1)
			randy = np.random.randint(0, dimy-DIMY+1)
			randx = np.random.randint(0, dimx-DIMX+1)
			image = image[randz:randz+DIMZ, randy:randy+DIMY, randx:randx+DIMX]
			label = label[randz:randz+DIMZ, randy:randy+DIMY, randx:randx+DIMX]

			if self.isTrain:
				seed = np.random.randint(0, 20152015)
				seed_image = np.random.randint(0, 2015)
				seed_label = np.random.randint(0, 2015)

				#TODO: augmentation here
				image = self.random_flip(image, seed=seed)        
				image = self.random_reverse(image, seed=seed)
				image = self.random_square_rotate(image, seed=seed)           
				# image = self.random_permute(image, seed=seed)           
				# image = self.random_elastic(image, seed=seed)
				# image = skimage.util.random_noise(image, seed=seed) # TODO
				# image = skimage.util.img_as_ubyte(image)

				label = self.random_flip(label, seed=seed)        
				label = self.random_reverse(label, seed=seed)
				label = self.random_square_rotate(label, seed=seed)   
				# label = self.random_permute(label, seed=seed)   
				# label = self.random_elastic(label, seed=seed)

				# image = self.random_reverse(image, seed=seed)
				# label = self.random_reverse(label, seed=seed)
				# Further augmentation in image

				image = skimage.util.random_noise(image, mean=0, var=0.001, seed=seed) # TODO
				image = skimage.util.img_as_ubyte(image)
				pixel = np.random.randint(-20, 20) 
				image = image + pixel

			# Downsample for test ting
			# image = skimage.transform.resize(image, output_shape=(DIMZ, DIMY, DIMX), order=1, preserve_range=True, anti_aliasing=True)
			# label = skimage.transform.resize(label, output_shape=(DIMZ, DIMY, DIMX), order=0, preserve_range=True)
			# label = label/255.0

			# Calculate vector field
			# dirsx, dirsy, dirsz = self.toVectorField(label)

			membr = np.zeros_like(label)
			for z in range(membr.shape[0]):
				membr[z,...] = 1-skimage.segmentation.find_boundaries(np.squeeze(label[z,...]), mode='thick') #, mode='inner'
			# membr = 1-skimage.segmentation.find_boundaries(label, mode='thick') #, mode='inner'
			membr = 255*membr
			membr[label==0] = 0

			# Calculate pointz
			array = np.zeros_like(label)
			point = array[0,...].copy()
			# point[label[0,...]==label[1,...]] = 255.0
			point = 255*np.equal(label[0,...], label[1,...])
			point[membr[0,...]==0] = 0;
			point[membr[1,...]==0] = 0;

			# image = np.expand_dims(image, axis=0)
			# label = np.expand_dims(label, axis=0)
			# dirsx = np.expand_dims(dirsx, axis=0)
			# dirsy = np.expand_dims(dirsy, axis=0)
			# membr = np.expand_dims(membr, axis=0)
			image = np.expand_dims(image, axis=0)
			membr = np.expand_dims(membr, axis=0)
			point = np.expand_dims(point, axis=0)

			point = np.expand_dims(point, axis=0)

			# image = np.expand_dims(image, axis=-1)
			# membr = np.expand_dims(membr, axis=-1)
			# point = np.expand_dims(point, axis=-1)

			# membr = np.expand_dims(membr, axis=-1)

			yield [image.astype(np.float32), 
				   membr.astype(np.float32),  
				   point.astype(np.float32)]

	def toVectorField(self, label):
		# Calculate vector fields
		# label_list = np.arange(15) #np.unique(label)
		# label_list = np.unique(label[::2, ::2])
		# colorFactor = -(-256//15)
		label_list = np.arange(0, NB_COLOURS)
		label_max  = np.max(label_list)

		colors = np.zeros_like(label)
		depths = np.zeros_like(label)
		dirsx = np.zeros_like(label)
		dirsy = np.zeros_like(label)
		dirsz = np.zeros_like(label)
		for label_i in label_list[1::]:
			# print label_i

			# Construct binary map
			color_i = np.zeros_like(colors)
			color_i[label_i==label] = 1;

			#Construct distance transform
			from scipy.ndimage.morphology import distance_transform_edt 
			depth_i = distance_transform_edt(color_i)
			depths  = depths + depth_i

			# Construct directional map   
			dir_x_i, dir_y_i, dir_z_i = np.gradient(depth_i)
			dirsx = dirsx + dir_x_i/1.0 #26.0
			dirsy = dirsy + dir_y_i/1.0 #26.0
			dirsz = dirsz + dir_z_i/1.0 #26.0
		return dirsx, dirsy, dirsz

	def random_permute(self, image, seed=None):
		assert ((image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_permute = np.random.randint(1,3)
		if random_permute==1:
			permuted = np.transpose(image, (1, 2, 0))
		elif random_permute==2:
			permuted = np.transpose(image, (2, 0, 1))
		elif random_permute==3:
			permuted = np.transpose(image, (0, 1, 2))
		image = permuted.copy()
		return image.astype(np.uint8)

	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
			random_reverse = np.random.randint(1,2)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)        
		random_rotatedeg = 90*np.random.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image
				
	def random_elastic(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		old_shape = image.shape

		if image.ndim==2:
			image = np.expand_dims(image, axis=0) # Make 3D
		new_shape = image.shape
		dimx, dimy = new_shape[1], new_shape[2]
		size = np.random.randint(4,16) #4,32
		ampl = np.random.randint(2, 5) #4,8
		du = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		dv = np.random.uniform(-ampl, ampl, size=(size, size)).astype(np.float32)
		# Done distort at boundary
		du[ 0,:] = 0
		du[-1,:] = 0
		du[:, 0] = 0
		du[:,-1] = 0
		dv[ 0,:] = 0
		dv[-1,:] = 0
		dv[:, 0] = 0
		dv[:,-1] = 0
		import cv2
		from scipy.ndimage.interpolation    import map_coordinates
		# Interpolate du
		DU = cv2.resize(du, (new_shape[1], new_shape[2])) 
		DV = cv2.resize(dv, (new_shape[1], new_shape[2])) 
		X, Y = np.meshgrid(np.arange(new_shape[1]), np.arange(new_shape[2]))
		indices = np.reshape(Y+DV, (-1, 1)), np.reshape(X+DU, (-1, 1))
		
		warped = image.copy()
		for z in range(new_shape[0]): #Loop over the channel
			# print z
			imageZ = np.squeeze(image[z,...])
			flowZ  = map_coordinates(imageZ, indices, order=0).astype(np.float32)

			warpedZ = flowZ.reshape(image[z,...].shape)
			warped[z,...] = warpedZ     
		warped = np.reshape(warped, old_shape)
		return warped

###############################################################################
def get_data(dataDir, isTrain=True):
	if isTrain:
		num=500
	else:
		num=1

	# Process the directories 
	names = ['trainA', 'trainB'] if isTrain else ['validA', 'validB']
	dset  = ImageDataFlow(os.path.join(dataDir, names[0]),
						  os.path.join(dataDir, names[1]),
						  num, 
						  isTrain=isTrain)
	return dset