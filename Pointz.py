#!/usr/bin/env python

from Utils import *

import os, sys
import argparse
import glob
from six.moves import map, zip, range
import numpy as np

from tensorpack import *
from tensorpack.utils.viz import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.varreplace import freeze_variables
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
import tensorflow as tf
from GAN import GANTrainer, GANModelDesc, SeparateGANTrainer


class ClipCallback(Callback):
	def _setup_graph(self):
		vars = tf.trainable_variables()
		ops = []
		for v in vars:
			n = v.op.name
			if not n.startswith('discrim/'):
				continue
			logger.info("Clip {}".format(n))
			ops.append(tf.assign(v, tf.clip_by_value(v, -0.01, 0.01)))
		self._op = tf.group(*ops, name='clip')

	def _trigger_step(self):
		self._op.run()

class Model(GANModelDesc):
	def _get_inputs(self):
		return [InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'image'),
				InputDesc(tf.float32, (None, 2, DIMY, DIMX), 'membr'), 
				InputDesc(tf.float32, (None, 1, DIMY, DIMX), 'point'), 
				]

	def build_losses(self, vecpos, vecneg, name="WGAN_loss"):
		with tf.name_scope(name=name):
			# the Wasserstein-GAN losses
			self.d_loss = tf.reduce_mean(vecneg - vecpos, name='d_loss')
			self.g_loss = tf.negative(tf.reduce_mean(vecneg), name='g_loss')
			# add_moving_summary(self.d_loss, self.g_loss)
			return self.g_loss, self.d_loss

	#FusionNet
	@auto_reuse_variable_scope
	def generator(self, img, last_dim=2):
		assert img is not None
		return arch_generator(img, last_dim)

	@auto_reuse_variable_scope
	def discriminator(self, img):
		assert img is not None
		return arch_discriminator(img)



	
	def _build_graph(self, inputs):
		A, M, P = inputs


		A = convert_to_range_tanh(A)
		M = convert_to_range_tanh(M)
		P = convert_to_range_tanh(P)

		A = tf.identity(A, name='A')
		M = tf.identity(M, name='M')
		P = tf.identity(P, name='P')

		# A = tf.transpose(A / 128.0 - 1.0, [0, 3, 1, 2], name='A')
		# R = tf.transpose(R / 255.0 - 0.0, [0, 3, 1, 2], name='R')
		# B = tf.transpose(B / 128.0 - 1.0, [0, 3, 1, 2], name='B')

		# use the initializers from torch
		with argscope([Conv2D, Deconv2D, FullyConnected],
					  W_init=tf.contrib.layers.variance_scaling_initializer(factor=0.333, uniform=True),
					  use_bias=False), \
				argscope(BatchNorm, gamma_init=tf.random_uniform_initializer()), \
				argscope([Conv2D, Deconv2D, BatchNorm], data_format='NCHW'), \
				argscope(LeakyReLU, alpha=0.2):
		

			with tf.variable_scope('gen'):
				with tf.variable_scope('membr'):
					AM = self.generator(A, last_dim=2)
				with tf.variable_scope('point'):
					AP = self.generator(AM, last_dim=1)


			with tf.variable_scope('discrim'):
				ccat_real = self.discriminator(tf.concat([A,  M,  P], axis=1, name='real'))
				ccat_fake = self.discriminator(tf.concat([A, AM, AP], axis=1, name='fake'))
		with tf.name_scope('MAE_losses'):
			# recon_frq_AA = tf.reduce_mean(tf.abs(complex2channel(RF(S01, tf.ones_like(R)) - RF(S1, tf.ones_like(R)))), name='recon_frq_AA')
			loss_membr = tf.reduce_mean(tf.abs(M-AM), name='loss_membr')
			loss_point = tf.reduce_mean(tf.abs(P-AP), name='loss_point')

		with tf.name_scope('GAN_losses'):
			G_loss, D_loss = self.build_losses(ccat_real, ccat_fake, name='GAN_ccat')

		self.g_loss = tf.add_n([G_loss,
								loss_membr,
								loss_point, 
								], name='G_loss_total')
		self.d_loss = tf.add_n([D_loss
								], name='D_loss_total')

		add_moving_summary(self.d_loss, self.g_loss)
		wd_g = regularize_cost('gen/.*/W', 		l2_regularizer(1e-5), name='G_regularize')
		wd_d = regularize_cost('discrim/.*/W', 	l2_regularizer(1e-5), name='D_regularize')

		self.g_loss = self.g_loss + wd_g
		self.d_loss = self.d_loss + wd_d

	
		self.collect_variables('gen', 'discrim')
		
		add_moving_summary(loss_membr, loss_point)

		def viz3(name, listTensor):
			im = tf.concat(listTensor, axis=3)
			im = tf.transpose(im, [0, 2, 3, 1])
			im = convert_to_range_imag(im)
			im = tf.clip_by_value(im, 0, 255)
			im = tf.cast(im, tf.uint8, name='viz_' + name)
			return im
			#tf.summary.image(name, im[...,0:1], max_outputs=50)

		# viz3('A_recon', [tf.cast((R-0.5)*2.0, tf.float32), M1, S1, T1, S01, tf.cast(tf.abs(S01-S1), tf.float32), tf.cast(tf.abs(S01-T1), tf.float32)])
		# viz3('B_recon', [tf.cast((R-0.5)*2.0, tf.float32), M2, S2, T2, S02, tf.cast(tf.abs(S02-S2), tf.float32), tf.cast(tf.abs(S02-T2), tf.float32)])
		viz_real = viz3('real', [A[:,0:1,...], A[:,1:2,...], M[:,0:1,...], M[:,1:2,...], P[:,0:1,...]])
		viz_fake = viz3('fake', [A[:,0:1,...], A[:,1:2,...], AM[:,0:1,...], AM[:,1:2,...], AP[:,0:1,...]])

		viz = tf.concat([viz_real, viz_fake], axis=1, name='viz')
		tf.summary.image('viz_train', viz[...,0:1], max_outputs=50)

		# def toImage(im, name='viz'):
		# 	im = tf.transpose(im, [0, 2, 3, 1])
		# 	im = convert_to_range_imag(tf.abs(im))
		# 	im = im-tf.reduce_min(im) 
		# 	im = im/tf.reduce_max(im) 
		# 	im = im * 255.0
		# 	im = tf.clip_by_value(im, 0, 255)
		# 	im = tf.cast(im, tf.uint8, name='viz_' + name)
		# 	return im

		# viz_M1 = toImage(M1, name='viz_M1')
		# viz_S1 = toImage(S1, name='viz_S1')
		# viz_T1 = toImage(T1, name='viz_T1')
		# viz_S1 = tf.identity(255*(S1), name='viz_S1')
		# viz_T1 = tf.identity(255*(T1), name='viz_T1')
		
	def _get_optimizer(self):
		lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
		return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)





def sample(imageDir, maskDir, labelDir, model_path, resultDir):
	# TODO
	# pred_config = PredictConfig(
	# 	session_init=get_model_loader(model_path),
	# 	model=Model(),
	# 	input_names=['A', 'R', 'B'],
	# 	# output_names=['fake_label'])
	# 	output_names=['viz_S1', 'viz_T1'])
	# 	# output_names=['viz_T1', 'psnr/PSNR_f1'])
	# ds_valid = ImageDataFlow(imageDir, maskDir, labelDir, 103, is_training=False)
	# pred = SimpleDatasetPredictor(pred_config, ds_valid)
	
	# filenames = glob.glob(imageDir + '/*.png')
	# from natsort import natsorted
	# filenames = natsorted(filenames)
	# print filenames

	# resultDir1=resultDir+'RefineGAN_R_mask_1/'
	# shutil.rmtree(resultDir1, ignore_errors=True)
	# os.makedirs(resultDir1)

	# resultDir2=resultDir+'RefineGAN_T_mask_1/'
	# shutil.rmtree(resultDir2, ignore_errors=True)
	# os.makedirs(resultDir2)
	# # k=0
	
	# for idx, o in enumerate(pred.get_result()):
	# 	print pred
	# 	print len(o)
	# 	print o[0].shape
	# 	imgS1 = o[0][:, :, :, :]
	# 	imgT1 = o[1][:, :, :, :]

	# 	# stack_patches(o, nr_row=3, nr_col=2, viz=True)
	# 	#colors.append(o) # Take the prediction
	# 	colors1 = np.array(imgS1).astype(np.uint8)
	# 	colors2 = np.array(imgT1).astype(np.uint8)
	# 	head, tail = os.path.split(filenames[idx])
	# 	print tail
	# 	import skimage.io
	# 	skimage.io.imsave(resultDir1+tail, np.squeeze(colors1))
	# 	skimage.io.imsave(resultDir2+tail, np.squeeze(colors2))
	pass
###############################################################################
class VisualizeRunner(Callback):
	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'membr', 'point'], 
			['viz']
			)

	def _before_train(self):
		
		self.valid_ds = get_data(args.data, isTrain=False)

	def _trigger(self):
		for lst in self.valid_ds.get_data():
			viz_valid = self.pred(lst)
			viz_valid = np.squeeze(np.array(viz_valid))

			#print viz_valid.shape

			self.trainer.monitors.put_image('viz_valid', viz_valid)

# if __name__ == '__main__':
def main():
	np.random.seed(2018)
	tf.set_random_seed(2018)
	#https://docs.python.org/3/library/argparse.html
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu',    help='comma seperated list of GPU(s) to use.')
	parser.add_argument('--data',   required=True, 
									help='Data directory, contain trainA/trainB/validA/validB')
	parser.add_argument('--load',   help='Load the model path')
	parser.add_argument('--sample', help='Run the deployment on an instance',
									action='store_true')

	args = parser.parse_args()
	global args
	# python Exp_FusionNet2D_-VectorField.py --gpu='0' --data='arranged/'

	# Set the logger directory
	logger.auto_set_dir()

	train_ds = get_data(args.data, isTrain=True)
	valid_ds = get_data(args.data, isTrain=False)

	train_ds = PrintData(train_ds)
	valid_ds = PrintData(valid_ds)

	train_ds = PrefetchDataZMQ(train_ds, 8)
	valid_ds = PrefetchDataZMQ(valid_ds, 1)
	# Set the GPU
	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	# Running train or deploy
	if args.sample:
		# TODO
		# sample
		pass
	else:
		SeparateGANTrainer(QueueInput(train_ds), Model(), g_period=2, d_period=1).train_with_defaults(
			callbacks=[
				# PeriodicTrigger(DumpParamAsImage('viz'), every_k_epochs=10),
				PeriodicTrigger(ModelSaver(), every_k_epochs=100),
				PeriodicTrigger(InferenceRunner(valid_ds, [ScalarStats('MAE_losses/loss_membr'), 
														   ScalarStats('MAE_losses/loss_point')]), every_k_epochs=1),
				PeriodicTrigger(VisualizeRunner(), every_k_epochs=5),
				ClipCallback(),
				ScheduledHyperParamSetter('learning_rate', 
					[(0, 2e-4), (100, 1e-4), (200, 2e-5), (300, 1e-5), (400, 2e-6), (500, 1e-6)], interp='linear')
				
				],
			# model=Model(),
			steps_per_epoch=train_ds.size(),
			max_epoch=500,
		)

