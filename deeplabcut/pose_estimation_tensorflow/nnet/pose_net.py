'''
Source: DeeperCut by Eldar Insafutdinov
https://github.com/eldar/pose-tensorflow
'''

import re
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import Batch
from deeplabcut.pose_estimation_tensorflow.nnet import losses


net_funcs = {'resnet_50': resnet_v1.resnet_v1_50,
			 'resnet_101': resnet_v1.resnet_v1_101}


def prediction_layer(cfg, input, name, num_outputs):
	with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], padding='SAME',
						activation_fn=None, normalizer_fn=None,
						weights_regularizer=slim.l2_regularizer(cfg.weight_decay)):
		with tf.variable_scope(name):
			pred = slim.conv2d_transpose(input, num_outputs,
										 kernel_size=[3, 3], stride=2,
										 scope='block4')
			return pred


def get_batch_spec(cfg):
	num_joints = cfg.num_joints
	batch_size = cfg.batch_size * cfg.num_views
	spec =  {
		Batch.inputs: [batch_size, None, None, 3],
		Batch.part_score_targets: [batch_size, None, None, num_joints],
		Batch.part_score_weights: [batch_size, None, None, num_joints],
		Batch.locref_targets: [batch_size, None, None, num_joints * 2],
		Batch.locref_mask: [batch_size, None, None, num_joints * 2]
	}
	if cfg.num_views > 1:
		spec[Batch.labels_3d] = [batch_size, num_joints, 4]
	return spec


class PoseNet:
	def __init__(self, cfg):
		self.cfg = cfg

	def extract_features(self, inputs):
		net_fun = net_funcs[self.cfg.net_type]

		mean = tf.constant(self.cfg.mean_pixel,
						   dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
		im_centered = inputs - mean

		# The next part of the code depends upon which tensorflow version you have.
		vers = tf.__version__
		vers = vers.split(".") #Updated based on https://github.com/AlexEMG/DeepLabCut/issues/44
		if int(vers[0])==1 and int(vers[1])<4: #check if lower than version 1.4.
			with slim.arg_scope(resnet_v1.resnet_arg_scope(False)):
				net, end_points = net_fun(im_centered,
										  global_pool=False, output_stride=16)
		else:
			with slim.arg_scope(resnet_v1.resnet_arg_scope()):
				net, end_points = net_fun(im_centered,
										  global_pool=False, output_stride=16,is_training=False)

		return net,end_points

	def prediction_layers(self, features, end_points, reuse=None):
		cfg = self.cfg

		num_layers = re.findall("resnet_([0-9]*)", cfg.net_type)[0]
		layer_name = 'resnet_v1_{}'.format(num_layers) + '/block{}/unit_{}/bottleneck_v1'

		out = {}
		with tf.variable_scope('pose', reuse=reuse):
			part_preds = []
			locrefs = []
			part_pred_interms = []
			for i in range(cfg.num_views):
				part_preds.append(prediction_layer(cfg, features[i::cfg.num_views], 'part_pred_%d'%i, cfg.num_joints))
				if cfg.location_refinement:
					locrefs.append(prediction_layer(cfg, features[i::cfg.num_views], 'locref_pred_%d'%i, cfg.num_joints * 2))
				if cfg.intermediate_supervision:
					interm_name = layer_name.format(3, cfg.intermediate_supervision_layer)
					block_interm_out = end_points[interm_name]
					part_pred_interms.append(prediction_layer(cfg, block_interm_out[i::cfg.num_views],
															   'intermediate_supervision_%d'%i,
															   cfg.num_joints))
			out['part_pred'] = tf.concat([part_preds[i][j, None] for i in range(cfg.num_views) for j in range(cfg.batch_size)], axis=0)

			if cfg.location_refinement:
				out['locref'] = tf.concat([locrefs[i][2*j:2*j+1] for i in range(cfg.num_views) for j in range(cfg.batch_size)], axis=0)

			if cfg.intermediate_supervision:
				out['part_pred_interm'] = tf.concat([part_pred_interms[i][j, None] for i in range(cfg.num_views) for j in range(cfg.batch_size)], axis=0)

			if cfg.multiview_step == 2:
				preds, scores = get_preds(cfg, out['part_pred'], out['locref'])
				preds = tf.stop_gradient(preds); scores = tf.stop_gradient(scores)

				scores = tf.reshape(scores, [1, cfg.num_views*cfg.num_joints])
				scores = slim.fully_connected(scores, cfg.num_joints*cfg.num_views,
					normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu,
					weights_initializer=tf.constant_initializer(np.eye(cfg.num_views*cfg.num_joints)),
					scope='reweighting/1')
				scores = slim.fully_connected(scores, cfg.num_joints*cfg.num_views,
					normalizer_fn=slim.batch_norm, activation_fn=None,
					weights_initializer=tf.constant_initializer(np.eye(cfg.num_views*cfg.num_joints)),
					scope='reweighting/2')
				scores = tf.sigmoid(tf.transpose(tf.reshape(scores, [cfg.num_views, cfg.num_joints]))) # reshaping in this way bc the initialization preserves the meaning of each vector element

				preds = tf.transpose(preds, [1, 0, 2]) # num_joints x num_views x (x,y)
				preds = tf.concat([preds, tf.ones([cfg.num_joints, cfg.num_views, 1])], axis=2)
				preds_3d = project_3d(cfg, cfg.projection_matrices, preds, scores)

				out['pred_3d'] = preds_3d

		return out

	def get_net(self, inputs):
		net, end_points = self.extract_features(inputs)
		return self.prediction_layers(net, end_points)

	def test(self, inputs):
		heads = self.get_net(inputs)
		# prob = tf.sigmoid(heads['part_pred'])
		# return {'part_prob': prob, 'locref': heads['locref']}
		heads['part_prob'] = tf.sigmoid(heads['part_pred'])
		return heads

	def train(self, batch):
		cfg = self.cfg

		heads = self.get_net(batch[Batch.inputs])

		if cfg.multiview_step == 2:
			preds_3d = heads['pred_3d']
			loss = {}
			total_loss = 0

			loss['3d_loss'] = tf.losses.mean_squared_error(batch[Batch.labels_3d][0][:,:3], preds_3d, batch[Batch.labels_3d][0][:,3, None])
			loss['euclidean_error'] = tf.reduce_mean(tf.reduce_sum((batch[Batch.labels_3d][0][:12,:3] - preds_3d[:12])**2, axis=1)**0.5)
			total_loss += loss['3d_loss']

			loss['total_loss'] = total_loss
			return loss

		weigh_part_predictions = cfg.weigh_part_predictions
		part_score_weights = batch[Batch.part_score_weights] if weigh_part_predictions else 1.0

		def add_part_loss(pred_layer):
			return tf.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets],
												   heads[pred_layer],
												   part_score_weights)

		loss = {}
		loss['part_loss'] = add_part_loss('part_pred')
		total_loss = loss['part_loss']
		if cfg.intermediate_supervision:
			loss['part_loss_interm'] = add_part_loss('part_pred_interm') 
			total_loss = total_loss + loss['part_loss_interm']

		if cfg.location_refinement:
			locref_pred = heads['locref']
			locref_targets = batch[Batch.locref_targets]
			locref_weights = batch[Batch.locref_mask]

			loss_func = losses.huber_loss if cfg.locref_huber_loss else tf.losses.mean_squared_error
			loss['locref_loss'] = cfg.locref_loss_weight * loss_func(locref_targets, locref_pred, locref_weights)
			total_loss = total_loss + loss['locref_loss']

		# loss['total_loss'] = slim.losses.get_total_loss(add_regularization_losses=params.regularize)
		loss['total_loss'] = total_loss
		return loss

# solve linear system to get 3D coordinates
# helpful explanation of equation found on pg 5 here: https://hal.inria.fr/inria-00524401/PDF/Sturm-cvpr05.pdf
# projection_matrices is a list, predictions is an array of shape n x num_views x 3
# confidences is n x num_views
def project_3d(cfg, projection_matrices, predictions, confidences):
    n = tf.shape(predictions)[0]
    A1 = tf.cast(tf.tile(tf.concat(projection_matrices, axis=0)[None], [n, 1, 1]), tf.float32)
    A2 = tf.stack(
	    	[tf.concat([tf.zeros([n, view*3]), predictions[:, view, :], tf.zeros([n, (cfg.num_views-view-1)*3])], axis=1) for view in range(cfg.num_views)],
    	axis=2)

    A = tf.concat([A1, A2], axis=2)

    A = A * tf.reshape(tf.tile(confidences[:,:,None], [1,1,3]), [n, 3*cfg.num_views, 1])

    s, u, v = tf.linalg.svd(A)
    preds3d = v[:,:,-1] # last column of V is eigenvector of smallest singular value
    preds3d = preds3d[:,:3] / preds3d[:,3,None]
    return preds3d

def get_preds(cfg, preds, lrefs):
	s = tf.shape(preds)
	predictions = spatial_argmax(preds)
	a, b = tf.meshgrid(tf.range(cfg.num_views), tf.range(cfg.num_joints), indexing='ij')
	a = a[:, :, None]; b = b[:, :, None]
	predictions_i = tf.cast(predictions, tf.int32)
	indices = tf.stack([tf.concat([a, predictions_i, 1+b*2], axis=2), tf.concat([a, predictions_i, b*2], axis=2)], axis=2)
	offsets = tf.gather_nd(lrefs, indices) # num_views x num_joints x i,j
	predictions = predictions*cfg.stride + 0.5*cfg.stride + offsets*cfg.locref_stdev
	predictions = predictions[:,:,::-1]
	scores = tf.gather_nd(preds, tf.concat([a, predictions_i, b], axis=2))
	return predictions, scores

def spatial_argmax(x):
	N, H, W, C = [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]]
	a = tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [N*C, H*W])
	ret = tf.cast(tf.unravel_index(tf.argmax(a, axis=1, output_type=tf.int32), [H,W]), tf.float32)
	ret = tf.reshape(tf.transpose(ret), [N,C,2])
	return ret