import tensorflow as tf
import numpy as np


class CNN(object):
	"""A CNN for image classification (Tianchi dataset)."""

	def __init__(self, batch_size, image_size, num_channels,
				 num_labels, patch_size, depth, num_hidden):
		"""
		Args:
			batch_size (int): size of the batch process passsed to the CNN.
			image_size (int): size of the width/height image, assumed equality.
			num_channels (int): RGB channels. For grayscale, num_channels=1.
			num_labels (int): number of labels/classes.
			patch_size (int): width/height of the convolutional filters.
			depth (int): number of filters to consider.
			num_hidden (int): number of hidden neurons.
		"""
		# Training batch input data placeholder.
		self.train_data = tf.placeholder(
			tf.float32, shape=(batch_size, image_size, image_size, num_channels),
			name='input_X')
		# Training batch input labels placeholder.
		self.train_labels = tf.placeholder(
			tf.float32, shape=(batch_size, num_labels), name='input_Y')

		# First layer of convolutions INPUT -> [CONV -> RELU -> MAXPOOL] ->
		with tf.name_scope('CONV-1'):
			W = tf.Variable(tf.truncated_normal(
				[patch_size, patch_size, num_channels, depth], stddev=0.1),
				name='W', trainable=True)
			b = tf.Variable(tf.zeros([depth]), name='b', trainable=True)
			# Convolutional Op with padding ('SAME' parameter).
			conv1 = tf.nn.conv2d(self.train_data, W, [1, 2, 2, 1], padding='SAME')
			# ReLU output.
			h1 = tf.nn.relu(tf.nn.bias_add(conv1, b), name='ReLU')
			# Maxpooling over the outputs. Keep 25% information with 2x2 maxpool.
			pool1 = tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
								   padding='SAME', name='max_pooling')

		W_hs = tf.histogram_summary('CONV1-W', W)
		b_hs = tf.histogram_summary('CONV1-b', b)
		h1_hs = tf.histogram_summary('CONV1-h1', h1)
		pool1_hs = tf.histogram_summary("CONV1-pool", pool1)

		# Second layer of convolutions -> [CONV -> RELU -> MAXPOOL] ->
		with tf.name_scope('CONV-2'):
			W = tf.Variable(tf.truncated_normal(
				[patch_size, patch_size, depth, depth], stddev=0.1),
				name='W', trainable=True)
			b = tf.Variable(tf.zeros([depth]), name='b', trainable=True)
			# Convolutional Op with padding ('SAME' parameter).
			conv2 = tf.nn.conv2d(pool1, W, [1, 2, 2, 1], padding='SAME')
			# ReLU output.
			h2 = tf.nn.relu(tf.nn.bias_add(conv2, b), name='ReLU')
			# Maxpooling over the outputs
			pool2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
								   padding='SAME', name='max_pooling')

		# -> [FC -> RELU] ->
		with tf.name_scope('FC-ReLU'):
			# Shape of W = (16384, 64)
			# Size has to be divided by 16 because we apply 2*(2x2) softmax and
			# then we keep 25% of 25% of the whole information.
			W = tf.Variable(tf.truncated_normal(
				[(image_size // 4 * image_size // 4 * depth) // 16, num_hidden],
				stddev=0.1), name='W', trainable=True)
			b = tf.Variable(tf.constant(1.0, shape=[num_hidden]),
							name='b', trainable=True)
			# Reshaping the CONV-2.pool output.
			shape = pool2.get_shape().as_list()
			pool2_reshaped = tf.reshape(pool2,
										[shape[0], shape[1] * shape[2] * shape[3]])
			h3 = tf.nn.relu(tf.matmul(pool2_reshaped, W) + b)

		# -> [FC] -> logits
		with tf.name_scope('FC'):
			W = tf.Variable(tf.truncated_normal(
				[num_hidden, num_labels], stddev=0.1),
				name='W', trainable=True)
			b = tf.Variable(tf.constant(1.0, shape=[num_labels]),
							name='b', trainable=True)
			# Scores obtained as output of the network.
			self.logits = tf.add(tf.matmul(h3, W), b, name='logits')
			# The predictions of the network.
			self.predictions = tf.argmax(self.logits, 1, name='predictions')

		# Applying sotfmax to the logits previously computed.
		with tf.name_scope('Loss'):
			# Loss is computed between predicted value (logits) and real labels
			# contained in self.train_labels with shape [batch_size, num_labels]
			losses = tf.nn.softmax_cross_entropy_with_logits(
				self.logits,
				self.train_labels,
				name='cross_entropy')
			self.loss = tf.reduce_mean(losses)

		# Accuracy
		with tf.name_scope("Accuracy"):
			# Test whehter the predictions made match the real labels values.
			correct_predictions = tf.equal(self.predictions,
										   tf.argmax(self.train_labels, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'),
										   name='accuracy')
