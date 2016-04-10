from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import json
import sys

from Loader import Loader
from CNN import CNN

# Command shell to run tensorboard from ..src/cnn/
# tensorboard --logdir=logs/summaries/
# Checkpoint directory logs/checkpoints/YYYYMMDD-HHMMSS/

# Define the model hyperparameters
tf.flags.DEFINE_integer('batch_size', 16, """Number of images to process in
                                          a batch.""")
tf.flags.DEFINE_integer('image_size', 128, """Size of the images assuming width
                                          and height are equal.""")
tf.flags.DEFINE_integer('num_channels', 1, """Data processed is grayscale.""")
# Pick a number between 1 and 9.
tf.flags.DEFINE_integer('num_labels', 9, """Number of classes.""")
tf.flags.DEFINE_integer('patch_size', 5, """Filter size.""")
tf.flags.DEFINE_integer('depth', 16, """Number of filters.""")
tf.flags.DEFINE_integer('num_hidden', 64, """Number of hidden neurons.""")
tf.flags.DEFINE_integer('num_epochs', 5, """Number of epochs used for training.""")
tf.flags.DEFINE_integer("evaluate_every", 10, """Evaluate model after n steps.""")
tf.flags.DEFINE_integer("checkpoint_every", 10, """Save model after n steps.""")
tf.flags.DEFINE_integer('reduction', 40, """1/reduction of the whole data.""")
tf.flags.DEFINE_string('log_dir', '/logs/', """Logging directory.""")
tf.flags.DEFINE_string('ckpt_dir', '/logs/checkpoints/', """Checkpoint logging
                                                           directory.""")

FLAGS = tf.flags.FLAGS

# Sanity checks.
assert 1 <= FLAGS.num_labels <= 9, 'Currently there are 9 classes handled [1..9].'
assert isinstance(FLAGS.num_labels, int), 'FLAGS.num_labels should be an int.'


def dump_parameters(file_path):
	with open(file_path, 'w') as f:
		params = {
			'reduction': FLAGS.reduction,
			'image_size': FLAGS.image_size,
			'num_labels': FLAGS.num_labels,
			'batch_size': FLAGS.batch_size,
			'patch_size': FLAGS.patch_size,
			'depth': FLAGS.depth,
			'num_hidden': FLAGS.num_hidden,
			'num_epochs': FLAGS.num_epochs,
			'num_channels': FLAGS.num_channels
		}
		f.write(json.dumps(params))


if __name__ == '__main__':
	# Set seed for repoducible results
	np.random.seed(10)

	# Handling all variables summaries
	date = time.strftime('%Y%m%d-%H%M%S')
	out_dir = os.path.join(os.path.abspath('logs'), date)
	print('output dir {}'.format(date), file=sys.stderr)

	# Create it if not exists
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

	# log the hyperparameters
	dump_parameters(os.path.join(out_dir, 'hyper_params'))
	# Data loader
	loader = Loader(classes=FLAGS.num_labels, reduction=FLAGS.reduction,
					batch_size=FLAGS.batch_size, patch_size=FLAGS.patch_size,
					depth=FLAGS.depth, num_hidden=FLAGS.num_hidden,
					epoch=FLAGS.num_epochs, img_size=FLAGS.image_size)

	# Return the container with data/labels for train data.
	ctrain = loader.run(category='Training')
	ctest = loader.run(category='Testing')

	# Variables need to be catched in the environnement.
	# tf.trainable_variables() to know which variables are being catched.
	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():  # Run the default session.
			cnn = CNN(batch_size=FLAGS.batch_size,
					  image_size=FLAGS.image_size,
					  num_channels=FLAGS.num_channels,
					  num_labels=FLAGS.num_labels,
					  patch_size=FLAGS.patch_size,
					  depth=FLAGS.depth,
					  num_hidden=FLAGS.num_hidden)

			# Variables counting the number of looping done.
			global_step = tf.Variable(0, name='global_step', trainable=False)

			# Handling the optimizer.
			optimizer = tf.train.GradientDescentOptimizer(0.05)
			grads_and_vars = optimizer.compute_gradients(cnn.loss)
			train_op = optimizer.apply_gradients(grads_and_vars,
												 global_step=global_step)

			# Loss and accuracy summaries
			loss_summary = tf.scalar_summary("Loss", cnn.loss)
			acc_summary = tf.scalar_summary("Accuracy", cnn.accuracy)  # TODO

			# Train Summaries
			train_summary_op = tf.merge_summary([loss_summary, acc_summary])
			train_summary_dir = os.path.join(out_dir, "summaries")
			train_summary_writer = tf.train.SummaryWriter(
				train_summary_dir,
				sess.graph.as_graph_def(add_shapes=True))

			# Checkpoint directory.
			checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
			checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
			if not os.path.exists(checkpoint_dir):
				os.makedirs(checkpoint_dir)

			# Stores the variables.
			saver = tf.train.Saver(tf.all_variables())

			# Initialize all variables
			init = tf.initialize_all_variables()  # Init all variables/Ops
			sess.run(init)
			print('Variables Initialized.')

			# Training section.
			_t_start = time.clock()
			_t_wall_start = time.time()
			for epoch in range(FLAGS.num_epochs):
				print('\rCurrently processing %d epoch...' % epoch)
				for step in range(loader.num_batch):
					# Compute the offset of the current step.
					offset = (step * FLAGS.batch_size) % \
							 (ctrain.labels.shape[0] - FLAGS.batch_size)
					# Subset the batch data with FLAGS.batch_size.
					batch_data = ctrain.data[offset:(offset + FLAGS.batch_size), :, :, :]
					# Subset the batch labels with FLAGS.batch_size.
					batch_labels = ctrain.labels[offset:(offset + FLAGS.batch_size), :]
					feed_dict = {cnn.train_data: batch_data,
								 cnn.train_labels: batch_labels}
					# Running one batch of updating values.
					# fetches returns: if Op -> None, Tensor -> numpy representation.
					_, cstep, summaries, loss, accuracy = sess.run(
						fetches=[train_op, global_step, train_summary_op,
								 cnn.loss, cnn.accuracy],
						feed_dict=feed_dict)
					# Log the summaries for the current batch into log dir.
					train_summary_writer.add_summary(summaries, cstep)
			print('Time used for training: %.2f s' % (time.clock() - _t_start), file=sys.stderr)
			print('Time used for training:(wall) %d s' % (time.time() - _t_wall_start), file=sys.stderr)
			path = saver.save(sess, checkpoint_prefix)
			print('Model checkpoint saved into\n{}'.format(path), file=sys.stderr)

			# ***** Evaluation section ********
			# Collect all the predictions here.
			_t_start = time.clock()
			_t_wall_start = time.time()
			all_predictions = []  # ndarray
			for batch in range(loader.eval_batch):  # Number of batchs to process
				offset = (batch * FLAGS.batch_size) % \
						 (ctest.labels.shape[0] - FLAGS.batch_size)
				batch_data = ctest.data[offset:(offset + FLAGS.batch_size), :, :, :]
				feed_dict = {cnn.train_data: batch_data}
				batch_predictions = sess.run(cnn.predictions, feed_dict)
				all_predictions = np.concatenate([all_predictions, batch_predictions])

			# Class labels of the images.
			labels = np.argmax(ctest.labels, 1)
			# Predicted classes.
			all_predictions = all_predictions.astype(int)  # ndarray
			# Matches between the predictions and the real labels.
			print('Total number of validation examples given: %d' % all_predictions.shape[0], file=sys.stderr)
			test_length = all_predictions.shape[0]
			if all_predictions.shape[0] != labels.shape[0]:
				test_length = min(all_predictions.shape[0], labels.shape[0])
				print('Predictions are made on the first %d images!' % test_length)
			# Which are the the good values.
			correct_predictions = np.sum( \
				all_predictions[:test_length] == labels[:test_length])
			print(all_predictions)
			print(labels)
			# Accuracy
			acc = 100 * (correct_predictions / float(test_length))
			print('Accuracy is %.2f%%' % acc, file=sys.stderr)
			print('Time used for validation %.2f s' % (time.clock() - _t_start), file=sys.stderr)
			print('Time used for validation (wall)  %d s' % (time.time() - _t_wall_start), file=sys.stderr)
