from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import json

from Loader import Loader
from CNN import CNN

# Command shell to run tensorboard from ..src/cnn/
# tensorboard --logdir=logs/summaries/
# Checkpoint directory logs/checkpoints/YYYYMMDD-HHMMSS/


def load_hyper_parameters(log_dir):
    # Define the model hyperparameters
    with open(os.path.join(log_dir, 'heper_params')) as f:
        params = json.loads(f.read())
        tf.flags.DEFINE_integer('batch_size', params['batch_size'], """Number of images to process in
                                                  a batch.""")
        # Pick a number between 1 and 9.
        tf.flags.DEFINE_integer('num_labels', params['num_labels'], """Number of classes.""")
        tf.flags.DEFINE_integer('patch_size', params['patch_size'], """Filter size.""")
        tf.flags.DEFINE_integer('depth',     params['depth'], """Number of filters.""")
        tf.flags.DEFINE_integer('num_hidden', params['num_hidden'], """Number of hidden neurons.""")
        tf.flags.DEFINE_integer('num_epochs', params['num_epochs'], """Number of epochs used for training.""")
        tf.flags.DEFINE_integer('reduction', params['reduction'], """1/reduction of the whole data.""")

        # the following are not needed for evaluation
        # tf.flags.DEFINE_integer('image_size', params['image_size'], """Size of the images assuming width
        #                                           and height are equal.""")
        # tf.flags.DEFINE_integer('num_channels', params['num_channels'], """Data processed is grayscale.""")
        #
        # tf.flags.DEFINE_integer("evaluate_every", 100, """Evaluate model after n steps.""")
        # tf.flags.DEFINE_integer("checkpoint_every", 5, """Save model after n steps.""")
        # tf.flags.DEFINE_string('log_dir', '../logs/', """Logging directory.""")
        # tf.flags.DEFINE_string('ckpt_dir', ckpt_dir, """Checkpoint logging
        #                                                            directory.""")


# Such file is evaluated from cnn/. Then need to handle relative paths.
log_dir = os.path.abspath('logs/20160409-103358')
ckpt_dir = os.path.abspath('logs/20160409-103358/checkpoints')
load_hyper_parameters(log_dir)

FLAGS = tf.flags.FLAGS
# Sanity checks.
assert 1<=FLAGS.num_labels<=9, 'Currently there are 9 classes handled [1..9].'
assert isinstance(FLAGS.num_labels, int), 'FLAGS.num_labels should be an int.'


# Getting the checkpoint file.
# last_train = os.path.abspath('/'.join([ckpt_dir, os.listdir(FLAGS.ckpt_dir)[-1]]))
# print last_train
# ckpt_file = tf.train.latest_checkpoint(checkpoint_dir=last_train,
#                                        latest_filename='ckpt')

latest_checkpoint = os.path.abspath('/'.join([ckpt_dir,
                                              os.listdir(FLAGS.ckpt_dir)[-1]]))

# ckpt_file = tf.train.get_checkpoint_state(latest_checkpoint)
# if ckpt_file and ckpt_file.model_checkpoint_path:
#     pass
# else:
#     raise NameError('Checkpoint file could not be loaded.')
#
# ckpt_file = tf.train.latest_checkpoint(checkpoint_dir=latest_checkpoint)
# print 'ckpt_file'
# print ckpt_file
#
# print 'ckpt file'
# print ckpt_file
# print ckpt_file.__class__

# Create the default graph.
graph = tf.Graph()
with graph.as_default():
    # Set seed for repoducible results
    np.random.seed(10)

    # Data loader
    loader = Loader(classes=FLAGS.num_labels, reduction=FLAGS.reduction,
                    batch_size=FLAGS.batch_size, patch_size=FLAGS.patch_size,
                    depth=FLAGS.depth, num_hidden=FLAGS.num_hidden,
                    epoch=FLAGS.num_epochs)

    # Return the container with data/labels for train/test datasets.
    ctest = loader.run(category='Testing')

    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables.
        saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_file))
        saver.restore(sess, ckpt_file)

        # Get the placeholders from the graph by name.
        input_x = graph.get_operation_by_name("input_X").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate.
        predictions = graph.get_operation_by_name("FC/predictions").outputs[0]

        # Collect all the predictions here.
        all_predictions = []
        # Evaluation section.
        for batch in range(loader.eval_batch): # Number of batchs to process
            offset = (batch * FLAGS.batch_size) % \
                                (ctest.labels.shape[0] - FLAGS.batch_size)
            batch_data = ctest.data[offset:(offset + FLAGS.batch_size),:,:,:]
            feed_dict = {cnn.train_data: batch_data}
            batch_predictions = sess.run(cnn.predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        # Matches between the predictions and the real labels.
        matches = all_predictions.astype(int)==np.argmax(ctest.labels, 1)
        # Total number of matches.
        correct_predictions = sum(matches)
        # Number of test data used.
        test_length = ctest.labels.shape[0]
        # Accuracy
        acc = 100 * (correct_predictions / float(test_length))
        print('Total number of test examples: %d' % test_length)
        print('Accuracy is %.2f%%' % acc)
