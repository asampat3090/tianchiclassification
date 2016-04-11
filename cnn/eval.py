from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import json

from Loader import Loader
from CNN import CNN

# Such file is evaluated from cnn/. Then need to handle relative paths.
test_dir = os.path.abspath('/home/ubuntu/output32/valid1/files/')
log_dir = os.path.abspath('/home/ubuntu/logs/20160411-061805/')
#ckpt_dir = os.path.join(log_dir, 'checkpoints')
ckpt_dir = os.path.abspath('/home/ubuntu/checkpoints_r1_32/')
#ckpt_file = os.path.join(ckpt_dir, 'ckpt')


# Command shell to run tensorboard from ..src/cnn/
# tensorboard --logdir=logs/summaries/
# Checkpoint directory logs/checkpoints/YYYYMMDD-HHMMSS/


def load_test_data(data_dir, num_of_image_per_class, image_size, num_of_class):
    class Container(object):
        """Object containing the data and labels."""

        def __init__(self, img_size=128):
            self.img_size = img_size
            self.nb_channels = 1

        def reformat(self):
            """Format the ndarray to fit into a tensorflow Tensor.

            Args:
                container (Container): contains the data and labels objects (ndarray).
            Returns:
                container: formatted data/labels ndarray.
            """
            self.data = self.data.reshape(
                (-1, self.img_size, self.img_size, self.nb_channels)).astype(np.float32)

        # print('%s set' % category, self.data.shape, 'labels', self.labels.shape)

    total_images = num_of_image_per_class * num_of_class

    data = np.zeros(shape=(total_images, image_size, image_size))
    labels = np.zeros(shape=(total_images, num_of_class))

    for label, data_file in enumerate(os.listdir(data_dir)):
        with open(os.path.join(data_dir, data_file), 'rb') as f:
            offset = num_of_image_per_class * label
            image_data = np.load(f)
            data[offset: offset + num_of_image_per_class, :, :] = image_data[0:num_of_image_per_class, :, :]
            labels[offset: offset + num_of_image_per_class, label] = 1

    container = Container(image_size)
    container.data = data
    container.labels = labels
    container.reformat()
    return container

def load_hyper_parameters(hyper_param_file):
    # Define the model hyperparameters
    with open(hyper_param_file) as f:
        params = json.loads(f.read())
        tf.flags.DEFINE_integer('batch_size', params['batch_size'], """Number of images to process in a batch.""")
        tf.flags.DEFINE_integer('num_labels', params['num_labels'], """Number of classes.""")
        tf.flags.DEFINE_integer('patch_size', params['patch_size'], """Filter size.""")
        tf.flags.DEFINE_integer('depth', params['depth'], """Number of filters.""")
        tf.flags.DEFINE_integer('num_hidden', params['num_hidden'], """Number of hidden neurons.""")
        tf.flags.DEFINE_integer('num_epochs', params['num_epochs'], """Number of epochs used for training.""")
        tf.flags.DEFINE_integer('reduction', params['reduction'], """1/reduction of the whole data.""")
        tf.flags.DEFINE_integer('image_size', params['image_size'],
                                """Size of the images assuming width and height are equal.""")
        tf.flags.DEFINE_integer('num_channels', params['num_channels'], """Data processed is grayscale.""")

    # the following are not needed for evaluation

    #
    # tf.flags.DEFINE_integer("evaluate_every", 100, """Evaluate model after n steps.""")
    # tf.flags.DEFINE_integer("checkpoint_every", 5, """Save model after n steps.""")
    # tf.flags.DEFINE_string('log_dir', '../logs/', """Logging directory.""")
    # tf.flags.DEFINE_string('ckpt_dir', ckpt_dir, """Checkpoint logging
    #                                                            directory.""")


load_hyper_parameters(os.path.join(log_dir, 'hyper_params'))

FLAGS = tf.flags.FLAGS
# Sanity checks.
assert 1 <= FLAGS.num_labels <= 9, 'Currently there are 9 classes handled [1..9].'
assert isinstance(FLAGS.num_labels, int), 'FLAGS.num_labels should be an int.'

# Getting the checkpoint file.
# last_train = os.path.abspath('/'.join([ckpt_dir, os.listdir(FLAGS.ckpt_dir)[-1]]))
# print last_train
# ckpt_file = tf.train.latest_checkpoint(checkpoint_dir=last_train,
#                                        latest_filename='ckpt')

# latest_checkpoint = os.path.abspath('/'.join([ckpt_dir,
# 											  os.listdir(FLAGS.ckpt_dir)[-1]]))

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

    # Return the container with data/labels for train/test datasets.
    # ctest = loader.run(category='Testing')

    test_container = load_test_data(test_dir, 1000, FLAGS.image_size, FLAGS.num_labels)
    eval_batch = int(test_container.data.shape[0] / FLAGS.batch_size)
    sess = tf.Session()

    cnn = CNN(batch_size=FLAGS.batch_size,
              image_size=FLAGS.image_size,
              num_channels=FLAGS.num_channels,
              num_labels=FLAGS.num_labels,
              patch_size=FLAGS.patch_size,
              depth=FLAGS.depth,
              num_hidden=FLAGS.num_hidden)
    saver = tf.train.Saver()
    with sess.as_default():

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path
            print(model_checkpoint_path)
            if not os.path.isabs(model_checkpoint_path):
                model_checkpoint_path = os.path.join(ckpt_dir, model_checkpoint_path)
            saver.restore(sess, model_checkpoint_path)
        else:
            print("error loading checkpoint")
        # Get the placeholders from the graph by name.
        input_x = graph.get_operation_by_name("input_X").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]

        # Tensors we want to evaluate.
        predictions = graph.get_operation_by_name("FC/predictions").outputs[0]

        # Collect all the predictions here.
        all_predictions = []
        # Evaluation section.
        for batch in range(eval_batch):  # Number of batchs to process
            offset = (batch * FLAGS.batch_size)
            if offset + FLAGS.batch_size > test_container.data.shape[0]:
                batch_data = test_container.data[offset:, :, :, :]
            else:
                batch_data = test_container.data[offset:(offset + FLAGS.batch_size), :, :, :]
            feed_dict = {cnn.train_data: batch_data}
            batch_predictions = sess.run(cnn.predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        print(all_predictions.shape, test_container.labels.shape)
        test_labels = np.argmax(test_container.labels, 1)
        test_length = min(test_labels.shape[0], all_predictions.shape[0])

        print("all predictions")
        print(all_predictions)
        # Matches between the predictions and the real labels.
        all_predictions = all_predictions.astype(int)[:test_length]
        test_labels = test_labels[:test_length]
        matches = all_predictions == test_labels
        # Total number of matches.
        correct_predictions = np.sum(matches)
        # Number of test data used.
        # test_length = test_container.labels.shape[0]
        # Accuracy
        acc = 100 * (correct_predictions / float(test_length))
        print('Total number of test examples: %d' % test_length)
        print('Accuracy is %.2f%%' % acc)

        # generate confusion matrix
        # confusion_matrix = np.ndarray(shape=(9, 9), dtype=np.int32)
        confusion_matrix = np.zeros(shape=(9, 9), dtype=np.int32)
        for label, prediction in zip(test_labels, all_predictions):
            confusion_matrix[label][prediction] += 1
        print(confusion_matrix)
