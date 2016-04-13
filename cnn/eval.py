from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os
import json

from Loader import Loader
from CNN import CNN

# Main repository. Juan's path, do not remove (create new one for youself).
TIANCHI = '/Users/perezmunoz/Documents/clothes/tianchiclassification'

# Command shell to run tensorboard from ..src/cnn/
# tensorboard --logdir=logs/YYYYMMDD-HHMMSS/
# Checkpoint directory checkpoints/YYYYMMDD-HHMMSS/

def path_last_model(path, most_recent=True):
    """Get the path of the directory of a saved model.

    most_recent (bool): indicates whether the most recent trained model is restored.
    path (string): directory path of the model to restore.
                   The format of this file is YYYYMMDD-HHMMSS.
    """
    ckpts = os.path.join(TIANCHI, 'checkpoints')
    if most_recent:
        return os.path.join(ckpts, os.listdir(ckpts)[-1])
    else:
        assert os.path.isfile(os.path.join(ckpts, path)), \
                            'Make sure that you train a model on %s' % path
        return os.path.join(ckpts, path)

def load_hyper_parameters(hyper_param_file):
    """Load the hyperparameters."""
    f = open(hyper_param_file, 'rb')
    params = json.loads(f.read())
    tf.flags.DEFINE_string('tianchi', TIANCHI, """Project's repository.""")
    tf.flags.DEFINE_integer('batch_size', params['batch_size'],
                                """Number of mages to process in a batch.""")
    tf.flags.DEFINE_integer('num_labels', params['num_labels'],
                                """Number of classes.""")
    tf.flags.DEFINE_integer('patch_size', params['patch_size'],
                                """Filter size.""")
    tf.flags.DEFINE_integer('depth', params['depth'],
                                """Number of filters.""")
    tf.flags.DEFINE_integer('num_hidden', params['num_hidden'],
                                """Number of hidden neurons.""")
    tf.flags.DEFINE_integer('num_epochs', params['num_epochs'],
                                """Number of epochs used for training.""")
    tf.flags.DEFINE_integer('reduction', params['reduction'],
                                """1/reduction of the whole data.""")
    tf.flags.DEFINE_integer('image_size', params['image_size'],
                                """Size of the images assuming width and height
                                are equal.""")
    tf.flags.DEFINE_integer('num_channels', params['num_channels'],
                                """Data processed is grayscale.""")

FLAGS = tf.flags.FLAGS

# Load the most recent saved model.
ckpt_dir = path_last_model(path=None)
load_hyper_parameters(os.path.join(ckpt_dir, 'hyper_params.json'))

# Sanity checks.
assert 1 <= FLAGS.num_labels <= 9, 'Currently there are 9 classes handled [1..9].'
assert isinstance(FLAGS.num_labels, int), 'FLAGS.num_labels should be an int.'


graph = tf.Graph()
with graph.as_default():
    # Set seed for repoducible results
    np.random.seed(10)
    
    # Data loader.
    loader = Loader(classes=FLAGS.num_labels, reduction=FLAGS.reduction,
                    batch_size=FLAGS.batch_size, patch_size=FLAGS.patch_size,
                    depth=FLAGS.depth, num_hidden=FLAGS.num_hidden,
                    epoch=FLAGS.num_epochs, img_size=FLAGS.image_size)

    # Return the container with data/labels for train data.
    ctest = loader.run_load(data_dir=os.path.join(os.path.join(FLAGS.tianchi, 'data'),
                                                  'test%d' % FLAGS.reduction),
                            category='Testing')

    # For 9000 images, there are 562.5 batchs to process.
    eval_batch = int(ctest.data.shape[0] / FLAGS.batch_size)

    cnn = CNN(batch_size=FLAGS.batch_size,
              image_size=FLAGS.image_size,
              num_channels=FLAGS.num_channels,
              num_labels=FLAGS.num_labels,
              patch_size=FLAGS.patch_size,
              depth=FLAGS.depth,
              num_hidden=FLAGS.num_hidden)
    
    saver = tf.train.Saver()
    
    sess = tf.Session()
    with sess.as_default():

        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            model_checkpoint_path = ckpt.model_checkpoint_path
            print(model_checkpoint_path)
            if not os.path.isabs(model_checkpoint_path):
                model_checkpoint_path = os.path.join(ckpt_dir, model_checkpoint_path)
            saver.restore(sess, model_checkpoint_path)
        else:
            print("Error while loading model checkpoint.")
        
        # Get the placeholders from the graph by name.
        input_X = graph.get_operation_by_name("input_X").outputs[0]

        # Tensors we want to evaluate.
        predictions = graph.get_operation_by_name("FC/predictions").outputs[0]

        # Collect all the predictions here.
        all_predictions = []
        # Evaluation section.
        for batch in range(eval_batch):  # Number of batchs to process
            offset = (batch * FLAGS.batch_size)
            if offset + FLAGS.batch_size > ctest.data.shape[0]:
                batch_data = ctest.data[offset:, :, :, :]
            else:   
                batch_data = ctest.data[offset:(offset + FLAGS.batch_size), :, :, :]
            feed_dict = {cnn.train_data: batch_data}
            batch_predictions = sess.run(cnn.predictions, feed_dict)
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        print(all_predictions.shape, ctest.labels.shape)
        test_labels = np.argmax(ctest.labels, 1)
        test_length = min(test_labels.shape[0], all_predictions.shape[0])

        # Matches between the predictions and the real labels.
        all_predictions = all_predictions.astype(int)[:test_length]
        test_labels = test_labels[:test_length]
        matches = all_predictions == test_labels
        # Total number of matches.
        correct_predictions = np.sum(matches)
        # Number of test data used.
        # test_length = ctest.labels.shape[0]
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
