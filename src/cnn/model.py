from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import time
import os


# ****************** global variables *****************

data_dir = '/Volumes/Macintosh HD/Users/Benze/Downloads/data'
train_data_path = "/".join([data_dir, 'training'])
test_data_path = "/".join([data_dir, 'test'])

# train ~ 6Gb and test ~ 1.2 Gb
train_filenames = ["/".join([train_data_path, f]) for f in os.listdir(train_data_path)]
test_filenames = ["/".join([test_data_path, f]) for f in os.listdir(test_data_path)]

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]

nb_total_train_img = 90000
nb_total_test_img = 18000

reduction = 20

image_size = 128
num_labels = 9
num_channels = 1  # grayscale

# ***************** neural network parameters ***********
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64
num_steps = 100


def print_data_details(zip_filenames):
    """Print the number of images and dimensions for each class.
    Args:
        zip_filenames (zipped list): Path to pickle files containing
            the files to process. Contain the test and train list.
            Contain also the classes labels.
    """
    for f in zip_filenames:
        in_f_train = open(f[0], 'rb')
        in_f_test = open(f[1], 'rb')
        train_images = pickle.load(in_f_train)
        test_images = pickle.load(in_f_test)
        print('Class %d details:' % f[2])
        print('\tTraining data set shape', train_images.shape)
        print('\tTesting data set shape ', test_images.shape)

# For prototyping, only take 1/10 fraction, i.e.:
#   - 1000 images per training class
#   - 200  images per testing class
# For prototyping, 1/10 doesn't fit the RAM... With 1/20:
#   - 500 images per training class
#   - 100 images per testing class

def subset(zip_filenames, reduction):
    """Take 1/times the size of the dataset for train/test.

    Args:
        zip_filenames (zipped list): Path to pickle files containing
            the files to process. Contain the test and train list.
            Contain also the classes labels.
        reduction (int): fraction of data to keep.
    """
    new_train_dir = "/".join([data_dir, "".join(["train", str(reduction)])])
    new_test_dir = "/".join([data_dir, "".join(["test", str(reduction)])])

    def check_exists(path):
            """Check if the path already exists. If not, create it.

            Args:
                path: Path to test.
            """
            if not os.path.exists(path):
                print('Path %s is being created.' % path)
                os.makedirs(path)

    check_exists(new_train_dir)
    check_exists(new_test_dir)

    def create_subset(parent_filename, sub_filename, data_dir, cls):
        """Create the subset of the data of type category.

        Args:
            parent_filename (string): filename of parent file.
            sub_filename (string): filename to create.
            data_dir (string): Directory where the file belongs.
            cls (int): current class processed.
        """
        file_obj = "/".join([data_dir, sub_filename])
        if not os.path.exists(file_obj):
            print('\tCreating file %s.' % sub_filename)
            in_f = open(parent_filename, 'rb')
            in_images = pickle.load(in_f)
            # Total number of images in the initial dataset
            nb_images = in_images.shape[0]
            # Number of images to keep
            images_kept = nb_images / reduction
            # Images are taken randomly
            sub_in_images = in_images[random.sample(xrange(0, nb_images), images_kept),:,:]
            out_f = open(file_obj, 'wb')
            pickle.dump(sub_in_images, out_f)
        else:
            print('\tFile %s already created.' % sub_filename)

    for f in zip_filenames:
        train_filename = f[0].split("/")[-1]
        test_filename = f[1].split("/")[-1]
        print('Processing images of class %d' % f[2])
        create_subset(f[0], train_filename, new_train_dir, f[2])
        create_subset(f[1], test_filename, new_test_dir, f[2])

    return new_train_dir, new_test_dir


def create_idx(nb_images, classes):
    """Create the indexes to shuffle the images.

    Args:
        nb_images (int): total number of images
        classes (int list): list of classes.
    Returns:
        dict: indexes for each class.
    """
    result = {}
    idx = list(range(nb_images))
    random.shuffle(idx)

    # Number of images per class. With reduction=10, nb=1000
    # Indeed, nb_images=9000 et classes=9
    nb = nb_images / len(classes)
    for c in classes:
        result[c] = {}
        for i in range(nb):
            result[c][i] = idx[i+nb*c]
    return result


def merge_classes(data_dir, category, nb_images, classes):
    """Merge the files into one matrix.
    Uesr need to know in advance how many images there are.

    Args:
        data_dir (string): directory of the dataset to merge.
        category (string): category of the current data (train/test).
        nb_images (int): total number of images.
        classes (int list): list of the classes to process.
    Returns:
        ndarray: merged dataset
    """
    t0 = time.time()
    # Only is tested data.pickle. Indeed, both files are created at
    # the same time. Impossible to have only one file.
    if not os.path.exists('/'.join([data_dir, 'data.pickle'])):
        print('Generating indexes.')
        idx = create_idx(nb_images, classes)
        print('Indexes generated.')
        # Result data and labels ndarray that is being created.
        data = np.zeros(shape=(nb_images, image_size, image_size))
        labels = np.zeros(shape=(nb_images, len(classes)))
        print('Empty data and labels matrix created.')

        filenames = os.listdir(data_dir) # List all class files.
        for f in zip(filenames, classes): # Use to retrieve the indexes.
            if(f[0] == '.DS_Store'):
                continue
            print('Processing file %s' % f[0])
            in_f = open("/".join([data_dir, f[0]]), 'rb')
            images = pickle.load(in_f)
            for i in range(len(images)):
                ix = idx[f[1]][i] # Retrieve the new index
                data[ix,:,:] = images[i,:,:] # Copy the image into data
                labels[ix,f[1]] = 1
            print('Enf of processing of %s' % f[0])
        print('Writing the new data...')
        out_data_f = open('/'.join([data_dir, 'data.pickle']), 'wb')
        pickle.dump(data, out_data_f)
        out_data_f.close()
        print('Data written.')
        print('Writing the new labels...')
        out_labels_f = open('/'.join([data_dir, 'labels.pickle']), 'wb')
        pickle.dump(labels, out_labels_f)
        out_labels_f.close()
        print('Labels written.')
        print('Time processing: %d sec' % (t0-time.time()))
    else: # Files have already been created.
        print('Files have already been created.')
        print('Time processing: %d sec' % (t0-time.time()))


def load_data(data_dir):
    """Load the data in the data_dir.

    Args:
        data_dir (string): data directory to load.
    Returns:
        dataset: dataset for train/test.
        labels: labels for train/test.
    """
    def load(path):
        """Load pickled file.

        Args:
            path (string): path of the pickle file.
        Returns:
            data (ndarray): pickled file.
        """
        f = open(path, 'rb')
        return pickle.load(f)

    data = load('/'.join([data_dir, 'data.pickle']))
    labels = load('/'.join([data_dir, 'labels.pickle']))
    return data, labels


def reformat(dataset, labels):
    """Format the ndarray to fit into a tensorflow Tensor.

    Args:
        dataset (numpy ndarray): Input data set.
        labels (numpy ndarray): Input data labels.
    Returns:
        ndarray: formatted dataset.
        ndarray: formatted labels.
    """
    dataset = dataset.reshape(
        (-1, image_size, image_size, num_channels)).astype(np.float32)
#     labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def model(data):
    """Create the flow of the model.

    Args:
        data (4-D Tensor): training data.
    Returns:
        4D-Tensor: result of the prediction for each class.
    """
    # 1st vanilla convolution with padding.
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    # 1st ReLU output.
    hidden = tf.nn.relu(conv + layer1_biases)
    # 2nd vanilla convolution with padding.
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    # 2nd ReLU output.
    hidden = tf.nn.relu(conv + layer2_biases)
    # Reshaping the hidden output.
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    # Fully connected neural network.
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases

def model(data):
    """Create the flow of the model.

    Args:
        data (4-D Tensor): training data.
    Returns:
        4D-Tensor: result of the prediction for each class.
    """
    # 1st vanilla convolution with padding.
    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
    # 1st ReLU output.
    hidden = tf.nn.relu(conv + layer1_biases)
    # 2nd vanilla convolution with padding.
    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
    # 2nd ReLU output.
    hidden = tf.nn.relu(conv + layer2_biases)
    # Reshaping the hidden output.
    shape = hidden.get_shape().as_list()
    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
    # Fully connected neural network.
    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    return tf.matmul(hidden, layer4_weights) + layer4_biases


if __name__ == "__main__":
    # print("****** data details *******")
    # print_data_details(zip(train_filenames, test_filenames, classes))

    # create subset of the original data
    new_train_dir, new_test_dir = subset(zip(train_filenames, test_filenames, classes), reduction)
    print('New training directory is: %s' % new_train_dir)
    print('New testing directory is: %s' % new_test_dir)

    print('Merging datasets with reduction %d.' % reduction)
    print('Processing training datatsets.')
    merge_classes(new_train_dir, "train", nb_total_train_img / reduction, classes)
    print('End of processing training datasets.')
    print('Processing testing datatsets.')
    merge_classes(new_test_dir, "test", nb_total_test_img / reduction, classes)
    print('End of processing test datasets.')

    # Load data into memory
    if not ('test_data' in vars() and 'test_labels' in vars()):
        t0 = time.time()
        print('Loading testing data...')
        test_data, test_labels = load_data(new_test_dir)
        print('Testing data/labels loaded in %d sec' % (time.time() - t0))
    else:
        print('Testing data/labels is already loaded.')

    if not (('train_data' in vars()) and ('train_labels' in vars())):
        t0 = time.time()
        print('Loading training data...')
        train_data, train_labels = load_data(new_train_dir)
        print('Training data/labels loaded in %d sec' % (time.time() - t0))
    else:
        print('Training data/labels is alreayd loaded.')

    # Reformat data
    t0 = time.time()
    print('Formatting testing data...')
    test_data, test_labels = reformat(test_data, test_labels)
    print('Testing data/labels formatted in %d sec' % (time.time()-t0))
    t0 = time.time()
    print('Formatting training data...')
    train_data, train_labels = reformat(train_data, train_labels)
    print('Training data/labels formatted in %d sec' % (time.time()-t0))

    print('Training set dataset', train_data.shape, 'labels', train_labels.shape)
    print('Testing set dataset', test_data.shape, 'labels', test_labels.shape)

    graph = tf.Graph()

    # Seting the data, weights and biases variables within the default Graph.
    # Use tf.trainable_variables() gives the trainable variables currently in
    # the graph that are used by the optimizer. Need to catch everything in the
    # environment.
    with graph.as_default():
        # Training data placeholder.
        tf_train_dataset = tf.placeholder(
            tf.float32, shape=(batch_size, image_size, image_size, num_channels),
            name='tf_train_dataset')
        # Training labels placeholder.
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),
            name='tf_train_labels')
        # Testing data placeholder.
        tf_test_dataset = tf.constant(test_data,
            name='tf_test_dataset')

        # 1st convolutional layer Weight and Biases.
        layer1_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, num_channels, depth], stddev=0.1),
                name='layer1_weights', trainable=True)
        layer1_biases = tf.Variable(tf.zeros([depth]),
                name='layer1_biases', trainable=True)
        # 2nd convolutional layer Weight and Biases.
        layer2_weights = tf.Variable(tf.truncated_normal(
                [patch_size, patch_size, depth, depth], stddev=0.1),
                name='layer2_weights', trainable=True)
        layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
                name='layer2_biases', trainable=True)
        # 3rd layer Weight and Biases.
        layer3_weights = tf.Variable(tf.truncated_normal(
                [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),
                name='layer3_weights', trainable=True)
        layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]),
                name='layer3_biases', trainable=True)
        # 4th layer Weight and Biases.
        layer4_weights = tf.Variable(tf.truncated_normal(
                [num_hidden, num_labels], stddev=0.1),
                name='layer4_weights', trainable=True)
        layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),
                name='layer4_biases', trainable=True)
        # Connecting TensorFlow Ops.
        logits = model(tf_train_dataset)

        # Following model uses the cross-entropy model.
        loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels, name='cross_entropy'),
                    name='loss')

        # Optmizer tensor.
        optimizer = tf.train.GradientDescentOptimizer(0.05, use_locking=False,
                                                      name='gradient_descent').minimize(loss, var_list=[
                                                      layer1_weights, layer1_biases,
                                                      layer2_weights, layer2_biases,
                                                      layer3_weights, layer3_biases,
                                                      layer4_weights, layer4_biases])

        # Training computation.
        logits = model(tf_train_dataset)

        # Predictions for the training, validation (not yet implemented), and test data.
        train_prediction = tf.nn.softmax(logits, name='train_prediction')
        # valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
        test_prediction = tf.nn.softmax(model(tf_test_dataset), name='test_prediction')

    # NB: before launching the model, need to restart the network.
    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initialized')
        print(train_labels.shape)

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_data[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 2 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    #           print('Validation accuracy: %.1f%%' % accuracy(
    #                 valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
