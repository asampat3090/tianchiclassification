from __future__ import print_function
# from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import random
import time
import os


class Loader(object):
    """Loader class to post process and load the data used to feed the model."""

    # Default values given.
    def __init__(self, classes=9, reduction=40, batch_size=16, patch_size=5, img_size=128,
                 depth=16, num_hidden=64, epoch=100):
        self.data_dir = '/Users/perezmunoz/Documents/clothes/tianchiclassification/data'
        self.train_dir = '/'.join([self.data_dir, 'train1'])
        self.test_dir = '/'.join([self.data_dir, 'test1'])
        # Filenames templates
        trainname = 'train-file-%d.npy'
        testname = 'test-file-%d.npy'
        # With such construction, we can select a subset of the whole classes.
        self.train_filenames = ['/'.join([self.train_dir, 'files', trainname % cl]) \
                                for cl in range(classes)]
        self.test_filenames = ['/'.join([self.test_dir, 'files', testname % cl]) \
                               for cl in range(classes)]
        self.classes = [i for i in range(classes)]
        self.zip = zip(self.train_filenames, self.test_filenames, self.classes)
        self.new_train_dir = ''
        self.new_test_dir = ''
        self.nb_labels = classes
        self.reduction = reduction
        self.tot_train_img = 10000 * classes  # Number of training images
        self.tot_validate_img = 1000 * classes  # Number of training images
        self.tot_test_img = 1000 * classes  # Number of testing images
        self.img_size = img_size
        self.nb_channels = 1  # grayscale
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = 16
        self.num_hidden = 64  # TO BETTER DEFINE THE HIDDEN NEURONS
        self.epoch = epoch
        self.num_batch = int(self.tot_train_img / self.reduction / self.batch_size)
        # +1 is used to make sure that the whole data is used for the predictions.
        self.eval_batch = int(self.tot_test_img / self.reduction / self.batch_size + 1)

    def run(self, category):
        """Load accordingly the data.

        Returns:
            data (ndarray): images
        """
        # Print the Tianchi data details
        # loader.print_data_details()

        # Generates the subset of the data
        self.subset()

        # Merge datasets. self.subset() need to be run to get the new_paths
        self.merge()

        if category == 'Training':
            # Loading the training data.
            return self.run_load(self.new_train_dir, 'Training')
        else:
            return self.run_load(self.new_test_dir, 'Testing')

    def print_data_details(self):
        """Print the number of images and dimensions for each class.

        Args:
            zip_filenames (zipped list): Path to pickle files containing
                the files to process. Contain the test and train list.
                Contain also the classes labels.
        """
        for f in self.zip:
            in_f_train = open(f[0], 'rb')
            in_f_test = open(f[1], 'rb')
            train_images = np.load(in_f_train)
            test_images = np.load(in_f_test)
            print('Class %d details:' % f[2])
            print('\tTraining data set shape', train_images.shape)
            print('\tTesting data set shape ', test_images.shape)
            # Close the connections to externals files
            in_f_train.close()
            in_f_test.close()
            # Free the memory of the temporary variables read
            del train_images
            del test_images

    def set_new_paths(self, new_train_dir, new_test_dir):
        """Set the new variables of the train/test dir and zip.

        Args:
            new_train_dir (strin): path of the new train data dir with reduction
            new_test_dir (strin): path of the new test data dir with reduction
        """
        self.new_train_dir = new_train_dir
        self.new_test_dir = new_test_dir
        self.new_train_filenames = ['/'.join([self.new_train_dir, 'files', cl]) \
                                    for cl in os.listdir('/'.join([self.train_dir, 'files']))]
        self.new_test_filenames = ['/'.join([self.new_test_dir, 'files', cl]) \
                                   for cl in os.listdir('/'.join([self.test_dir, 'files']))]
        self.new_zip = zip(self.new_train_filenames, self.new_test_filenames, \
                           self.classes)

    def subset(self):
        """Reduces by `reduction` the size of the train/test dataset."""
        # New paths containg the reduced dataset
        new_train_dir = '/'.join([self.data_dir, 'train%d' % self.reduction])
        new_test_dir = '/'.join([self.data_dir, 'test%d' % self.reduction])

        def check_exists(path):
            """Check if the path already exists. If not, create it.

            Args:
                path: Path to test.
            """
            if not os.path.exists(path):
                print('Path %s is created.' % path)
                os.makedirs(path)
                os.makedirs('/'.join([path, 'files']))

        # Does it exists?
        check_exists(new_train_dir)
        check_exists(new_test_dir)

        # Set the paths into the object
        self.set_new_paths(new_train_dir, new_test_dir)

        def create_subset(cls_filename, to_dir, from_dir):
            """Create the subset of the data of type category.

            Args:
                cls_filename (string): filename of the class file to create.
                to_dir (string): directory containing the files to create.
                from_dir (string): directory containing the files to subset.
            """
            to_obj = '/'.join([to_dir, 'files', cls_filename])
            from_obj = '/'.join([from_dir, 'files', cls_filename])
            if not os.path.exists(to_obj):
                print('\tCreating file %s.' % to_obj)
                in_f = open(from_obj, 'rb')
                in_images = np.load(in_f)
                # Total number of images in the initial dataset
                # In full dataset: 10 000
                nb_images = in_images.shape[0]
                # Number of images to keep
                images_kept = int(nb_images / self.reduction)
                # Images are taken randomly
                sub_in_images = in_images[random.sample(range(0, nb_images), images_kept), :, :]
                out_f = open(to_obj, 'wb')

                # pickle.dump(sub_in_images, out_f, pickle.HIGHEST_PROTOCOL)
                np.save(out_f, sub_in_images, allow_pickle=False)

                # Close both connections
                in_f.close()
                out_f.close()
                # After opening and subsetting, free the memory with temp files
                del in_images
                del sub_in_images
            else:
                print('\tFile %s already created.' % to_obj)

        # Create the subsets
        for f in self.new_zip:
            train_f = f[0].split('/')[-1]  # Get the train class file.
            test_f = f[1].split('/')[-1]  # Get the test class file.
            print('Processing images of class %d' % f[2])
            create_subset(train_f, self.new_train_dir, self.train_dir)
            create_subset(test_f, self.new_test_dir, self.test_dir)

    def create_idx(self, category):
        """Create the indexes to shuffle the images.

        Args:
            nb_images (int): total number of images
        Returns:
            dict: indexes for each class.
        """
        if category == 'train':
            nb_images = int(self.tot_train_img / self.reduction)
        else:
            nb_images = int(self.tot_test_img / self.reduction)
        result = {}
        idx = list(range(nb_images))
        random.shuffle(idx)

        # Number of images per class. With reduction=10, nb=1000
        # Indeed, nb_images=90000 when we consider classes=9
        nb = int(nb_images / self.nb_labels)
        for c in self.classes:
            result[c] = {}
            for i in range(nb):
                result[c][i] = idx[i + nb * c]
        return result

    def merge(self):
        """Merger of the classes."""
        self.merge_classes(self.new_train_dir, 'train')
        self.merge_classes(self.new_test_dir, 'test')

    def merge_classes(self, data_dir, category):
        """Merge the files into one matrix.
        User need to know in advance how many images there are.

        Args:
            data_dir (string): directory from where merge the files.
            category (string): either `train` or `test`.
        Returns:
            ndarray: merged dataset.
        """
        print('Merging the classes for %s...' % category)
        if category == 'train':
            nb = self.tot_train_img / self.reduction
        else:
            nb = self.tot_test_img / self.reduction
        t0 = time.time()  # Start
        print('Total number of images to merge %d' % nb)
        # Only is tested data.pickle. Indeed, both files are created at
        # the same time. Impossible to have only one file.
        if not os.path.exists('/'.join([data_dir, str(self.nb_labels), 'data.npy'])):
            # Create the directory containing the merged class data/labels.npy.
            os.makedirs('/'.join([data_dir, str(self.nb_labels)]))
            print('\tGenerating indexes...')
            idx = self.create_idx(category)
            print('\tIndexes generated.')
            # Result data and labels ndarray that is being created.
            data = np.zeros(shape=(nb, self.img_size, self.img_size))
            labels = np.zeros(shape=(nb, self.nb_labels))
            print('\tEmpty data and labels matrix created.')

            # Sanity check in order to avoid incoherences in list length.
            if '.DS_Store' in os.listdir(os.path.join(data_dir, 'files')):
                os.remove('/'.join([data_dir, 'files', '.DS_Store']))
            # List all class files. Limited to the classes selected.
            filenames = os.listdir('/'.join([data_dir, 'files']))[:self.nb_labels]
            for f in zip(filenames, self.classes):  # Use to retrieve the indexes.
                print('\tProcessing file %s' % f[0])
                in_f = open("/".join([data_dir, 'files', f[0]]), 'rb')
                images = np.load(in_f)
                for i in range(len(images)):
                    ix = idx[f[1]][i]  # Retrieve the new index
                    data[ix, :, :] = images[i, :, :]  # Copy the image into data
                    labels[ix, f[1]] = 1
                print('\tEnf of processing of %s' % f[0])
            print('Writing the new data...')
            with open('/'.join([data_dir, str(self.nb_labels), 'data.npy']), 'wb') as out_data_f:
                np.save(out_data_f, data, allow_pickle=False)
            del data
            print('Data written.')
            print('Writing the new labels...')
            out_labels_f = open('/'.join([data_dir, str(self.nb_labels), 'labels.npy']), 'wb')
            # pickle.dump(labels, out_labels_f, pickle.HIGHEST_PROTOCOL)
            with open('/'.join([data_dir, str(self.nb_labels), 'labels.npy']), 'wb') as out_labels_f:
                np.save(out_labels_f, labels, allow_pickle=False)
            del labels
            print('Labels written.')
            print('Time processing: %.2f sec' % (time.time() - t0))
        else:  # Files have already been created.
            print('Files have already been created.')

    def run_load(self, data_dir, category):
        """Runner to load the data in memory.

        Returns:
            container: data/labels loaded.
        """

        def load_data(data_dir):
            """Load the data stored in the data_dir.

            Args:
                data_dir (string): data directory to load.
            Returns:
                dataset: dataset for train/test.
                labels: labels for train/test.
            """
            t0 = time.time()
            print('Data from %s' % data_dir.split('/')[-1])
            print('\tLoading data...')
            data = load(os.path.join(data_dir, str(self.nb_labels), 'data.npy'))
            print('\tData loaded in %.2f' % (time.time() - t0))
            t0 = time.time()
            print('\tLoading the labels...')
            labels = load(os.path.join(data_dir, str(self.nb_labels), 'labels.npy'))
            print('\tLabels loaded in %.2f' % (time.time() - t0))
            return data, labels

        def load(path):
            """Load npy file.

            Args:
                path (string): path of the pickle file.
            Returns:
                data (ndarray): pickled file.
            """
            f = open(path, 'rb')
            return np.load(f)

        class Container(object):
            """Object containing the data and labels."""

            def __init__(self, img_size=128):
                self.img_size = img_size
                self.nb_channels = 1

            def reformat(self, category):
                """Format the ndarray to fit into a tensorflow Tensor.

                Args:
                    container (Container): contains the data and labels objects (ndarray).
                Returns:
                    container: formatted data/labels ndarray.
                """
                self.data = self.data.reshape(
                    (-1, self.img_size, self.img_size, self.nb_channels)).astype(np.float32)
                print('%s set' % category, self.data.shape, 'labels', self.labels.shape)

        container = Container(self.img_size)
        print('Printing data_dir %s of category %s' % (data_dir, category))
        container.data, container.labels = load_data(data_dir)
        container.reformat(category)
        return container
