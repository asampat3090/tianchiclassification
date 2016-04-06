from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import random
import time
import os


class Loader(object):
    """Loader class to post process and load the data used to feed the model."""


    def __init__(self, classes=9, reduction=40, batch_size=16, patch_size=5,
                 depth=16, num_hidden=64, epoch=100):
        self.data_dir = '/Volumes/Macintosh HD/Users/perezmunoz/Data/tianchi'
        self.train_dir = '/'.join([self.data_dir, 'train1'])
        self.test_dir = '/'.join([self.data_dir, 'test1'])
        self.train_filenames = ['/'.join([self.train_dir, cl]) \
                                for cl in os.listdir(self.train_dir)]
        self.test_filenames = ['/'.join([self.test_dir, cl]) \
                                for cl in os.listdir(self.test_dir)]
        self.classes = [i for i in range(classes)]
        self.zip = zip(self.train_filenames, self.test_filenames, self.classes)
        self.new_train_dir = ""
        self.new_test_dir = ""
        self.nb_labels = 9
        self.reduction = reduction
        self.tot_train_img = 90000
        self.tot_test_img = 18000
        self.img_size = 128
        self.nb_channels = 1  # grayscale
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.depth = 16
        self.num_hidden = 64 # TO BETTER DEFINE THE HIDDEN NEURONS
        self.epoch = epoch
        self.num_steps = self.tot_train_img / self.reduction / self.batch_size

    def run(self):
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

        # Loading the training data
        container_train = self.run_load(self.new_train_dir)
        # Loading the testing data
        container_test = self.run_load(self.new_test_dir)
        return container_train, container_test

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
            train_images = pickle.load(in_f_train)
            test_images = pickle.load(in_f_test)
            print 'Class %d details:' % f[2]
            print '\tTraining data set shape', train_images.shape
            print '\tTesting data set shape ', test_images.shape
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
        self.new_train_filenames = ['/'.join([self.new_train_dir, cl]) \
                                    for cl in os.listdir(self.train_dir)]
        self.new_test_filenames = ['/'.join([self.new_test_dir, cl]) \
                                for cl in os.listdir(self.test_dir)]
        self.new_zip = zip(self.new_train_filenames, self.new_test_filenames,\
                           self.classes)

    def subset(self):
        """Reduces by `reduction` the size of the train/test dataset."""
        # New paths containg the reduced dataset
        new_train_dir = '/'.join([self.data_dir, ''.join(['train', str(self.reduction)])])
        new_test_dir = '/'.join([self.data_dir, ''.join(['test', str(self.reduction)])])

        def check_exists(path):
                """Check if the path already exists. If not, create it.

                Args:
                    path: Path to test.
                """
                if not os.path.exists(path):
                    print('Path %s is being created.' % path)
                    os.makedirs(path)

        # Does it exists?
        check_exists(new_train_dir)
        check_exists(new_test_dir)

        # Set the paths into the object
        self.set_new_paths(new_train_dir, new_test_dir)

        def create_subset(cls_filename, to_dir, from_dir):
            """Create the subset of the data of type category.

            Args:
                cls_filename (string): filename of the class file to create.
                data_dir (string): directory containing the files to create.
            """
            to_obj = '/'.join([to_dir, cls_filename])
            from_obj = '/'.join([from_dir, cls_filename])
            if not os.path.exists(to_obj):
                print '\tCreating file %s.' % to_obj
                in_f = open(from_obj, 'rb')
                in_images = pickle.load(in_f)
                # Total number of images in the initial dataset
                # In full dataset: 10 000
                nb_images = in_images.shape[0]
                # Number of images to keep
                images_kept = nb_images / self.reduction
                # Images are taken randomly
                sub_in_images = in_images[random.sample(xrange(0, nb_images),\
                                                               images_kept),:,:]
                out_f = open(to_obj, 'wb')
                pickle.dump(sub_in_images, out_f)
                # Close both connections
                in_f.close()
                out_f.close()
                # After opening and subsetting, free the memory with temp files
                del in_images
                del sub_in_images
            else:
                print '\tFile %s already created.' % to_obj

        # Create the subsets
        for f in self.new_zip:
            train_f = f[0].split('/')[-1] # Get the train class file
            test_f = f[1].split('/')[-1] # Get the test class file
            print 'Processing images of class %d' % f[2]
            create_subset(train_f, self.new_train_dir, self.train_dir)
            create_subset(test_f, self.new_test_dir, self.test_dir)

    def create_idx(self):
        """Create the indexes to shuffle the images.

        Args:
            nb_images (int): total number of images
        Returns:
            dict: indexes for each class.
        """
        nb_images = self.tot_train_img / self.reduction
        result = {}
        idx = list(range(nb_images))
        random.shuffle(idx)

        # Number of images per class. With reduction=10, nb=1000
        # Indeed, nb_images=9000 et classes=9
        nb = nb_images / len(self.classes)
        for c in self.classes:
            result[c] = {}
            for i in range(nb):
                result[c][i] = idx[i+nb*c]
        return result

    def merge(self):
        """Merger of the classes."""
        self.merge_classes(self.new_train_dir, 'train')
        self.merge_classes(self.new_test_dir, 'test')

    def merge_classes(self, data_dir, category):
        """Merge the files into one matrix.
        User need to know in advance how many images there are.

        Returns:
            ndarray: merged dataset
        """
        print 'Merging the classes for %s...' % category
        nb = self.tot_train_img / self.reduction
        t0 = time.time() # Start
        # Only is tested data.pickle. Indeed, both files are created at
        # the same time. Impossible to have only one file.
        if not os.path.exists('/'.join([data_dir, 'data.pickle'])):
            print '\tGenerating indexes...'
            idx = self.create_idx()
            print '\tIndexes generated.'
            # Result data and labels ndarray that is being created.
            data = np.zeros(shape=(nb, self.img_size, self.img_size))
            labels = np.zeros(shape=(nb, len(self.classes)))
            print '\tEmpty data and labels matrix created.'

            filenames = os.listdir(data_dir) # List all class files.
            for f in zip(filenames, self.classes): # Use to retrieve the indexes.
                if f[0]=='.DS_Store':
                    continue
                print '\tProcessing file %s' % f[0]
                in_f = open("/".join([data_dir, f[0]]), 'rb')
                images = pickle.load(in_f)
                for i in range(len(images)):
                    ix = idx[f[1]][i] # Retrieve the new index
                    data[ix,:,:] = images[i,:,:] # Copy the image into data
                    labels[ix,f[1]] = 1
                print '\tEnf of processing of %s' % f[0]
            print 'Writing the new data...'
            out_data_f = open('/'.join([data_dir, 'data.pickle']), 'wb')
            pickle.dump(data, out_data_f)
            out_data_f.close()
            del data
            print 'Data written.'
            print 'Writing the new labels...'
            out_labels_f = open('/'.join([data_dir, 'labels.pickle']), 'wb')
            pickle.dump(labels, out_labels_f)
            out_labels_f.close()
            del labels
            print 'Labels written.'
            print 'Time processing: %.2f sec' % (time.time()-t0)
        else: # Files have already been created.
            print 'Files have already been created.'

    def run_load(self, data_dir):
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
            print 'Loading data...'
            data = load('/'.join([data_dir, 'data.pickle']))
            print 'Data loaded in %.2f' % (time.time()-t0)
            t0 = time.time()
            print 'Loading the labels...'
            labels = load('/'.join([data_dir, 'labels.pickle']))
            print 'Labels loaded in %.2f' % (time.time()-t0)
            return data, labels

        def load(path):
            """Load pickled file.

            Args:
                path (string): path of the pickle file.
            Returns:
                data (ndarray): pickled file.
            """
            f = open(path, 'rb')
            return pickle.load(f)

        class Container(object):
            """Object containing the data and labels."""
            def __init__(self):
                self.img_size = 128
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
                print 'Training set', self.data.shape, 'labels', self.labels.shape
                print 'Testing set', self.data.shape, 'labels', self.labels.shape

        container = Container()
        container.data, container.labels = load_data(data_dir)
        container.reformat()
        return container




