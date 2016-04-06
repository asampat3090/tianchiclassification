import tensorflow as tf


class CNN(object):
    """Class handling the Convolutional Neural Network."""
    
    def __init__(self, arg):
        super(CNN, self).__init__()
        self.arg = arg

# TODO: reformat this code!
# def accuracy(predictions, labels):
#     return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


# def model(data):
#     """Create the flow of the model.

#     Args:
#         data (4-D Tensor): training data.
#     Returns:
#         4D-Tensor: result of the prediction for each class.
#     """
#     # 1st vanilla convolution with padding.
#     conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
#     # 1st ReLU output.
#     hidden = tf.nn.relu(conv + layer1_biases)
#     # 2nd vanilla convolution with padding.
#     conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
#     # 2nd ReLU output.
#     hidden = tf.nn.relu(conv + layer2_biases)
#     # Reshaping the hidden output.
#     shape = hidden.get_shape().as_list()
#     reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
#     # Fully connected neural network.
#     hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
#     return tf.matmul(hidden, layer4_weights) + layer4_biases

    # graph = tf.Graph()

    # # Seting the data, weights and biases variables within the default Graph.
    # # Use tf.trainable_variables() gives the trainable variables currently in
    # # the graph that are used by the optimizer. Need to catch everything in the
    # # environment.
    # with graph.as_default():
    #     # Training data placeholder.
    #     tf_train_dataset = tf.placeholder(
    #         tf.float32, shape=(batch_size, image_size, image_size, num_channels),
    #         name='tf_train_dataset')
    #     # Training labels placeholder.
    #     tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),
    #         name='tf_train_labels')
    #     # Testing data placeholder.
    #     tf_test_dataset = tf.constant(test_data,
    #         name='tf_test_dataset')

    #     # 1st convolutional layer Weight and Biases.
    #     layer1_weights = tf.Variable(tf.truncated_normal(
    #             [patch_size, patch_size, num_channels, depth], stddev=0.1),
    #             name='layer1_weights', trainable=True)
    #     layer1_biases = tf.Variable(tf.zeros([depth]),
    #             name='layer1_biases', trainable=True)
    #     # 2nd convolutional layer Weight and Biases.
    #     layer2_weights = tf.Variable(tf.truncated_normal(
    #             [patch_size, patch_size, depth, depth], stddev=0.1),
    #             name='layer2_weights', trainable=True)
    #     layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]),
    #             name='layer2_biases', trainable=True)
    #     # 3rd layer Weight and Biases.
    #     layer3_weights = tf.Variable(tf.truncated_normal(
    #             [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1),
    #             name='layer3_weights', trainable=True)
    #     layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]),
    #             name='layer3_biases', trainable=True)
    #     # 4th layer Weight and Biases.
    #     layer4_weights = tf.Variable(tf.truncated_normal(
    #             [num_hidden, num_labels], stddev=0.1),
    #             name='layer4_weights', trainable=True)
    #     layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]),
    #             name='layer4_biases', trainable=True)
    #     # Connecting TensorFlow Ops.
    #     logits = model(tf_train_dataset)

    #     # Following model uses the cross-entropy model.
    #     loss = tf.reduce_mean(
    #                 tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels, name='cross_entropy'),
    #                 name='loss')

    #     # loss summary
    #     tf.scalar_summary("loss", loss)

    #     # Optmizer tensor.
    #     optimizer = tf.train.GradientDescentOptimizer(0.05, use_locking=False,
    #                                                   name='gradient_descent').minimize(loss, var_list=[
    #                                                   layer1_weights, layer1_biases,
    #                                                   layer2_weights, layer2_biases,
    #                                                   layer3_weights, layer3_biases,
    #                                                   layer4_weights, layer4_biases])

    #     # Training computation.
    #     logits = model(tf_train_dataset)

    #     # Predictions for the training, validation (not yet implemented), and test data.
    #     train_prediction = tf.nn.softmax(logits, name='train_prediction')
    #     # valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    #     test_prediction = tf.nn.softmax(model(tf_test_dataset), name='test_prediction')

    #     # https://www.tensorflow.org/versions/master/how_tos/graph_viz/index.html#tensorboard-graph-visualization
    #     # TensorFlow graph visualization
    #     tf.histogram_summary("layer 1 weights", layer1_weights)
    #     tf.histogram_summary("layer 1 biases", layer1_biases)
    #     tf.histogram_summary("layer 2 weights", layer2_weights)
    #     tf.histogram_summary("layer 2 biases", layer2_biases)
    #     tf.histogram_summary("layer 3 weights", layer3_weights)
    #     tf.histogram_summary("layer 3 biases", layer3_biases)
    #     tf.histogram_summary("layer 4 weights", layer4_weights)
    #     tf.histogram_summary("layer 4 biases", layer4_biases)
    #     tf.histogram_summary("train prediction", train_prediction)
    #     tf.histogram_summary("test prediction", test_prediction)

    #     # Merge all the summaries and write them out to /tmp/mnist_logs
    #     merged_summaries = tf.merge_all_summaries()


    # # NB: before launching the model, need to restart the network.
    # with tf.Session(graph=graph) as session:
    #     # https://www.tensorflow.org/versions/master/how_tos/variables/index.html#saving-and-restoring
    #     # Add ops to save and restore all the variables.
    #     saver = tf.train.Saver()

    #     writer = tf.train.SummaryWriter("../../logs/",
    #                                     session.graph.as_graph_def(add_shapes=True))
    #     tf.initialize_all_variables().run()
    #     print('Initialized')
    #     print(train_labels.shape)

    #     for i in range(epoch):
    #         print('epoch %d...' % i)
    #         for step in range(num_steps):
    #             offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    #             batch_data = train_data[offset:(offset + batch_size), :, :, :]
    #             batch_labels = train_labels[offset:(offset + batch_size), :]
    #             feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
    #             _, summary_str, l, predictions = session.run(
    #                 [optimizer, merged_summaries, loss, train_prediction], feed_dict=feed_dict)
    #             if (step % 100 == 0):
    #                 writer.add_summary(summary_str, step)
    #                 print('Minibatch loss at step %d: %f' % (step, l))
    #                 print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
    #                 # print('Validation accuracy: %.1f%%' % accuracy(
    #                 #         valid_prediction.eval(), valid_labels))
    #               # Save the variables to disk.
    #             # if (step%50==0): # every 50 step saves it
    #                 # Here, all the variables are saved because nothing is specified.
    #                 # Otherwise only a set of variables are saved and then need
    #                 # to initializes the rest of the variables in the graph
    #                 # save_path = saver.save(session, '../../ckpt/%d-%d.ckpt' % (i,step))
    #                 # print("Model saved in file: %s" % save_path)

    #         print('Test accuracy epoch %d: %.1f%%' % \
    #                         (i, (accuracy(test_prediction.eval(), test_labels))))

