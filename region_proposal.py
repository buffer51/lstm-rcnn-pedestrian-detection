#!/usr/bin/env python

import sys, time
from PIL import Image, ImageDraw

import tensorflow as tf

sys.path.append('caltech-dataset')
from caltech import Caltech

### CNN ###
def get_weights(shape):
    return tf.get_variable('weights', shape, initializer = tf.random_normal_initializer(stddev=0.01))
def get_biases(shape):
    return tf.get_variable('biases', shape, initializer = tf.zeros_initializer)

# Implementing CNN part of VGG (16 layers, model 'D') based on http://arxiv.org/pdf/1409.1556v6.pdf
def VGG16D(X):
    with tf.variable_scope('VGG16D'):
        # First, two conv3-64
        with tf.variable_scope('layer1'): # Layer 1, 3x3 depth 64
            l1 = tf.nn.relu(tf.nn.conv2d(tf.cast(X, tf.float32), get_weights([3, 3, 3, 64]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([64]))
        with tf.variable_scope('layer2'): # Layer 2, 3x3 depth 64
            l2 = tf.nn.relu(tf.nn.conv2d(l1, get_weights([3, 3, 64, 64]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([64]))

        # Maxpooling
        m2_3 = tf.nn.max_pool(l2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        # Second, two conv3-128
        with tf.variable_scope('layer3'): # Layer 3, 3x3 depth 128
            l3 = tf.nn.relu(tf.nn.conv2d(m2_3, get_weights([3, 3, 64, 128]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([128]))
        with tf.variable_scope('layer4'): # Layer 4, 3x3 depth 128
            l4 = tf.nn.relu(tf.nn.conv2d(l3, get_weights([3, 3, 128, 128]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([128]))

        # Maxpooling
        m4_5 = tf.nn.max_pool(l4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        # Third, three conv3-256
        with tf.variable_scope('layer5'): # Layer 5, 3x3 depth 256
            l5 = tf.nn.relu(tf.nn.conv2d(m4_5, get_weights([3, 3, 128, 256]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([256]))
        with tf.variable_scope('layer6'): # Layer 6, 3x3 depth 256
            l6 = tf.nn.relu(tf.nn.conv2d(l5, get_weights([3, 3, 256, 256]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([256]))
        with tf.variable_scope('layer7'): # Layer 7, 3x3 depth 256
            l7 = tf.nn.relu(tf.nn.conv2d(l6, get_weights([3, 3, 256, 256]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([256]))

        # Maxpooling
        m7_8 = tf.nn.max_pool(l7, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        # Fourth, three conv3-512
        with tf.variable_scope('layer8'): # Layer 8, 3x3 depth 512
            l8 = tf.nn.relu(tf.nn.conv2d(m7_8, get_weights([3, 3, 256, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))
        with tf.variable_scope('layer9'): # Layer 9, 3x3 depth 512
            l9 = tf.nn.relu(tf.nn.conv2d(l8, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))
        with tf.variable_scope('layer10'): # Layer 10, 3x3 depth 512
            l10 = tf.nn.relu(tf.nn.conv2d(l9, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))
        # Maxpooling
        m10_11 = tf.nn.max_pool(l10, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

        # Fifth, three conv3-512
        with tf.variable_scope('layer11'): # Layer 11, 3x3 depth 512
            l11 = tf.nn.relu(tf.nn.conv2d(m10_11, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))
        with tf.variable_scope('layer12'): # Layer 12, 3x3 depth 512
            l12 = tf.nn.relu(tf.nn.conv2d(l11, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))
        with tf.variable_scope('layer13'): # Layer 13, 3x3 depth 512
            l13 = tf.nn.relu(tf.nn.conv2d(l12, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))

        return l13

# Implementing the additionnal layers for RPN based on http://arxiv.org/pdf/1506.01497.pdf
# Has 2 different outputs: regression of the boxes, and classification of those
def RPN(X, caltech_dataset, training = False):
    k = caltech_dataset.pattern_anchors.num # Number of anchors

    with tf.variable_scope('RPN'):
        # First, a shared conv3-512 layer between the two outputs
        with tf.variable_scope('shared'): # Shared layer, 3x3 depth 512
            shared_layer = tf.nn.relu(tf.nn.conv2d(X, get_weights([3, 3, 512, 512]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([512]))

        # Classification layer: conv1-2*k
        with tf.variable_scope('cls'): # Classification layer, 1x1 depth 2*k
            clas_layer = tf.nn.relu(tf.nn.conv2d(X, get_weights([1, 1, 512, 2*k]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([2*k]))
            clas_layer = tf.reshape(clas_layer, [-1, 2])

        # Regression layer: conv1-4*k
        with tf.variable_scope('reg'): # Regression layer, 1x1 depth 4*k
            reg_layer = tf.nn.relu(tf.nn.conv2d(X, get_weights([1, 1, 512, 4*k]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([4*k]))
            reg_layer = tf.reshape(reg_layer, [-1, 4])

    return shared_layer, clas_layer, reg_layer

def create_summaries(learning_rate, clas_loss, reg_loss, rpn_loss):
    learning_rate_summary = tf.scalar_summary('learning_rate', learning_rate)
    loss_clas_summary = tf.scalar_summary('loss_clas', clas_loss)
    loss_reg_summary = tf.scalar_summary('loss_reg', reg_loss)
    loss_rpn_summary = tf.scalar_summary('loss_rpn', rpn_loss)

    return [learning_rate_summary, loss_clas_summary, loss_reg_summary, loss_rpn_summary]

def create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss):
    with tf.name_scope('train'):
        return tf.merge_summary(create_summaries(learning_rate, clas_loss, reg_loss, rpn_loss))

def create_test_summaries(learning_rate, clas_loss, reg_loss, rpn_loss):
    with tf.name_scope('test'):
        return tf.merge_summary(create_summaries(learning_rate, clas_loss, reg_loss, rpn_loss))

def trainer(caltech_dataset, input_placeholder, clas_placeholder, reg_placeholder):
    # Shared CNN
    shared_cnn = VGG16D(input_placeholder)

    # RPN
    shared_rpn, clas_rpn, reg_rpn = RPN(shared_cnn, caltech_dataset)

    # Get classification truth, to be used to learn the regression only on positive examples
    clas_truth = tf.reshape(tf.cast(clas_placeholder, tf.float32), [-1, 2]) # Reshape to a big list
    clas_examples = tf.reduce_sum(clas_truth, reduction_indices = 1) # All examples (positive or negative, but not unknown) set to 1.0
    clas_positive_examples = tf.squeeze(tf.slice(clas_truth, [0, 1], [-1, 1])) # Only positive examples set to 1.0

    # Reshape regression truth
    reg_truth = tf.reshape(reg_placeholder, [-1, 4])

    # Declare loss functions
    clas_loss = tf.nn.softmax_cross_entropy_with_logits(clas_rpn, clas_truth)
    clas_loss = tf.reduce_sum(tf.mul(clas_loss, clas_examples))
    clas_loss = tf.div(clas_loss, Caltech.MINIBATCH_SIZE) # Normalization

    reg_loss = tf.sub(reg_rpn, reg_truth) # This is not actually the Smooth L1 as defined in http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    reg_loss = tf.square(reg_loss)
    reg_loss = tf.reduce_sum(reg_loss, reduction_indices = 1)
    reg_loss = tf.mul(reg_loss, clas_positive_examples) # Only care for positive examples
    reg_loss = tf.reduce_mean(reg_loss) # Normalization
    lambda_ = tf.Variable(10.0, trainable = False, name = 'lambda')
    reg_loss = tf.mul(reg_loss, lambda_) # Scaling

    rpn_loss = tf.add(clas_loss, reg_loss)

    global_step = tf.Variable(0, trainable = False, name = 'global_step')
    learning_rate = tf.train.exponential_decay(
        0.01,               # Base learning rate.
        global_step,        # Current index into the dataset.
        1000,              # Decay step.
        0.95,               # Decay rate.
        staircase=True)

    # Use simple momentum for the optimization.
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(rpn_loss,
                                                       global_step = global_step)

    test_step = rpn_loss

    # Creating summaries
    train_summaries = create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss)
    test_summaries = create_test_summaries(learning_rate, clas_loss, reg_loss, rpn_loss)

    return global_step, learning_rate, train_step, train_summaries, test_step, test_summaries

if __name__ == '__main__':
    ### Create the training & testing sets ###
    caltech_dataset = Caltech()
    caltech_dataset.init()

    ### Declare input & output ###
    input_placeholder = tf.placeholder(tf.uint8, [None, caltech_dataset.input_height, caltech_dataset.input_width, 3]) # 640x480 images, RGB (depth 3)
    clas_placeholder = tf.placeholder(tf.bool, [None, caltech_dataset.output_height, caltech_dataset.output_width, caltech_dataset.pattern_anchors.num, 2])
    reg_placeholder = tf.placeholder(tf.float32, [None, caltech_dataset.output_height, caltech_dataset.output_width, caltech_dataset.pattern_anchors.num, 4])

    ### Creating the trainer ###
    global_step, learning_rate, train_step, train_summaries, test_step, test_summaries = trainer(caltech_dataset, input_placeholder, clas_placeholder, reg_placeholder)

    with tf.Session() as sess:
        # Initialize variables
        tf.initialize_all_variables().run()

        # Start summary writers
        train_writer = tf.train.SummaryWriter('log/train', sess.graph, flush_secs = 10)
        test_writer = tf.train.SummaryWriter('log/test', flush_secs = 10)

        while caltech_dataset.epoch < 5:
            # Do one pass of the whole training_set
            results = sess.run([train_step, train_summaries], feed_dict = caltech_dataset.get_minibatch(input_placeholder, clas_placeholder, reg_placeholder))
            train_writer.add_summary(results[1], global_step = tf.train.global_step(sess, global_step))

            # if caltech_dataset.epoch != 0 and caltech_dataset.epoch % 5 == 0:
            #     # Do one pass of the whole testing set
            #     for frame in testing_set:
            #         [summary_results, clas_results, reg_results] = sess.run([test_summaries] + test_step, feed_dict = create_feed_dict(frame[0], frame[1], frame[2], annotations, pattern_anchors))
            #         test_writer.add_summary(summary_results, global_step = tf.train.global_step(sess, global_step))
            #
            # sess.run([increment_epoch])
