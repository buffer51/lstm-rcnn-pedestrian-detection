#!/usr/bin/env python

import sys, time

import numpy as np
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
            l1 = tf.nn.relu(tf.nn.conv2d(X, get_weights([3, 3, 3, 64]), strides = [1, 1, 1, 1], padding = 'SAME')
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
            clas_layer = tf.nn.relu(tf.nn.conv2d(shared_layer, get_weights([1, 1, 512, 2*k]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([2*k]))
            clas_layer = tf.reshape(clas_layer, [-1, 2])

        # Regression layer: conv1-4*k
        with tf.variable_scope('reg'): # Regression layer, 1x1 depth 4*k
            reg_layer = tf.nn.relu(tf.nn.conv2d(shared_layer, get_weights([1, 1, 512, 4*k]), strides = [1, 1, 1, 1], padding = 'SAME')
                            + get_biases([4*k]))
            reg_layer = tf.reshape(reg_layer, [-1, 4])

    return shared_layer, clas_layer, reg_layer

def create_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy):
    learning_rate_summary = tf.scalar_summary('learning_rate', learning_rate)

    loss_clas_summary = tf.scalar_summary('loss_clas', clas_loss)
    loss_reg_summary = tf.scalar_summary('loss_reg', reg_loss)
    loss_rpn_summary = tf.scalar_summary('loss_rpn', rpn_loss)

    stat_accuracy_summary = tf.scalar_summary('stat_accuracy', clas_accuracy)
    stat_positive_percentage_summary = tf.scalar_summary('stat_positive_percentage', clas_positive_percentage)
    stat_positive_accuracy_summary = tf.scalar_summary('stat_positive_accuracy', clas_positive_accuracy)

    return [learning_rate_summary, loss_clas_summary, loss_reg_summary, loss_rpn_summary, stat_accuracy_summary, stat_positive_percentage_summary, stat_positive_accuracy_summary]

def create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy):
    with tf.name_scope('train'):
        return tf.merge_summary(create_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy))

def compute_test_stats(test_placeholders, confusion_matrix):
    print('Confusion matrix:\n{}'.format(confusion_matrix))

    accuracy = float(np.trace(confusion_matrix)) / float(np.sum(confusion_matrix))
    print('Accuracy: {}%'.format(100.0 * accuracy))

    positive_recall = negative_recall = positive_precision = negative_precision = 0.0

    if confusion_matrix[0][0] != 0:
        positive_recall = float(confusion_matrix[0][0]) / float(np.sum(confusion_matrix, axis=1)[0])
        positive_precision = float(confusion_matrix[0][0]) / float(np.sum(confusion_matrix, axis=0)[0])
    if confusion_matrix[1][1] != 0:
        negative_recall = float(confusion_matrix[1][1]) / float(np.sum(confusion_matrix, axis=1)[1])
        negative_precision = float(confusion_matrix[1][1]) / float(np.sum(confusion_matrix, axis=0)[1])

    recall = (positive_recall + negative_recall) / 2.0
    precision = (positive_precision + negative_precision) / 2.0

    print('Recall:\t\t{:.2f}%\t(positive {:.2f}%,\tnegative {:.2f}%)'.format(100.0 * recall, 100.0 * positive_recall, 100.0 * negative_recall))
    print('Precision:\t{:.2f}%\t(positive {:.2f}%,\tnegative {:.2f}%)'.format(100.0 * precision, 100.0 * positive_precision, 100.0 * negative_precision))

    F_score = 2.0 * (precision * recall) / (precision + recall)
    print('F-score: {:.2f}%'.format(100.0 * F_score))

    return {
        test_placeholders[0]: accuracy,
        test_placeholders[1]: positive_recall,
        test_placeholders[2]: negative_recall,
        test_placeholders[3]: recall,
        test_placeholders[4]: positive_precision,
        test_placeholders[5]: negative_precision,
        test_placeholders[6]: precision,
        test_placeholders[7]: F_score
    }

def create_test_summaries(test_placeholders):
    with tf.name_scope('test'):
        accuracy_summary = tf.scalar_summary('accuracy', test_placeholders[0])

        positive_recall_summary = tf.scalar_summary('recall/positive', test_placeholders[1])
        negative_recall_summary = tf.scalar_summary('recall/negative', test_placeholders[2])
        recall_summary = tf.scalar_summary('recall/global', test_placeholders[3])

        positive_precision_summary = tf.scalar_summary('precision/positive', test_placeholders[4])
        negative_precision_summary = tf.scalar_summary('precision/negative', test_placeholders[5])
        precision_summary = tf.scalar_summary('precision/global', test_placeholders[6])

        F_score_summary = tf.scalar_summary('F-score', test_placeholders[7])

        return tf.merge_summary([accuracy_summary, positive_recall_summary, negative_recall_summary, recall_summary, positive_precision_summary, negative_precision_summary,precision_summary, F_score_summary])

def trainer(caltech_dataset, input_placeholder, clas_placeholder, reg_placeholder):
    # Shared CNN
    input_data = tf.div(tf.sub(tf.cast(input_placeholder, tf.float32), 128.0), 128.0)
    shared_cnn = VGG16D(input_data)

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
    clas_loss = tf.div(clas_loss, tf.reduce_sum(clas_examples)) # Normalization

    reg_loss = tf.sub(reg_rpn, reg_truth) # This is not actually the Smooth L1 as defined in http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    reg_loss = tf.square(reg_loss)
    reg_loss = tf.reduce_sum(reg_loss, reduction_indices = 1)
    reg_loss = tf.mul(reg_loss, clas_positive_examples) # Only care for positive examples
    reg_loss = tf.reduce_mean(reg_loss) # Normalization
    lambda_ = tf.Variable(10.0, trainable = False, name = 'lambda')
    reg_loss = tf.mul(reg_loss, lambda_) # Scaling

    rpn_loss = tf.add(clas_loss, reg_loss)

    # Diagnostic statistics
    clas_answer = tf.argmax(clas_truth, 1)
    clas_guess = tf.argmax(clas_rpn, 1)
    clas_comparison = tf.cast(tf.equal(clas_answer, clas_guess), tf.float32)

    clas_accuracy = tf.div(tf.reduce_sum(tf.mul(clas_comparison, clas_examples)), tf.reduce_sum(clas_examples))
    clas_positive_percentage = tf.div(tf.reduce_sum(clas_positive_examples), tf.reduce_sum(clas_examples))
    clas_positive_accuracy = tf.div(tf.reduce_sum(tf.mul(clas_comparison, clas_positive_examples)), tf.reduce_sum(clas_positive_examples))

    global_step = tf.Variable(0, trainable = False, name = 'global_step')
    learning_rate = tf.train.exponential_decay(
        0.01,               # Base learning rate.
        global_step,        # Current index into the dataset.
        1000,               # Decay step.
        0.95,               # Decay rate.
        staircase=True)

    # Use simple momentum for the optimization.
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(rpn_loss,
                                                       global_step = global_step)

    test_steps = [clas_examples, clas_answer, clas_guess]

    # Creating summaries
    train_summaries = create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy)

    return global_step, learning_rate, train_step, train_summaries, test_steps

if __name__ == '__main__':
    ### Create the training & testing sets ###
    caltech_dataset = Caltech()
    caltech_dataset.init()

    ### Declare input & output ###
    input_placeholder = tf.placeholder(tf.uint8, [None, caltech_dataset.input_height, caltech_dataset.input_width, 3]) # 640x480 images, RGB (depth 3)
    clas_placeholder = tf.placeholder(tf.float32, [None, None, 2])
    reg_placeholder = tf.placeholder(tf.float32, [None, None, 4])

    ### Creating the trainer ###
    global_step, learning_rate, train_step, train_summaries, test_steps = trainer(caltech_dataset, input_placeholder, clas_placeholder, reg_placeholder)

    ### Creating test summaries ###
    test_placeholders = [tf.placeholder(tf.float32) for i in range(8)]
    test_summaries = create_test_summaries(test_placeholders)

    ### Create a saver/loader ###
    saver = tf.train.Saver()

    with tf.Session() as sess:
        restore_path = None
        if restore_path:
            # Restore variables from disk.
            saver.restore(sess, restore_path)
            print('Model restored from: {}.'.format(restore_path))
        else:
            # Initialize variables
            tf.initialize_all_variables().run()

        # Start summary writers
        train_writer = tf.train.SummaryWriter('log/train', sess.graph, flush_secs = 10)
        test_writer = tf.train.SummaryWriter('log/test', flush_secs = 10)

        last_epoch = 0
        while caltech_dataset.epoch < 5:
            results = sess.run([train_step, train_summaries], feed_dict = caltech_dataset.get_train_minibatch(input_placeholder, clas_placeholder, reg_placeholder))
            train_writer.add_summary(results[1], global_step = tf.train.global_step(sess, global_step))

            if caltech_dataset.epoch != last_epoch:
                last_epoch = caltech_dataset.epoch

                # Do one pass of the whole testing set
                print('Evaluating...')
                confusion_matrix = np.zeros((2, 2), dtype = np.int64) # Truth as rows, guess as columns
                while caltech_dataset.is_test_minibatch_left():
                    clas_examples, clas_answer, clas_guess = sess.run(test_steps, feed_dict = caltech_dataset.get_test_minibatch(input_placeholder, clas_placeholder, reg_placeholder))

                    true_positives = np.dot(clas_examples, np.multiply(clas_answer, clas_guess))
                    positives = np.dot(clas_examples, clas_answer)
                    true_negatives = np.dot(clas_examples, np.multiply(1 - clas_answer, 1 - clas_guess))
                    negatives = np.dot(clas_examples, 1 - clas_answer)

                    confusion_matrix[0][0] += true_positives # True positives
                    confusion_matrix[0][1] += positives - true_positives # False positives
                    confusion_matrix[1][1] += true_negatives # True negatives
                    confusion_matrix[1][0] += negatives - true_negatives # False negatives

                results = sess.run(test_summaries, feed_dict = compute_test_stats(test_placeholders, confusion_matrix))
                test_writer.add_summary(results, global_step = tf.train.global_step(sess, global_step))

        # Save the model to disk
        save_path = saver.save(sess, '/tmp/model.ckpt')
        print('Model saved in file: {}'.format(save_path))
