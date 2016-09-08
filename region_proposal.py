#!/usr/bin/env python

import sys, time

import numpy as np
import tensorflow as tf

sys.path.append('caltech-dataset')
from rework import CaltechDataset

sys.path.append('vgg16')
from vgg16 import VGG16D

def get_weights(shape):
    return tf.get_variable('weights', shape, initializer = tf.random_normal_initializer(stddev = 0.01))
def get_biases(shape):
    return tf.get_variable('biases', shape, initializer = tf.zeros_initializer)

# Implementing additional layers for classification
# This is based on http://arxiv.org/pdf/1506.01497.pdf, but without RoI pooling
# and instead a deeper RPN
def RPN(X, num_anchors, training = False):
    with tf.variable_scope('RPN'):
        # First, a conv3-4096 layer to increase the receptive field
        with tf.variable_scope('layer1'): # Layer 1, 3x3 depth 4096
            l1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(X, get_weights([3, 3, 512, 4096]), strides = [1, 1, 1, 1], padding = 'SAME'),
                            get_biases([4096])))

        # Second, a conv1-4096 layer to increase depth
        with tf.variable_scope('layer2'): # Layer 2, 1x1 depth 4096
            l2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l1, get_weights([1, 1, 4096, 4096]), strides = [1, 1, 1, 1], padding = 'SAME'),
                            get_biases([4096])))

        # Third, a classification layer
        with tf.variable_scope('cls'): # Classification layer, 1x1 depth 2 * num_anchors
            clas_layer = tf.nn.bias_add(tf.nn.conv2d(l2, get_weights([1, 1, 4096, 2 * num_anchors]), strides = [1, 1, 1, 1], padding = 'SAME'),
                            get_biases([2 * num_anchors]))

        # And a classification layer
        with tf.variable_scope('reg'): # Regression layer, 1x1 depth 4 * num_anchors
            reg_layer = tf.nn.bias_add(tf.nn.conv2d(l2, get_weights([1, 1, 4096, 4 * num_anchors]), strides = [1, 1, 1, 1], padding = 'SAME'),
                            get_biases([4 * num_anchors]))

    return clas_layer, reg_layer

def create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy, VGG16D_activations, clas_activations):
    with tf.name_scope('train'):
        learning_rate_summary = tf.scalar_summary('learning_rate', learning_rate)

        loss_clas_summary = tf.scalar_summary('loss/clas', clas_loss)
        loss_reg_summary = tf.scalar_summary('loss/reg', reg_loss)
        loss_rpn_summary = tf.scalar_summary('loss/rpn', rpn_loss)

        stat_accuracy_summary = tf.scalar_summary('stat/accuracy', clas_accuracy)
        stat_positive_percentage_summary = tf.scalar_summary('stat/positive_percentage', clas_positive_percentage)
        stat_positive_accuracy_summary = tf.scalar_summary('stat/positive_accuracy', clas_positive_accuracy)

        VGG16D_histogram = tf.histogram_summary('activations/VGG16D', VGG16D_activations)
        clas_histogram = tf.histogram_summary('activations/clas', clas_activations)

        return tf.merge_summary([learning_rate_summary, loss_clas_summary, loss_reg_summary, loss_rpn_summary, stat_accuracy_summary, stat_positive_percentage_summary, stat_positive_accuracy_summary, VGG16D_histogram, clas_histogram])

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

def trainer(caltech, input_placeholder, clas_placeholder, reg_placeholder):
    # Shared CNN
    input_data = tf.cast(input_placeholder, tf.float32)

    vgg = VGG16D()
    shared_cnn = vgg.build(input_data)

    # RPN
    clas_rpn, reg_rpn = RPN(shared_cnn, caltech.anchors.num)
    clas_rpn = tf.reshape(clas_rpn, [-1, 2]) # Reshape to a big list
    reg_rpn = tf.reshape(reg_rpn, [-1, 4]) # Reshape to a big list

    # Get classification truth, to be used to learn the regression only on positive examples
    clas_truth = tf.reshape(tf.cast(clas_placeholder, tf.float32), [-1, 2]) # Reshape to a big list
    clas_examples = tf.reduce_sum(clas_truth, reduction_indices = 1) # All examples (positive or negative, but not unknown) set to 1.0
    clas_positive_examples = tf.squeeze(tf.slice(clas_truth, [0, 1], [-1, 1])) # Only positive examples set to 1.0

    # Get regression truth
    reg_truth = tf.reshape(tf.cast(reg_placeholder, tf.float32), [-1, 4]) # Reshape to a big list

    # Declare loss functions
    clas_loss = tf.nn.softmax_cross_entropy_with_logits(clas_rpn, clas_truth)
    clas_positive_weight = tf.Variable(CaltechDataset.CLAS_POSITIVE_WEIGHT, trainable = False, name = 'clas_positive_weight')
    clas_loss = tf.reduce_sum((tf.mul(clas_loss, clas_examples) + (clas_positive_weight - 1.0) * tf.mul(clas_loss, clas_positive_examples)) / clas_positive_weight)
    clas_loss = tf.div(clas_loss, tf.reduce_sum(clas_examples)) # Normalization

    reg_loss = tf.abs(tf.sub(reg_rpn, reg_truth))
    # # This is Smooth L1 as defined in http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Girshick_Fast_R-CNN_ICCV_2015_paper.pdf
    # # 0.5 * x^2 if |x| < 1
    # # |x| - 0.5 otherwise
    reg_loss = tf.select(tf.less(reg_loss, 1), tf.mul(tf.square(reg_loss), 0.5), tf.sub(reg_loss, 0.5))
    reg_loss = tf.reduce_sum(reg_loss, reduction_indices = 1)
    reg_loss = tf.reduce_mean(tf.mul(reg_loss, clas_positive_examples))
    lambda_ = tf.Variable(CaltechDataset.LOSS_LAMBDA, trainable = False, name = 'lambda') # Roughly reg_loss & clas_loss are equal, because 100.0 ~ 6000 (num total anchors) / 64 (minibatch size)
    reg_loss = tf.mul(reg_loss, lambda_) # Scaling

    rpn_loss = tf.add(clas_loss, reg_loss)

    # Diagnostic statistics
    clas_answer = tf.argmax(clas_truth, 1)
    clas_guess = tf.argmax(clas_rpn, 1)
    clas_comparison = tf.cast(tf.equal(clas_answer, clas_guess), tf.float32)
    clas_prob = tf.nn.softmax(clas_rpn)

    clas_accuracy = tf.div(tf.reduce_sum(tf.mul(clas_comparison, clas_examples)), tf.reduce_sum(clas_examples))
    clas_positive_percentage = tf.div(tf.reduce_sum(clas_positive_examples), tf.reduce_sum(clas_examples))
    clas_positive_accuracy = tf.div(tf.reduce_sum(tf.mul(clas_comparison, clas_positive_examples)), tf.reduce_sum(clas_positive_examples))

    global_step = tf.Variable(0, trainable = False, name = 'global_step')
    learning_rate = tf.train.exponential_decay(
        0.001,               # Base learning rate.
        global_step,        # Current index into the dataset.
        1000,               # Decay step.
        0.95,               # Decay rate.
        staircase = True)

    # Use simple momentum for the optimization.
    train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(rpn_loss,
                                                       global_step = global_step)

    test_steps = [clas_examples, clas_answer, clas_guess, clas_prob, reg_rpn]

    # Creating summaries
    train_summaries = create_train_summaries(learning_rate, clas_loss, reg_loss, rpn_loss, clas_accuracy, clas_positive_percentage, clas_positive_accuracy, shared_cnn, clas_rpn)

    return global_step, learning_rate, train_step, train_summaries, test_steps, vgg

def accumulate_confusion_matrix(confusion_matrix, clas_examples, clas_answer, clas_guess):
    true_positives = np.dot(clas_examples, np.multiply(clas_answer, clas_guess))
    positives = np.dot(clas_examples, clas_answer)
    true_negatives = np.dot(clas_examples, np.multiply(1 - clas_answer, 1 - clas_guess))
    negatives = np.dot(clas_examples, 1 - clas_answer)

    confusion_matrix[0][0] += true_positives # True positives
    confusion_matrix[0][1] += positives - true_positives # False positives
    confusion_matrix[1][1] += true_negatives # True negatives
    confusion_matrix[1][0] += negatives - true_negatives # False negatives

    return confusion_matrix

if __name__ == '__main__':
    ### Create the training & testing sets ###
    caltech = CaltechDataset()

    ### Declare input & output ###
    input_placeholder = tf.placeholder(tf.uint8, [None, caltech.INPUT_SIZE[0], caltech.INPUT_SIZE[1], 3]) # 640x480 images, RGB (depth 3)
    clas_placeholder = tf.placeholder(tf.uint8, [None, caltech.OUTPUT_SIZE[0], caltech.OUTPUT_SIZE[1], caltech.anchors.num, 2])
    reg_placeholder = tf.placeholder(tf.uint8, [None, caltech.OUTPUT_SIZE[0], caltech.OUTPUT_SIZE[1], caltech.anchors.num, 4])

    ### Creating the trainer ###
    global_step, learning_rate, train_step, train_summaries, test_steps, vgg = trainer(caltech, input_placeholder, clas_placeholder, reg_placeholder)

    ### Creating test summaries ###
    test_placeholders = [tf.placeholder(tf.float32) for i in range(8)]
    test_summaries = create_test_summaries(test_placeholders)

    ### Create a saver/loader ###
    vgg_saver = tf.train.Saver(vgg.get_all_variables(), name = 'vgg_saver') # Restores VGG weights & biases
    full_saver = tf.train.Saver(name = 'full_saver', max_to_keep = None)

    with tf.Session() as sess:
        # Initialize variables
        tf.initialize_all_variables().run()

        vgg_restore_path = 'vgg16/VGG16D.ckpt'
        full_restore_path = None # 'models/2016-09-08-64minibatch-50posratio-norelu-withreg-4000training/model.11.ckpt'

        if full_restore_path:
            # Restore variables from disk.
            full_saver.restore(sess, full_restore_path)
            print('Full model restored from: {}.'.format(full_restore_path))
        elif vgg_restore_path:
            # Restore variables from disk.
            vgg_saver.restore(sess, vgg_restore_path)
            print('VGG model restored from: {}.'.format(vgg_restore_path))

        # Start summary writers
        train_writer = tf.train.SummaryWriter('log/train', sess.graph, flush_secs = 10)
        valid_writer = tf.train.SummaryWriter('log/valid', flush_secs = 10)
        test_writer = tf.train.SummaryWriter('log/test', flush_secs = 10)

        if not full_restore_path:
            # Train the model first (and save it)
            last_epoch = 0
            confusion_matrix = np.zeros((2, 2), dtype = np.int64) # Truth as rows, guess as columns
            print('#### EPOCH {:02d} ####'.format(last_epoch))
            while caltech.epoch < CaltechDataset.MAX_EPOCHS:
                results = sess.run([train_step, train_summaries] + test_steps, feed_dict = caltech.get_training_minibatch(input_placeholder, clas_placeholder, reg_placeholder))
                train_writer.add_summary(results[1], global_step = tf.train.global_step(sess, global_step))

                confusion_matrix = accumulate_confusion_matrix(confusion_matrix, results[2], results[3], results[4])

                if caltech.epoch != last_epoch:
                    last_epoch = caltech.epoch

                    # Write training evaluation
                    results = sess.run(test_summaries, feed_dict = compute_test_stats(test_placeholders, confusion_matrix))
                    train_writer.add_summary(results, global_step = tf.train.global_step(sess, global_step))

                    # Do one pass of the whole validation set
                    print('Validating...')
                    confusion_matrix = np.zeros((2, 2), dtype = np.int64)
                    last_frame = False
                    while not last_frame:
                        feed_dict, last_frame = caltech.get_validation_minibatch(input_placeholder, clas_placeholder, reg_placeholder)
                        results = sess.run(test_steps, feed_dict = feed_dict)

                        confusion_matrix = accumulate_confusion_matrix(confusion_matrix, results[0], results[1], results[2])

                    results = sess.run(test_summaries, feed_dict = compute_test_stats(test_placeholders, confusion_matrix))
                    valid_writer.add_summary(results, global_step = tf.train.global_step(sess, global_step))

                    # Reset for training accumulation
                    confusion_matrix = np.zeros((2, 2), dtype = np.int64)

                    # Save the model to disk
                    save_path = full_saver.save(sess, 'model.{}.ckpt'.format(caltech.epoch - 1))
                    print('Model saved: {}'.format(save_path))

                    if caltech.epoch != CaltechDataset.MAX_EPOCHS:
                        print('#### EPOCH {:02d} ####'.format(last_epoch))

        # Do one pass of the whole testing set
        print('Testing...')
        confusion_matrix = np.zeros((2, 2), dtype = np.int64)
        last_frame = False
        while not last_frame:
            feed_dict, minibatch_used, last_frame = caltech.get_testing_minibatch(input_placeholder, clas_placeholder, reg_placeholder)
            results = sess.run(test_steps, feed_dict = feed_dict)

            confusion_matrix = accumulate_confusion_matrix(confusion_matrix, results[0], results[1], results[2])

            if CaltechDataset.TESTING_SIZE == -1: # Save results only when doing full testing
                clas_guess, guess_pos, guess_scores = caltech.parse_results(results[2], results[3], results[4])
                final_pos, final_scores = caltech.NMS(guess_pos, guess_scores)
                caltech.save_results(minibatch_used[0], minibatch_used[1], minibatch_used[2], final_pos, final_scores)

        results = sess.run(test_summaries, feed_dict = compute_test_stats(test_placeholders, confusion_matrix))
        test_writer.add_summary(results, global_step = tf.train.global_step(sess, global_step))
