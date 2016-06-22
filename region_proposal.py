#!/usr/bin/env python

import os, glob, random
from math import sqrt, floor, log, exp
from PIL import Image, ImageDraw
import json
from functools import reduce
import operator

import numpy as np
import tensorflow as tf

MINIBATCH_SIZE = 40

DATASET = 'caltech-dataset/dataset'
TRAIN_SETS = range(0, 5+1)
TEST_SETS = range(6, 10+1)
IMAGE_MODULO = 30 # 1 image per second

### Input ###
def list_files(set_number, seq_number):
    files = glob.glob(DATASET + '/images/set{:02d}/V{:03d}.seq/*.jpg'.format(set_number, seq_number))
    random.shuffle(files)
    return files

def show_image(set_number, seq_number, frame_number, annotations = None):
    image = Image.open(DATASET + '/images/set{:02d}/V{:03d}.seq/{}.jpg'.format(set_number, seq_number, frame_number))

    if annotations != None:
        # Read annotations concerning this frame
        try:
            objects = annotations['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]

            dr = ImageDraw.Draw(image)
            for o in objects:
                pos = o['pos']
                if o['lbl'] == 'person':
                    color = 'red'
                elif o['lbl'] == 'people':
                    color = 'yellow'
                else:
                    color = 'blue'
                dr.rectangle((pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3]), outline = color)
        except KeyError as e:
            print('ERROR: Missing key {}'.format(e))

    image.show()

def IoU(rect1, rect2):
    center1 = ((rect1[0]+rect1[2])/2.0, (rect1[1]+rect1[3])/2.0)
    center2 = ((rect2[0]+rect2[2])/2.0, (rect2[1]+rect2[3])/2.0)

    W1 = rect1[2] - rect1[0]
    H1 = rect1[3] - rect1[1]
    W2 = rect2[2] - rect2[0]
    H2 = rect2[3] - rect2[1]
    W = (W1 + W2)/2.0
    H = (H1 + H2)/2.0

    # Check if they intersect by distance
    if abs(center1[0] - center2[0]) >= W or abs(center1[1] - center2[1]) >= H:
        return 0.0

    # If they do, the intersection's width and height are:
    W_int = W - abs(center1[0] - center2[0])
    H_int = H - abs(center1[1] - center2[1])
    # Hence the intersection area is
    int_area = W_int * H_int

    # Intersection over union
    return int_area / (W1 * H1 + W2 * H2 - int_area)

def parse_annotations():
    with open(DATASET + '/annotations.json') as json_file:
        annotations = json.load(json_file)

    return annotations

def create_anchors(pattern_anchors, image_height, image_width):
    # This is regarding the sliding window after the convolutional layers
    windowSize = (3.0, 3.0) # (height, width)
    windowStride = (1.0, 1.0) # (height, width)
    padding = (floor(windowSize[0]/2.0), floor(windowSize[1]/2.0))

    # This is to be able to compute image coordinates
    imageStride = (16.0, 16.0) # (height, width)

    anchors = []
    for currentWindow in ((y, x) for y in range(int(-padding[0]), int(image_height/imageStride[0] - padding[0])) for x in range(int(-padding[1]), int(image_width/imageStride[1] - padding[1]))): # (height, width), top-left coordinates
        currentImage = (currentWindow[0]*imageStride[0], currentWindow[1]*imageStride[1]) # (height, width), top-left coordinates
        currentImageCenter = (currentImage[0] + windowSize[0]*imageStride[0]/2.0, currentImage[1] + windowSize[1]*imageStride[1]/2.0)# (height, width), center coordinates

        for h, w in zip(pattern_anchors.height_list, pattern_anchors.width_list):
            rect = (currentImageCenter[1] - w/2.0, currentImageCenter[0] - h/2.0, currentImageCenter[1] + w/2.0, currentImageCenter[0] + h/2.0)
            anchors.append({'rect': rect, 'positive': None, 'person': None})

    return anchors

def create_minibatch(set_number, seq_number, frame_number, annotations, pattern_anchors, display = False):
    image = Image.open(DATASET + '/images/set{:02d}/V{:03d}.seq/{}.jpg'.format(set_number, seq_number, frame_number))

    ### Compute person rectangles
    persons = []
    try:
        objects = annotations['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]

        for o in objects:
            pos = o['pos']
            if o['lbl'] in ['person', 'people']:
                rect = (pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3])
                persons.append({'rect': rect, 'used': False})

    except KeyError as e:
        print('ERROR: Missing key {}'.format(e))

    ### Compute anchor rectangles
    anchors = create_anchors(pattern_anchors, image.size[1], image.size[0])

    ### Compute positive and negative examples
    # Compute all IoUs for each (anchor, person)
    IoUs = [[IoU(a['rect'], p['rect']) for p in persons] for a in anchors]

    # Remove anchors that cross the image borders (set their IoUs to 0)
    for i in range(len(anchors)):
        rect = anchors[i]['rect']
        if rect[0] < 0 or rect[1] < 0 or rect[2] >= image.size[0] or rect[3] >= image.size[1]:
            IoUs[i] = [0 for p in persons]

    for i in range(len(anchors)): # For each anchor, find the biggest IoU with a person
        if len(IoUs[i]) == 0:
            anchors[i]['positive'] = False
            continue

        maxIoU = max(IoUs[i])
        personIndexMax = IoUs[i].index(maxIoU)

        if maxIoU >= 0.7:
            persons[personIndexMax]['used'] = True
            anchors[i]['positive'] = True
            anchors[i]['person'] = persons[personIndexMax]['rect']
        elif maxIoU <= 0.3:
            anchors[i]['positive'] = False

    for j in range(len(persons)): # For each person
        if not persons[j]['used']: # Find the biggest IoU if no anchor was positive for this person
            maxIoU = IoUs[0][j]
            anchorIndexMax = 0
            for i in range(1, len(anchors)):
                if IoUs[i][j] > maxIoU:
                    maxIoU = IoUs[i][j]
                    anchorIndexMax = i

            if anchors[anchorIndexMax]['positive']: # Already positive for another person
                pass # Do nothing, this person won't have an anchor, sorry!
            else:
                anchors[anchorIndexMax]['positive'] = True
                anchors[anchorIndexMax]['person'] = persons[j]['rect']
                persons[j]['used'] = True

    # Determine positive and negative examples
    positiveAnchors = [i for i in range(len(anchors)) if anchors[i]['positive'] == True]
    negativeAnchors = [i for i in range(len(anchors)) if anchors[i]['positive'] == False]

    # Select MINIBATCH_SIZE/2 (or less) positives examples randomly
    random.shuffle(positiveAnchors)
    if len(positiveAnchors) > MINIBATCH_SIZE/2:
        for i in positiveAnchors[MINIBATCH_SIZE/2:]: # In order to do that, we remove the 'positiveness' from the ones not selected
            anchors[i]['positive'] = None
            anchors[i]['person'] = None
        positiveAnchors = positiveAnchors[:MINIBATCH_SIZE/2]

    # Select negative examples to have a total number of examples of MINIBATCH_SIZE
    random.shuffle(negativeAnchors)
    if len(negativeAnchors) + len(positiveAnchors) > MINIBATCH_SIZE:
        for i in negativeAnchors[MINIBATCH_SIZE - len(positiveAnchors):]: # Same here
            anchors[i]['positive'] = None
        negativeAnchors = negativeAnchors[:MINIBATCH_SIZE - len(positiveAnchors)]

    ### Create input tensor
    pixels_list = list(image.getdata())
    pixels_data = np.expand_dims(np.reshape(np.array(pixels_list), [image.size[1], image.size[0], 3]), axis = 0)

    ### Create output tensors for training
    # Create the tensor p, which is used for classification (values are (negative, positive))
    # NOTE: The order is negative then positive, because if both are 0 (unused), argmax will choose the first one, and we want that to be negative.
    clas_list = [[0.0 for k in range(2)] for i in range(len(anchors))]

    # Create the tensor t, which is used for regression (values are (tx, ty, tw, th))
    reg_list = [[0.0 for k in range(4)] for i in range(len(anchors))]

    for i in positiveAnchors:
        clas_list[i][1] = 1.0

        # Regression values
        xAnchor = anchors[i]['rect'][0]
        yAnchor = anchors[i]['rect'][1]
        wAnchor = anchors[i]['rect'][2] - xAnchor
        hAnchor = anchors[i]['rect'][3] - yAnchor

        xPerson = anchors[i]['person'][0]
        yPerson = anchors[i]['person'][1]
        wPerson = anchors[i]['person'][2] - xPerson
        hPerson = anchors[i]['person'][3] - yPerson

        reg_list[i][0] = (xPerson - xAnchor) / wAnchor # tx
        reg_list[i][1] = (yPerson - yAnchor) / hAnchor # ty
        reg_list[i][2] = log(wPerson / wAnchor) # tw
        reg_list[i][3] = log(hPerson / hAnchor) # th
    for i in negativeAnchors:
        clas_list[i][0] = 1.0

    clas_data = np.array(clas_list)
    reg_data = np.array(reg_list)

    ### Display, if enabled
    if display:
        dr = ImageDraw.Draw(image)

        for p in persons:
            dr.rectangle(p['rect'], outline = 'red')

        for i in positiveAnchors:
            dr.rectangle(anchors[i]['rect'], outline = 'green')

        image.show()

    return pixels_data, clas_data, reg_data


files = list_files(0, 0)
annotations = parse_annotations()

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
def RPN(X, pattern_anchors, training = False):
    k = pattern_anchors.num # Number of anchors

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

### Declare input & output ###
pixels_placeholder = tf.placeholder("float", [None, 480, 640, 3]) # 640x480 images, RGB (depth 3)
clas_placeholder = tf.placeholder("float", [None, 2])
reg_placeholder = tf.placeholder("float", [None, 4])

### Shared CNN ###
shared_cnn = VGG16D(pixels_placeholder)

### RPN ###
# Declare anchors
class Anchors:
    def __init__(self, scales, width_to_height_ratios):
        self.num = len(scales) * len(width_to_height_ratios)

        heights = []
        widths = []
        for s in scales:
            s = float(s)
            # Desired scales are the square of what is passed to the function
            s = pow(s, 2)
            for r in width_to_height_ratios:
                r = float(r)
                h = sqrt(s/r)
                w = r * h
                heights.append(h)
                widths.append(w)

        self.height_list = heights
        self.width_list = widths
        self.heights = tf.constant(heights)
        self.widths = tf.constant(widths)

pattern_anchors = Anchors([50, 75, 100], [1/1.8, 1/2.4, 1/3.0])

shared_rpn, clas_rpn, reg_rpn = RPN(shared_cnn, pattern_anchors)

# Get positive & negative truth
clas_truth = tf.squeeze(tf.slice(clas_placeholder, [0, 1], [-1, 1]))
positive_ratio = tf.div(tf.reduce_sum(clas_truth), MINIBATCH_SIZE)

# Accuracy function
clas_guess = tf.argmax(clas_rpn, 1)
true_positives = tf.slice(tf.mul(clas_placeholder, clas_rpn), [0, 1], [-1, 1])
clas_accuracy = tf.div(tf.reduce_sum(true_positives), tf.reduce_sum(clas_truth))

# Declare loss functions
clas_loss_1 = tf.nn.softmax_cross_entropy_with_logits(clas_rpn, clas_placeholder)
clas_useful = tf.reduce_sum(clas_placeholder, reduction_indices = 1)
clas_loss = tf.reduce_sum(tf.mul(clas_loss_1, clas_useful))
clas_loss = tf.div(clas_loss, MINIBATCH_SIZE) # Normalization
clas_positive_loss = tf.mul(clas_loss, clas_truth)
clas_positive_loss = tf.reduce_sum(clas_positive_loss)
clas_positive_loss = tf.div(clas_positive_loss, MINIBATCH_SIZE) # Normalization

reg_loss = tf.sub(reg_rpn, reg_placeholder) # The regression loss takes more steps
reg_loss = tf.mul(reg_loss, reg_loss)
reg_loss = tf.reduce_sum(reg_loss, reduction_indices = 1)
reg_loss = tf.mul(reg_loss, clas_truth)
reg_loss = tf.reduce_mean(reg_loss) # Normalization
lambda_ = tf.Variable(10.0, trainable = False, name = 'lambda')
reg_loss = tf.mul(reg_loss, lambda_) # Scaling

rpn_loss = tf.add(clas_loss, reg_loss)

global_step = tf.Variable(0, trainable = False, name = 'global_step')
learning_rate = tf.train.exponential_decay(
    0.01,               # Base learning rate.
    global_step * MINIBATCH_SIZE,  # Current index into the dataset.
    10000,              # Decay step.
    0.95,               # Decay rate.
    staircase=True)

epoch = tf.Variable(0.0, trainable = False, name = 'epoch')
increment_epoch = epoch.assign_add(1.0)

# Use simple momentum for the optimization.
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(rpn_loss,
                                                   global_step = global_step)

test_step = [tf.argmax(clas_rpn, 1), reg_rpn]

def create_summaries():
    epoch_summary = tf.scalar_summary('epoch', epoch)
    positive_ratio_summary = tf.scalar_summary('positive_ratio', positive_ratio)
    clas_accuracy_summary = tf.scalar_summary('clas_accuracy', clas_accuracy)
    clas_summary = tf.scalar_summary('clas_loss', clas_loss)
    clas_positive_summary = tf.scalar_summary('clas_positive_loss', clas_positive_loss)
    reg_summary = tf.scalar_summary('reg_loss', reg_loss)
    rpn_summary = tf.scalar_summary('rpn_loss', rpn_loss)
    learning_rate_summary = tf.scalar_summary('learning_rate', learning_rate)

    return [epoch_summary, positive_ratio_summary, clas_accuracy_summary, clas_summary, clas_positive_summary, reg_summary, rpn_summary, learning_rate_summary]

def create_train_summaries():
    with tf.name_scope('train'):
        return tf.merge_summary(create_summaries())

def create_test_summaries():
    with tf.name_scope('test'):
        return tf.merge_summary(create_summaries())

train_summaries = create_train_summaries()
test_summaries = create_test_summaries()

def evaluate_test_results(set_number, seq_number, frame_number, annotations, pattern_anchors, clas_results, reg_results, display = False):
    # NOTE: argmax should have already been applied to clas_results

    image = Image.open(DATASET + '/images/set{:02d}/V{:03d}.seq/{}.jpg'.format(set_number, seq_number, frame_number))

    ### Compute person rectangles
    persons = []
    try:
        objects = annotations['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]

        for o in objects:
            pos = o['pos']
            if o['lbl'] in ['person', 'people']:
                rect = (pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3])
                persons.append({'rect': rect, 'used': False})

    except KeyError as e:
        print('ERROR: Missing key {}'.format(e))


    ### Compute anchor rectangles
    anchors = create_anchors(pattern_anchors, image.size[1], image.size[0])

    ### For positive guesses, compute coordinates
    positiveAnchors = np.where(clas_results == 1)[0]
    guesses = []
    for i in positiveAnchors:
        # Regression values
        xAnchor = anchors[i]['rect'][0]
        yAnchor = anchors[i]['rect'][1]
        wAnchor = anchors[i]['rect'][2] - xAnchor
        hAnchor = anchors[i]['rect'][3] - yAnchor

        tx = reg_results[i][0]
        ty = reg_results[i][1]
        tw = reg_results[i][2]
        th = reg_results[i][3]

        xGuess = tx * wAnchor + xAnchor
        yGuess = ty * hAnchor + yAnchor
        wGuess = wAnchor * exp(tw)
        hGuess = hAnchor * exp(th)

        rect = (xGuess, yGuess, xGuess + wGuess, yGuess + hGuess)
        guesses.append({'rect': rect})

    if display:
        dr = ImageDraw.Draw(image)

        for p in persons:
            dr.rectangle(p['rect'], outline = 'red')

        for g in guesses:
            dr.rectangle(g['rect'], outline = 'green')

        image.show()

def create_feed_dict(set_number, seq_number, frame_number, annotations, pattern_anchors):
    pixels_data, clas_data, reg_data = create_minibatch(set_number, seq_number, frame_number, annotations, pattern_anchors, display = False)

    return {pixels_placeholder: pixels_data,
            clas_placeholder: clas_data,
            reg_placeholder: reg_data}

def create_dataset(sets):
    dataset = []
    for set_number in sets:
        seq_number = 0
        for seq in sorted([f for f in os.listdir(DATASET + '/images/set{:02d}'.format(set_number))]):
            frame_number = 0
            count = 0
            for image in sorted([f for f in os.listdir(DATASET + '/images/set{:02d}'.format(set_number) + '/' + seq)]):
                if frame_number % IMAGE_MODULO == 0:
                    dataset.append((set_number, seq_number, frame_number))
                    count += 1
                frame_number += 1
            seq_number += 1

    return dataset

def create_training_testing_set():
    training_set = create_dataset(TRAIN_SETS)
    testing_set = create_dataset(TEST_SETS)

    return training_set, testing_set

with tf.Session() as sess:
    # Initialize variables
    tf.initialize_all_variables().run()

    # Create training/testing datasets
    training_set, testing_set = create_training_testing_set()

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    train_writer = tf.train.SummaryWriter('log/train', sess.graph, flush_secs = 10)
    test_writer = tf.train.SummaryWriter('log/test', flush_secs = 10)

    for i in range(10): # Epochs
        # Do one pass of the whole training_set
        for frame in training_set:
            results = sess.run([train_step, train_summaries], feed_dict = create_feed_dict(frame[0], frame[1], frame[2], annotations, pattern_anchors))
            train_writer.add_summary(results[1], global_step = tf.train.global_step(sess, global_step))

        if i != 0 and i % 5 == 0:
            # Do one pass of the whole testing set
            for frame in testing_set:
                [summary_results, clas_results, reg_results] = sess.run([test_summaries] + test_step, feed_dict = create_feed_dict(frame[0], frame[1], frame[2], annotations, pattern_anchors))
                test_writer.add_summary(summary_results, global_step = tf.train.global_step(sess, global_step))

        sess.run([increment_epoch])

    coord.request_stop()
    coord.join(threads)
