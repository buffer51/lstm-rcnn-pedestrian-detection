#!/usr/bin/env python

import os, glob, random
from math import floor, sqrt

import numpy as np

import prepare

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
        # self.heights = tf.constant(heights)
        # self.widths = tf.constant(widths)

class Caltech:
    FRAME_MODULO = 30 # Modulo for selecting frames from sequences

    # Minibatch settings
    FRAMES_PER_TRAIN_MINIBATCH = 1 # Number of frames passed in each train train_minibatch
    ANCHORS_PER_TRAIN_FRAME = 256 # Number of total anchors (positive + negative) kept in each frame (for training)
    TRAIN_MINIBATCH_SIZE = FRAMES_PER_TRAIN_MINIBATCH * ANCHORS_PER_TRAIN_FRAME

    FRAMES_PER_TEST_MINIBATCH = 1 # Number of frames passed in each test test_minibatch
    MINIBATCH_FOR_EVAL = 100 # Number of minibatches used to evaluate performance during training

    # Balancing settings
    MINIMUM_POSITIVE_RATIO = 0.4

    def __init__(self, dataset_location = 'caltech-dataset/dataset'):
        self.dataset_location = dataset_location

        # Define anchors
        self.pattern_anchors = Anchors([50, 75, 100], [1/1.8, 1/2.4, 1/3.0])

        print('Generating training & testing sets...')
        self.training = []
        self.testing = []
        for set_number in range(10 + 1):
            if set_number <= 5:
                self.training += self.parse_set(set_number)
            else:
                self.testing += self.parse_set(set_number)

        print('{} frames in training set'.format(len(self.training)))
        print('{} frames in testing set'.format(len(self.testing)))
        print('')

        print('Checking training & testing sets...')
        if (not self.check_dataset(self.training)) or (not self.check_dataset(self.testing)):
            print('Dataset not prepared! Preparing dataset...')
            self.prepare()

            if (not self.check_dataset(self.training)) or (not self.check_dataset(self.testing)):
                print('ERROR: Dataset still not prepared!')
                exit(1)
        print('')

    ### Dataset generation

    def parse_sequence(self, set_number, seq_number):
        folder = self.dataset_location + '/images/set{:02d}/V{:03d}.seq'.format(set_number, seq_number)

        num_total_frames = len(glob.glob(folder + '/*.jpg'))
        num_frames = int(floor(float(num_total_frames) / float(Caltech.FRAME_MODULO)))

        return [(set_number, seq_number, Caltech.FRAME_MODULO * i + Caltech.FRAME_MODULO - 1) for i in range(num_frames)]

    def parse_set(self, set_number):
        folder = self.dataset_location + '/images/set{:02d}'.format(set_number)
        num_sequences = len([f for f in os.listdir(folder) if os.path.isdir(folder + '/' + f)])

        tuples = []
        for seq_number in range(num_sequences):
            tuples += self.parse_sequence(set_number, seq_number)

        return tuples

    ### Dataset checking

    def check_dataset(self, dataset):
        if not os.path.isdir(self.dataset_location + '/prepared'):
            return False

        # Shallow check
        return True

    def prepare(self):
        prepare.prepare_dataset(self)

    ### Dataset usage

    def init(self):
        self.epoch = 0
        self.train_minibatch = 0
        self.eval_minibatch = 0
        self.test_minibatch = 0
        print('Epoch: {}'.format(self.epoch))

        random.shuffle(self.training)
        random.shuffle(self.testing)

        self.num_train_minibatches = int(ceil(float(len(self.training)) / float(Caltech.FRAMES_PER_TRAIN_MINIBATCH)))

        # Read first frame to configure input size
        data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(0, 0, 0))

        input_data = data['input']
        self.input_height = input_data.shape[1]
        self.input_width = input_data.shape[2]

    def get_train_frame(self):
        clas_positive = []
        while len(clas_positive) < int(Caltech.MINIMUM_POSITIVE_RATIO * Caltech.ANCHORS_PER_TRAIN_FRAME):
            data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(*self.training[self.train_minibatch]))

            clas_positive = data['clas_positive']

            self.train_minibatch += 1
            if self.train_minibatch == self.num_train_minibatches:
                self.train_minibatch = 0
                self.epoch += 1
                print('Epoch: {}'.format(self.epoch))

                random.shuffle(self.training)

        input_data = data['input']
        clas_negative = data['clas_negative']
        reg_data = data['reg']

        if len(clas_positive) > Caltech.ANCHORS_PER_TRAIN_FRAME / 2:
            clas_positive = random.sample(clas_positive, Caltech.ANCHORS_PER_TRAIN_FRAME / 2)

        if len(clas_negative) > Caltech.ANCHORS_PER_TRAIN_FRAME - len(clas_positive):
            clas_negative = random.sample(clas_negative, Caltech.ANCHORS_PER_TRAIN_FRAME - len(clas_positive))

        clas_data = np.zeros(reg_data.shape[:2] + (2,), dtype = np.float32)
        if len(clas_negative) > 0:
            clas_data[0][clas_negative] = [1.0, 0.0]
        if len(clas_positive) > 0:
            clas_data[0][clas_positive] = [0.0, 1.0]

        return input_data, clas_data, reg_data

    def get_train_minibatch(self, input_placeholder, clas_placeholder, reg_placeholder):
        if Caltech.FRAMES_PER_TRAIN_MINIBATCH != 1:
            print('ERROR: Not implemented')
            exit(1)

        input_data, clas_data, reg_data = self.get_train_frame()

        return {
            input_placeholder: input_data,
            clas_placeholder: clas_data,
            reg_placeholder: reg_data
        }

    def is_eval_minibatch_left(self):
        if self.eval_minibatch == Caltech.MINIBATCH_FOR_EVAL:
            self.eval_minibatch = 0
            return False

        return True

    def get_eval_frame(self):
        data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(*self.testing[self.eval_minibatch]))
        self.eval_minibatch += 1

        input_data = data['input']
        clas_negative = data['clas_negative']
        clas_positive = data['clas_positive']
        reg_data = data['reg']

        clas_data = np.zeros(reg_data.shape[:2] + (2,), dtype = np.float32)
        if len(clas_negative) > 0:
            clas_data[0][clas_negative] = [1.0, 0.0]
        if len(clas_positive) > 0:
            clas_data[0][clas_positive] = [0.0, 1.0]

        return input_data, clas_data, reg_data

    def get_eval_minibatch(self, input_placeholder, clas_placeholder, reg_placeholder):
        if Caltech.FRAMES_PER_TEST_MINIBATCH != 1:
            print('ERROR: Not implemented')
            exit(1)

        input_data, clas_data, reg_data = self.get_eval_frame()

        return {
            input_placeholder: input_data,
            clas_placeholder: clas_data,
            reg_placeholder: reg_data
        }

    def is_test_minibatch_left(self):
        if self.test_minibatch == len(self.testing):
            self.test_minibatch = 0
            return False

        return True

    def get_test_frame(self):
        data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(*self.testing[self.test_minibatch]))
        self.test_minibatch += 1

        input_data = data['input']
        clas_negative = data['clas_negative']
        clas_positive = data['clas_positive']
        reg_data = data['reg']

        clas_data = np.zeros(reg_data.shape[:2] + (2,), dtype = np.float32)
        if len(clas_negative) > 0:
            clas_data[0][clas_negative] = [1.0, 0.0]
        if len(clas_positive) > 0:
            clas_data[0][clas_positive] = [0.0, 1.0]

        return input_data, clas_data, reg_data

    def get_test_minibatch(self, input_placeholder, clas_placeholder, reg_placeholder):
        if Caltech.FRAMES_PER_TEST_MINIBATCH != 1:
            print('ERROR: Not implemented')
            exit(1)

        input_data, clas_data, reg_data = self.get_test_frame()

        return {
            input_placeholder: input_data,
            clas_placeholder: clas_data,
            reg_placeholder: reg_data
        }

if __name__ == '__main__':
    # This will automatically prepare the dataset
    caltech_dataset = Caltech('dataset')
    caltech_dataset.prepare()
