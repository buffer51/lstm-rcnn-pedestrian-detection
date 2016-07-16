#!/usr/bin/env python

import os, glob, random
from math import ceil, floor, sqrt, exp

import numpy as np
from PIL import Image, ImageDraw

import prepare

class Anchors:
    def __init__(self, scales, width_to_height_ratios, image_height, image_width):
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

        self.anchors = prepare.create_anchors_for_frame(self, image_height, image_width)[2]

    def unparametrize(self, anchor_id, parameters):
        # Convert back to real x, y, width & height from learned parameters
        t_x = parameters[0]
        t_y = parameters[1]
        t_w = parameters[2]
        t_h = parameters[3]

        xAnchor = self.anchors[anchor_id]['rect'][0]
        yAnchor = self.anchors[anchor_id]['rect'][1]
        wAnchor = self.anchors[anchor_id]['rect'][2] - xAnchor
        hAnchor = self.anchors[anchor_id]['rect'][3] - yAnchor

        x = (wAnchor * t_x) + xAnchor
        y = (hAnchor * t_y) + yAnchor
        w = wAnchor * exp(t_w)
        h = hAnchor * exp(t_h)

        return (x, y, w, h)

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

    # NMS settings
    NMS_IOU_THRESHOLD = 0.7
    NMS_TOP_N = 2000 # Kept after NMS

    def __init__(self, dataset_location = 'caltech-dataset/dataset'):
        self.dataset_location = dataset_location

        # Define anchors
        self.pattern_anchors = Anchors([30, 55, 80], [0.31, 0.41, 0.51], 480, 640)

        print('Generating training & testing sets...')
        self.training = []
        self.testing = []
        for set_number in range(10 + 1):
            if set_number <= 5:
                self.training += self.parse_set(set_number)
            else:
                self.testing += self.parse_set(set_number, skip_frames = True)

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

    def parse_sequence(self, set_number, seq_number, skip_frames):
        folder = self.dataset_location + '/images/set{:02d}/V{:03d}.seq'.format(set_number, seq_number)

        num_total_frames = len(glob.glob(folder + '/*.jpg'))

        if skip_frames:
            num_frames = int(floor(float(num_total_frames) / float(Caltech.FRAME_MODULO)))
            return [(set_number, seq_number, Caltech.FRAME_MODULO * i + Caltech.FRAME_MODULO - 1) for i in range(num_frames)]
        else:
            return [(set_number, seq_number, i) for i in range(num_total_frames)]

    def parse_set(self, set_number, skip_frames = False):
        folder = self.dataset_location + '/images/set{:02d}'.format(set_number)
        num_sequences = len([f for f in os.listdir(folder) if os.path.isdir(folder + '/' + f)])

        tuples = []
        for seq_number in range(num_sequences):
            tuples += self.parse_sequence(set_number, seq_number, skip_frames)

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
        data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(*self.training[0]))

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
        minibatch_used = self.testing[self.test_minibatch]
        data = np.load(self.dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(*minibatch_used))
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

        return minibatch_used, input_data, clas_data, reg_data

    def get_test_minibatch(self, input_placeholder, clas_placeholder, reg_placeholder):
        if Caltech.FRAMES_PER_TEST_MINIBATCH != 1:
            print('ERROR: Not implemented')
            exit(1)

        minibatch_used, input_data, clas_data, reg_data = self.get_test_frame()

        feed_dict = {
            input_placeholder: input_data,
            clas_placeholder: clas_data,
            reg_placeholder: reg_data
        }

        return minibatch_used, feed_dict

    def NMS(self, guess_boxes, guess_scores):
        index = np.argsort(guess_scores[:, 1])[::-1] # Decreasing order with [::-1]

        final_boxes = np.zeros((0, 4))
        final_scores = np.zeros((0, 1))
        while len(index) > 0 and final_scores.shape[0] < Caltech.NMS_TOP_N:
            final_boxes = np.vstack([final_boxes, guess_boxes[index[0]]])
            final_scores = np.vstack([final_scores, guess_scores[index[0]][1]])

            to_keep = []
            for i in range(1, len(index)):
                IoU = prepare.IoU_wh(guess_boxes[index[0]], guess_boxes[index[i]])

                if IoU < Caltech.NMS_IOU_THRESHOLD:
                    to_keep.append(i)

            index = index[to_keep]

        return final_boxes, final_scores

    def create_result(self, set_number, seq_number, frame_number, clas_examples, clas_guess, clas_prob, rpn_guess):
        guess_index = np.where(np.multiply(clas_guess, clas_examples) == 1)[0]
        guess_boxes = np.zeros((len(guess_index), 4))
        guess_scores = clas_prob[guess_index]
        for i in range(len(guess_index)):
            guess_boxes[i] = self.pattern_anchors.unparametrize(guess_index[i], rpn_guess[guess_index[i]])

        guess_boxes, guess_scores = self.NMS(guess_boxes, guess_scores)

        if not os.path.isdir(self.dataset_location + '/data/res/LSTM-RCNN/set{:02d}/V{:03d}'.format(set_number, seq_number)):
            os.makedirs(self.dataset_location + '/data/res/LSTM-RCNN/set{:02d}/V{:03d}'.format(set_number, seq_number))

        with open(self.dataset_location + '/data/res/LSTM-RCNN/set{:02d}/V{:03d}/I{:05d}.txt'.format(set_number, seq_number, frame_number), 'w') as f:
            for box, score in zip(guess_boxes, guess_scores):
                f.write('{}, {}, {}, {}, {}\n'.format(box[0], box[1], box[2], box[3], score[0]))

    def show_image_with_boxes(self, set_number, seq_number, frame_number, truth_boxes, guess_boxes):
        image = Image.open(self.dataset_location + '/images/set{:02d}/V{:03d}.seq/{}.jpg'.format(set_number, seq_number, frame_number))

        dr = ImageDraw.Draw(image)
        for box in truth_boxes:
            dr.rectangle((box[0], box[1], box[0]+box[2], box[1]+box[3]), outline = 'blue')
        for box in guess_boxes:
            dr.rectangle((box[0], box[1], box[0]+box[2], box[1]+box[3]), outline = 'red')

        image.show()


if __name__ == '__main__':
    # This will automatically prepare the dataset
    caltech_dataset = Caltech('dataset')
    caltech_dataset.prepare()
