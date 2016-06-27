#!/usr/bin/env python

import os, json, time
from math import sqrt, floor, log, exp
from PIL import Image

import numpy as np

### This script parses the input data (images/ & annotations.json)
### to compute both the input data & labels for learning

def load_annotations(dataset_location):
    with open(dataset_location + '/annotations.json') as json_file:
        annotations = json.load(json_file)

    return annotations

### Input / labels functions

def create_input_for_frame(image):
    return np.expand_dims(np.reshape(np.array(image.getdata(), dtype = np.uint8), [image.size[1], image.size[0], 3]), axis = 0) # [?, height, width, RGB]

def create_anchors_for_frame(pattern_anchors, image_height, image_width):
    # This is regarding the sliding window after the convolutional layers
    windowSize = (3.0, 3.0) # (height, width)
    windowStride = (1.0, 1.0) # (height, width)
    padding = (floor(windowSize[0]/2.0), floor(windowSize[1]/2.0))

    # This is to be able to compute image coordinates
    imageStride = (16.0, 16.0) # (height, width)

    anchors = []
    xs = [x for x in range(int(-padding[1]), int(image_width/imageStride[1] - padding[1]))]
    ys = [y for y in range(int(-padding[0]), int(image_height/imageStride[0] - padding[0]))]
    for x in xs:
        for y in ys:
            currentWindow = (y, x) # (height, width), top-left coordinates
            currentImage = (currentWindow[0]*imageStride[0], currentWindow[1]*imageStride[1]) # (height, width), top-left coordinates
            currentImageCenter = (currentImage[0] + windowSize[0]*imageStride[0]/2.0, currentImage[1] + windowSize[1]*imageStride[1]/2.0)# (height, width), center coordinates

            for h, w in zip(pattern_anchors.height_list, pattern_anchors.width_list):
                rect = (currentImageCenter[1] - w/2.0, currentImageCenter[0] - h/2.0, currentImageCenter[1] + w/2.0, currentImageCenter[0] + h/2.0) # (x_right, y_top, x_left, y_bottom)
                anchors.append({'rect': rect, 'positive': None, 'person': None})

    return len(ys), len(xs), anchors

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

def create_labels_for_frame(objects, pattern_anchors, image_height, image_width):
    ### Create anchors
    num_anchors_vertically, num_anchors_horizontally, anchors = create_anchors_for_frame(pattern_anchors, image_height, image_width)

    ### Compute person rectangles
    persons = []
    if objects:
        for o in objects:
            pos = o['pos']
            if o['lbl'] in ['person', 'people']:
                rect = (pos[0], pos[1], pos[0]+pos[2], pos[1]+pos[3])
                persons.append({'rect': rect, 'used': False})

    ### Determine positive and negative anchors
    if persons:
        # Compute all IoUs for each (anchor, person)
        IoUs = [[IoU(a['rect'], p['rect']) for p in persons] for a in anchors]

        # Remove anchors that cross the image borders (set their IoUs to 0)
        for i in range(len(anchors)):
            rect = anchors[i]['rect']
            if rect[0] < 0 or rect[1] < 0 or rect[2] >= image_width or rect[3] >= image_height:
                IoUs[i] = [0 for p in persons]

        for i in range(len(anchors)): # For each anchor, find the biggest IoU with a person
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
    else: # Nothing to find
        for i in range(len(anchors)):
            rect = anchors[i]['rect']
            # Set negative examples only for anchors that do not cross the borders
            if rect[0] >= 0 and rect[1] >= 0 and rect[2] < image_width and rect[3] < image_height:
                anchors[i]['positive'] = False

    ### Convert anchors to labels for regression & classification
    clas_data = np.zeros((1, num_anchors_vertically, num_anchors_horizontally, pattern_anchors.num, 2), dtype = np.bool_) # [?, height, width, # of anchors, 2 classes]
    reg_data = np.zeros((1, num_anchors_vertically, num_anchors_horizontally, pattern_anchors.num, 4), dtype = np.float32) # [?, height, width, # of anchors, 4 coordinates]

    for i in range(len(anchors)):
        if anchors[i]['positive'] != None:
            # Retrieve 3-D position of the anchors (in height, width, # of anchors)
            index = i / pattern_anchors.num
            if anchors[i]['positive']:
                clas_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 1] = True

                # Regression values
                xAnchor = anchors[i]['rect'][0]
                yAnchor = anchors[i]['rect'][1]
                wAnchor = anchors[i]['rect'][2] - xAnchor
                hAnchor = anchors[i]['rect'][3] - yAnchor

                xPerson = anchors[i]['person'][0]
                yPerson = anchors[i]['person'][1]
                wPerson = anchors[i]['person'][2] - xPerson
                hPerson = anchors[i]['person'][3] - yPerson

                reg_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 0] = (xPerson - xAnchor) / wAnchor # tx
                reg_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 1] = (yPerson - yAnchor) / hAnchor # ty
                reg_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 2] = log(wPerson / wAnchor) # tw
                reg_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 3] = log(hPerson / hAnchor) # th
            else:
                clas_data[0, index%num_anchors_vertically, index/num_anchors_vertically, i%pattern_anchors.num, 0] = True

    return clas_data, reg_data

def create_input_and_labels_for_frame(dataset_location, set_number, seq_number, frame_number, annotations, pattern_anchors):
    if os.path.isfile(dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(set_number, seq_number, frame_number)):
        return

    image = Image.open(dataset_location + '/images/set{:02d}/V{:03d}.seq/{}.jpg'.format(set_number, seq_number, frame_number))

    # Create input data
    input_data = create_input_for_frame(image)

    # Retrieve objects for that frame in annotations
    try:
        objects = annotations['set{:02d}'.format(set_number)]['V{:03d}'.format(seq_number)]['frames']['{}'.format(frame_number)]
    except KeyError as e:
        objects = None # Simply no objects for that frame

    # Create labels
    clas_data, reg_data = create_labels_for_frame(objects, pattern_anchors, image.size[1], image.size[0])

    # Save everything
    if not os.path.isdir(dataset_location + '/prepared/set{:02d}/V{:03d}.seq'.format(set_number, seq_number)):
        os.makedirs(dataset_location + '/prepared/set{:02d}/V{:03d}.seq'.format(set_number, seq_number))

    np.savez(dataset_location + '/prepared/set{:02d}/V{:03d}.seq/{}.npz'.format(set_number, seq_number, frame_number), input = input_data, clas = clas_data, reg = reg_data)

def prepare_dataset(caltech_dataset):
    # Load annotations
    annotations = load_annotations(caltech_dataset.dataset_location)

    print('Preparing training set...')
    current_set = -1
    for (set_number, seq_number, frame_number) in caltech_dataset.training:
        if current_set != set_number:
            print('set{:02d}'.format(set_number))
            current_set = set_number

        create_input_and_labels_for_frame(caltech_dataset.dataset_location, set_number, seq_number, frame_number, annotations, caltech_dataset.pattern_anchors)

    print('Preparing testing set...')
    for (set_number, seq_number, frame_number) in caltech_dataset.testing:
        if current_set != set_number:
            print('set{:02d}'.format(set_number))
            current_set = set_number

        create_input_and_labels_for_frame(caltech_dataset.dataset_location, set_number, seq_number, frame_number, annotations, caltech_dataset.pattern_anchors)

    print('')
