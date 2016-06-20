#!/usr/bin/env python

import os

def statsSeq(folder):
    images = [f for f in os.listdir(folder) if os.path.isfile(folder + '/' + f)]

    numImages = len(images)
    numImages1FPS = numImages / 30

    return numImages, numImages1FPS

def statsSet(folder):
    sequences = [f for f in os.listdir(folder) if os.path.isdir(folder + '/' + f)]

    numImages = 0
    numImages1FPS = 0
    for seq in sequences:
        (a, b) = statsSeq(folder + '/' + seq)
        numImages += a
        numImages1FPS += b

    return numImages, numImages1FPS

def statsTrainset(folder):
    numImages = 0
    numImages1FPS = 0

    for s in range(0, 6):
        (a, b) = statsSet(folder + '/set{:02d}'.format(s))
        numImages += a
        numImages1FPS += b

    print('Training set:')
    print('Total images: {}'.format(numImages))
    print('Total images (@ 1FPS): {}'.format(numImages1FPS))

def statsTestset(folder):
    numImages = 0
    numImages1FPS = 0

    for s in range(6, 11):
        (a, b) = statsSet(folder + '/set{:02d}'.format(s))
        numImages += a
        numImages1FPS += b

    print('Testing set:')
    print('Total images: {}'.format(numImages))
    print('Total images (@ 1FPS): {}'.format(numImages1FPS))

if __name__ == '__main__':
    statsTrainset('caltech-dataset/images')
    statsTestset('caltech-dataset/images')
