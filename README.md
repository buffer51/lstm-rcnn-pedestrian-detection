# lstm-rcnn-pedestrian-detection

This repository contains the code supporting my research internship at UC Berkeley
for the industrial chair Drive for All (MINES Paristech) and the Berkeley Deep Drive consortium.

My research subject was Pedestrian Detection using Deep Learning methods.
Although I didn't have time to get extensive results because of time constraints,
the code is in a working state. See below for its structure.

## Approach

The approach I explored is primarly based on the
[Faster R-CNN](http://arxiv.org/pdf/1506.01497.pdf) framework
(see [here](https://github.com/rbgirshick/py-faster-rcnn)
for Python code using **caffe**).

In their unified neural network, a first part called Region
Proposal Network generates proposals for areas of interest,
then a classifier generates labels for them.

When focusing on Pedestrian Detection, the classifier becomes redundant
because generating proposals is already classifying between 'background'
and 'pedestrian'. In this work, I attempt to remove the classifier and train
a deeper RPN to cope with the change. This has since been studied in
[another Microsoft Research paper](http://arxiv.org/abs/1607.07032),
yielding good results even with only the RPN.

The other aspect I had planned to investigate was using LSTM units
at the output of RPN to include context over time, hopefully improving
results. This is not present in this code.

## Dataset

My experiments are done on the
[Caltech Pedestrian Dataset](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/).
It is a dataset made of 250k images with pedestrian annoted, freely available.
More information can be found on their website.

## Code Structure

Code related to handling the dataset (generating input & output from images and annotations)
can be found under `caltech-dataset/`. This include Python code as well as several repositories
I used for reading the dataset (see `caltech-dataset/README.md`).

The `vgg16/` folder contains code to convert a pretrained model of the Convolutional
Neural Network [VGG16](http://arxiv.org/pdf/1409.1556v6.pdf),
from a **caffe** format to **TensorFlow**.

Finally, the main script setting up the neural network, training & logging is `region_proposal.py`.
It can be used train a model with parameters in `caltech-dataset/caltech.py`,
saving every few epochs. It can also generate results in the expected format for the Caltech Dataset
**MATLAB** code evaluation from the trained model.
