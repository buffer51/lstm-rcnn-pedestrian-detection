#!/usr/bin/env python

import os
os.environ["GLOG_minloglevel"] = "2"

import caffe
import tensorflow as tf

if __name__ == '__main__':
    net_caffe = caffe.Net('tensorflow-vgg16/VGG_2014_16.prototxt', 'tensorflow-vgg16/VGG_ILSVRC_16_layers.caffemodel', caffe.TEST)

    caffe_layers = {}
    for i, layer in enumerate(net_caffe.layers):
        layer_name = net_caffe._layer_names[i]
        caffe_layers[layer_name] = layer

    def caffe_weights(layer_name):
        f = caffe_layers[layer_name].blobs[0].data.transpose((2, 3, 1, 0))
        if name == 'conv1_1':
            f = f[:, :, ::-1, :]
        return f

    def caffe_biases(layer_name):
        layer = caffe_layers[layer_name]
        return layer.blobs[1].data

    originalNames = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']
    for i in range(len(originalNames)):
        name = originalNames[i]

        tf.Variable(caffe_weights(name), name = 'VGG16D/layer{}/weights'.format(i+1))
        tf.Variable(caffe_biases(name), name = 'VGG16D/layer{}/biases'.format(i+1))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        save_path = saver.save(sess, 'VGG16D.ckpt')
        os.remove('checkpoint')
        os.remove('VGG16D.ckpt.meta')
