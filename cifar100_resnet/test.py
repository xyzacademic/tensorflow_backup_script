#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 09:24:19 2017

@author: xueyunzhe
"""
import numpy as np
import tensorflow as tf


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

train_data = np.load('cifar10_traindata.npy')#uint8
train_data = np.swapaxes(train_data,1,2)
train_data = np.swapaxes(train_data,2,3)
test_data = np.load('cifar10_testdata.npy')#int64
train_label = np.load('cifar10_trainlabel.npy')
test_label = np.load('cifar10_testlabel.npy')
test_data = np.swapaxes(test_data,1,2)
test_data = np.swapaxes(test_data, 2,3)

def save_to_records(save_path, images, labels):
    writer = tf.python_io.TFRecordWriter(save_path)
    for i in range(images.shape[0]):
        image_raw = images[i].tostring()
        example = tf.train.Example(
                features=tf.train.Features(
                        feature={
                                'image_raw': _bytes_feature(image_raw),
                                'label': _int64_feature(int(labels[i]))
                                }))
        writer.write(example.SerializeToString())
        
save_to_records('train.tfrecords', train_data, train_label)
save_to_records('test.tfrecords', test_data, test_label)
#125.306
#122.95
#113.86