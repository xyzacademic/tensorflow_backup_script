#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 11:32:49 2017

@author: xueyunzhe
"""

import tensorflow as tf
import numpy as np

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def flip_img(image, random=True):
    image = tf.image.random_flip_left_right(image)
#    image = tf.image.random_flip_up_down(image)
    return image

def sub_mean(image):

    tf.subtract(image[:,:,0],tf.constant([125.306]))
    tf.subtract(image[:,:,1],tf.constant([122.95]))
    tf.subtract(image[:,:,0],tf.constant([113.86]))      

    return image

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
        

def read_tfrecord(load_path, image_shape=[32,32,3], image_dtype=tf.uint8, shuffle=False):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(load_path, None, shuffle=shuffle)  
    _, record_value = reader.read(filename_queue)
    features = tf.parse_single_example(record_value,
                                       {
                                        'image_raw': tf.FixedLenFeature([], tf.string),
                                        'label': tf.FixedLenFeature([], tf.int64),
                                               })        
    image = tf.decode_raw(features['image_raw'],image_dtype)#point
    image = tf.reshape(image, image_shape)
    image = tf.cast(image, tf.float32)
    label = features['label']
    
    return image, label

def softmax_onehot(logits, labels, ):
    label = tf.contrib.layers.one_hot_encoding(labels, 5, scope='OneHot')
    cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=label,
            logits=logits,
            weights=1.0,
            label_smoothing=0,
            scope='CrossEntropy',
            loss_collection=tf.GraphKeys.LOSSES,
            )

    return cross_entropy

def softmax_sparse(logits, labels):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
            labels=labels,
            logits=logits,
            weights=1.0,
            scope='CrossEntropy',
            loss_collection=tf.GraphKeys.LOSSES,
            )
    return cross_entropy

def accuracy(logits, labels,):
#    label = tf.contrib.layers.one_hot_encoding(labels, 5, scope='OneHot')
    accuracy = tf.contrib.metrics.accuracy(
            predictions=tf.argmax(logits, 1),
            labels=labels,
            weights=None,

            )
    return accuracy

def top_in_k(logits, labels, k): #top k accuracy
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels, k), tf.float32))#top in ke

 
def make_train_batch(data_path, batch_size, flip=True, shuffle=False):
    with tf.device('/cpu:0'):
        train_data, train_label = read_tfrecord(data_path, shuffle=shuffle)
        train_data = tf.image.resize_image_with_crop_or_pad(train_data, 40, 40)
        train_data = tf.random_crop(train_data, [32,32,3])
        if flip is True:
            train_data = tf.image.random_flip_left_right(train_data)
#        train_data = tf.image.per_image_standardization(train_data)
        train_data = tf.transpose(train_data, [2,0,1])
        train_data_batch, train_label_batch = tf.train.shuffle_batch(
                [train_data, train_label], batch_size=batch_size,
                num_threads=8, 
                capacity=200*batch_size,
                min_after_dequeue=8*batch_size,
                )
        
        return train_data_batch, train_label_batch

def make_test_batch(data_path, batch_size, shuffle=False):
    with tf.device('/cpu:0'):
        test_data, test_label = read_tfrecord(data_path, shuffle=shuffle)
#        test_data = tf.image.per_image_standardization(test_data)
        test_data = tf.transpose(test_data, [2,0,1])
        test_data_batch, test_label_batch = tf.train.batch(
                [test_data, test_label], batch_size=batch_size,
                num_threads=8, capacity=100*batch_size, 
                )
        return test_data_batch, test_label_batch
    
    
