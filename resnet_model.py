#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:35:20 2017

@author: xueyunzhe
"""

import tensorflow as tf

w_init = tf.contrib.layers.variance_scaling_initializer()
w_regular = tf.contrib.layers.l2_regularizer(0.0002)
act_fn = tf.nn.relu
train=True

def conv_factory(data, num_filter, kernel, stride, pad, act_fn=tf.nn.relu, layers=0, conv_type=0, train=True, name=None):
    if conv_type == 0:
        conv = tf.contrib.layers.conv2d(
            inputs=data,
            num_outputs=num_filter,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            activation_fn=tf.identity,
            weights_initializer=w_init,
            weights_regularizer=w_regular,
            biases_initializer=None,
#            trainable=train,
            variables_collections=['weights'],
            outputs_collections=['output'],
            data_format =  "NCHW",
            scope=name,
            )
        bn = tf.contrib.layers.batch_norm(
            inputs=conv,
            is_training=train,
            decay=0.9,
            updates_collections=None,
#            variables_collections=["batch_norm_non_trainable_variables_collection"],
            outputs_collections=None,
            scale=True,
            epsilon=2e-5,
            fused=True,
            data_format =  "NCHW",
            scope=name,            
            )
        act = act_fn(bn)
        return act
    
    elif conv_type == 1:
        conv = tf.contrib.layers.conv2d(
            inputs=data,
            num_outputs=num_filter,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            activation_fn=tf.identity,
            weights_initializer=w_init,
            weights_regularizer=w_regular,
            biases_initializer=None,
#            trainable=train,
            variables_collections=['weights'],
            outputs_collections=['output'],
            data_format =  "NCHW",
            scope=name,
            )
        bn = tf.contrib.layers.batch_norm(
            inputs=conv,
            is_training=train,
            decay=0.9,
            scale=True,
            updates_collections=None,
#            variables_collections=["batch_norm_non_trainable_variables_collection"],
            outputs_collections=None,
            epsilon=2e-5,
            fused=True,
            data_format =  "NCHW",
            scope=name,            
            )
        return bn        

def residual_factory(data, num_filter, dim_match, act_fn=tf.nn.relu, layers=0, train=True):
    if dim_match is True:
        identity_data = data
        conv1 = conv_factory(data=data,
                             num_filter=num_filter,
                             kernel=3,
                             stride=1,
                             pad='SAME',
                             act_fn= act_fn,
                             layers=layers,
                             conv_type=0,
                             train=train,
                             )
        conv2 = conv_factory(data=conv1,
                             num_filter=num_filter,
                             kernel=3,
                             stride=1,
                             pad='SAME',
                             layers=layers+1,
                             conv_type=1,
                             train=train,
                             )
        if identity_data.shape[1] < conv2.shape[1]:
            identity_data = conv_factory(data=identity_data,
                     num_filter=num_filter,
                     kernel=1,
                     stride=1,
                     pad='SAME',
                     conv_type=1,
                     train=train,
                     )
        new_data = tf.add(conv2, identity_data)
        act = act_fn(new_data)
        return act
    
    else:
        conv1 = conv_factory(data=data,
                             num_filter=num_filter,
                             kernel=3,
                             stride=2,
                             pad='SAME',
                             act_fn= act_fn,
                             layers=layers,                             
                             conv_type=0,
                             train=train,
                             )        
        conv2 = conv_factory(data=conv1,
                             num_filter=num_filter,
                             kernel=3,
                             stride=1,
                             pad='SAME',
                             layers=layers+1,                             
                             conv_type=1,
                             train=train,
                             )
        project_data = conv_factory(data=data,
                             num_filter=num_filter,
                             kernel=2,
                             stride=2,
                             pad='SAME',
                             conv_type=1,
                             train=train,
                             )  
        new_data = tf.add(project_data, conv2)
        act = act_fn(new_data)
        return act
    
def residual_net(data, n, train, k=1):
    with tf.name_scope('block1'):
        for i in range(n):
            data = residual_factory(data=data, num_filter=16*k, dim_match=True, train=train)
    
    with tf.name_scope('block2'):    
        for i in range(n):
            if i==0:
                data = residual_factory(data=data, num_filter=32*k, dim_match=False, train=train)
            else:
                data = residual_factory(data=data, num_filter=32*k, dim_match=True, train=train)

    with tf.name_scope('block3'):         
        for i in range(n):
            if i==0:
                data = residual_factory(data=data, num_filter=64*k, dim_match=False, train=train)
            else:
                data = residual_factory(data=data, num_filter=64*k, dim_match=True, train=train)
        
    return data

def bk_block(data, num_classes):
    blocks = []
    for i in range(num_classes):
        layer1 = conv_factory(data, 128, 3, 2, 'SAME', name='bk1_'+str(i))
        layer2 = conv_factory(layer1, 16, 3, 2, 'SAME', name='bk2_'+str(i))
        layer3 = conv_factory(layer2, 1, 2, 2, 'VALID', name='bk3_'+str(i))
        blocks.append(layer3)
    return tf.concat(blocks, axis=1)

def bk_block2(data, num_classes):
    return conv_factory(data, num_classes, 8, 1, 'VALID')

def Resnet(data, n=3, train=True, k=1,num_classes=10):
    data = tf.contrib.layers.batch_norm(
            inputs=data,
            is_training=train,
            decay=0.9,
            scale=True,
            updates_collections=None,
#            variables_collections=["batch_norm_non_trainable_variables_collection"],
            outputs_collections=None,
            epsilon=2e-5,
            fused=True,
            data_format =  "NCHW",
#            scope='#'+layers+'_BN',            
            )
    conv = conv_factory(data=data,
                             num_filter=16,
                             kernel=3,
                             stride=1,
                             pad='SAME',
                             act_fn= act_fn,
                             layers=0,
                             conv_type=0,
                             train=train
                             )
    r = residual_net(conv, n, train = train, k=k)
    pool = tf.contrib.layers.avg_pool2d(
            inputs=r,
            kernel_size=8,
            stride=1,
            padding='VALID',
            data_format =  "NCHW",
            )
    flatten = tf.contrib.layers.flatten(
            inputs=pool,
            )
    logits = tf.contrib.layers.fully_connected(
            inputs=flatten,
            num_outputs=num_classes,
            activation_fn=tf.identity,
            weights_initializer= tf.contrib.layers.variance_scaling_initializer(factor=1.0, uniform=True),
            weights_regularizer=None,
            biases_initializer=tf.constant_initializer(),
#            trainable=train,
            variables_collections=['weights'],
            outputs_collections=None,
            scope='Logits',
            )
#    bk = bk_block(r, num_classes)
#    logits = tf.contrib.layers.flatten(
#            inputs = bk,
#            )
    return logits