#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 17:15:02 2017

@author: wowjoy
"""

#==============================================================================
# Import module
#==============================================================================
import tensorflow as tf
import model_tool
from resnet_model import Resnet, bk_block
import shutil
import time
import os

#==============================================================================
# Global Flags and related preparation
#==============================================================================
batch_size =128
n_epoch = 200
train_iters = int(50000/batch_size)
val_iters = int(10000/100)
train_data = ['train.tfrecords']
test_data = ['test.tfrecords']

tf.set_random_seed(2017)


#==============================================================================
# Input and global variables
#==============================================================================

with tf.name_scope('Input'):
    phase_train = tf.placeholder(bool, name='Phase_train')#Train mode or validation mode
    global_step = tf.Variable(0, trainable=False, name='global_step')
    with tf.name_scope('Batch_input'):
        train_data_batch, train_label_batch = model_tool.make_train_batch(train_data, batch_size, flip=True, shuffle=True)
        test_data_batch, test_label_batch = model_tool.make_test_batch(test_data, 100, shuffle=False)
        image_batch, label_batch = tf.cond(phase_train,
                                   lambda:(train_data_batch, train_label_batch),#Return when Phase_train is True
                                   lambda:(test_data_batch, test_label_batch)#Return when Phase_train is False
                                   )


#==============================================================================
# Network interface
#==============================================================================
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

r = Resnet(image_batch, n = 3, train=False, k=1, num_classes=100)# The number of layers is n*6+2, k is the multiply factor of filters

saver = tf.train.Saver()

bk = bk_block(r, num_classes=100)
logits = tf.contrib.layers.flatten(
        inputs = bk,
        )

#==============================================================================
# Loss 
#==============================================================================
with tf.name_scope('Loss'):
    loss_ = model_tool.softmax_sparse(logits=logits, labels=label_batch)#Loss of softmatx layer
    loss_re = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)#Loss of regularization
    loss = tf.add_n([loss_]+loss_re)# Sum of two losses
#    tf.summary.scalar('loss', loss)
    

#==============================================================================
# Evaluation method
#==============================================================================
with tf.name_scope('Evaluation'):
    accuracy = model_tool.accuracy(logits=logits, labels=label_batch)#Common accuracy
#    tf.summary.scalar('accuracy', accuracy)   

#==============================================================================
# Optimizer define
#==============================================================================
with tf.name_scope('Train_op'):
    lr = tf.train.exponential_decay(0.1,
                          global_step-78000,
                          50000/batch_size*10,
                          0.1,
                          staircase=True)
      
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9, use_nesterov=True)
    fine_tune = [_ for _ in tf.trainable_variables() if 'bk' in _.name]
    
    
    grads = optimizer.compute_gradients(loss, 
                                        var_list=fine_tune,
                                        )
    train_op = optimizer.apply_gradients(grads, global_step)    
    grad_list = []
    
    for grad, var in grads:
        if grad is not None and ('weights' in var.op.name):
            grad_list.append(tf.summary.histogram(var.op.name +'/gradients', grad))
            
#==============================================================================
# Monitor
#==============================================================================
base_list = [tf.summary.scalar('loss', loss), tf.summary.scalar('accuracy', accuracy)]
advanced_list = base_list + grad_list 

merge1 = tf.summary.merge(advanced_list)
merge2 = tf.summary.merge(base_list)

#==============================================================================
# Save and summary
#==============================================================================

#try:
#    shutil.rmtree('logs/wide_resnet3')
#except:
#    pass


train_writer = tf.summary.FileWriter(logdir='logs/wide_resnet3/train', graph=sess.graph)
val_writer = tf.summary.FileWriter(logdir='logs/wide_resnet3/val')
#merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess.run(init) 
saver.restore(sess,'logs/wide_resnet3/model-78000')
#==============================================================================
# Train and test
#==============================================================================
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

def train_val(train_iters):
    for j in range(train_iters):

        _, train_summary = sess.run([train_op, merge1], feed_dict={phase_train:True})
        train_writer.add_summary(train_summary, sess.run(global_step))
        val_summary = sess.run(merge2, feed_dict={phase_train:False})
        val_writer.add_summary(val_summary, sess.run(global_step))
        
        print(sess.run(global_step))
        
    print('finish') 
    
def time_cost(train_iters):
    for j in range(train_iters):
        a = time.time()
        _, train_summary = sess.run([train_op, merge1], feed_dict={phase_train:True})
        b = time.time()
        train_writer.add_summary(train_summary, sess.run(global_step))
        print('Train_op cost: %0.5f'%(time.time() - a))
        print('Write summary cost: %0.5f'%(time.time() - b))
        c = time.time()
        val_summary = sess.run(merged, feed_dict={phase_train:False})
        print('validation cost: %0.5f'%(time.time() - c))
        val_writer.add_summary(val_summary, sess.run(global_step))    
        print(sess.run(global_step))     
        
def train(train_iters):
    sum_acc  = 0
    sum_loss = 0
    
    for j in range(train_iters):
        _, train_acc, train_loss = sess.run([train_op, accuracy, loss], feed_dict={phase_train:True})
        sum_acc += train_acc
        sum_loss += train_loss
    avg_acc = sum_acc/train_iters
    avg_loss = sum_loss/train_iters
    print('Training data\'s accuracy: %0.4f' %avg_acc)
    print('Training data\'s loss: %0.4f'%avg_loss)    
    
def val(iters):
    sum_acc  = 0
    sum_loss = 0

    for i in range(iters):
        val_acc, val_loss = sess.run([accuracy, loss], feed_dict = {phase_train:False})
        sum_acc += val_acc
        sum_loss += val_loss
    avg_acc = sum_acc/iters
    avg_loss = sum_loss/iters
    print('Validation data\'s accuracy: %0.4f' %avg_acc)
    print('Validation data\'s loss: %0.4f'%avg_loss)        

#==============================================================================
#   Main
#==============================================================================
for i in range(n_epoch):
    a = time.time()
#    train_val(train_iters)
    train(train_iters)
    val(val_iters)
    print('epoch %d is finished'%(i+1))
    print('This epoch cost: %f'%(time.time() - a))
#    if i > 160:
#        val(val_iters)
#val(val_iters)
#saver.save(sess=sess, save_path='logs/wide_resnet3/model' , global_step=global_step)
