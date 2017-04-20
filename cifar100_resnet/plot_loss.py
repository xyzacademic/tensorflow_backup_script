#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 09:41:39 2017

@author: wowjoy
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

f = open('resnet5.pkl','rb')
w = pickle.load(f)
f.close()
f = open('resnet6.pkl', 'rb')
bk = pickle.load(f)
f.close()
n = np.arange(200)
fig = plt.figure()

key = ['train_acc','train_loss','val_acc','val_loss']
plt.subplot(2,2,1)
plt.plot(n, w[key[0]], c='r')
plt.plot(n, bk[key[0]],c='b')

plt.subplot(2,2,2)
plt.plot(n, w[key[1]], c='r')
plt.plot(n, bk[key[1]],c='b')

plt.subplot(2,2,3)
plt.plot(n, w[key[2]], c='r')
plt.plot(n, bk[key[2]],c='b')

plt.subplot(2,2,4)
plt.plot(n, w[key[3]], c='r')
plt.plot(n, bk[key[3]],c='b')


plt.show()