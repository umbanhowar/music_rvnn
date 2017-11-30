#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 21:09:58 2017

@author: nathan
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ((A (B D)) E )
# A B D ) ) E )

buf = np.identity(7)
#buf[[2, 5, 6, 8],:] = 0
mask = np.ones((7,))
mask[[3, 4, 6]] = 0

# Fake encoder
W = np.random.randn(10, 5)

stack = np.zeros(buf.shape)
queue = np.zeros(buf.shape)
qstart = None
qend = None

for i in xrange(len(buf)):
    if qstart is None and qend is None:
        stack[i, :] = buf[i, :]
        queue[i, :] = stack[i, :]
        qstart = i
        qend = i
    else:
        # Combine
        #combined = np.concatenate((queue[qstart, :], queue[(qstart + 1) % len(queue), :]), axis=0)
        #new = combined.dot(W)
        new = queue[qstart, :] + queue[(qstart + 1) % len(queue), :]
        # If mask, just shift from buffer, else put the new thing
        stack[i, :] = mask[i] * buf[i, :] + (1 - mask[i]) * new
        # Set next thing in queue to be whatever we just calculated
        qend = (qend + 1) % len(queue)
        queue[qend] = stack[i, :]
        # Conditionally update the start of the queue to remove the two combined things if not mask
        qstart = int((qstart + (1 - mask[i]) * 2) % len(queue))
        print np.array_equal(stack, queue)

print buf
print
print stack
print
print queue
print
print np.array_equal(stack, queue)
