from model import Model

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import pdb


def read_ESC(fname):
    with open(fname + '_ESC') as f:
        lines = f.readlines()
        return dict([[int(num) for num in line.strip().split(',')] for line in lines])

def batch_raw_to_tensor(batch_raw):
    # We want to create a pytorch tensor having dimensions (time, batch, feature)
    # Max song length
    max_len = max([pr.shape[1] for pr in batch_raw])
    # Pad all songs with zeroes to this length
    for i in range(len(batch_raw)):
        song_len = batch_raw[i].shape[1]
        batch_raw[i] = np.concatenate((batch_raw[i], np.zeros([128, max_len-song_len])), axis=1)
    # Make the batch dimension first to get (batch, feature, time)
    batch_ndarray = np.stack(batch_raw, axis=0)
    # Transpose to desired axis ordering
    batch_ndarray = np.transpose(batch_ndarray, (2, 0, 1))
    return torch.Tensor(batch_ndarray)

if __name__=='__main__':
    model = Model()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for i in xrange(10000):
        batch_names = ['chor245', 'prelude67-14']
        batch_raw = [np.load(x + '.npy') for x in batch_names]
        # Get a mapping of ESC end times (inclusive) to ESC IDs
        batch_ESC = [read_ESC(x) for x in batch_names]
        batch_tensor = batch_raw_to_tensor(batch_raw)

        model.forward(batch_tensor, batch_ESC)

	#print model.loss

        optimizer.zero_grad()
        model.backward()
        optimizer.step()
