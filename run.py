from model import Model

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import pdb

import data.data_utils


def read_ESC(fname):
    """
    From a filename, create a dict mapping ESC end times to ESC IDs.
    """
    with open(fname + '_ESC') as f:
        lines = f.readlines()
        return dict([[int(num) for num in line.strip().split(',')] for line in lines])

def batch_raw_to_tensor(batch_raw):
    """
    Create a pytorch tensor of a batch of songs to enable RNN processing.
    """
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

    lc = None
    for i in xrange(1000):
        print i
        batch_names = ['chor245', 'prelude67-14', 'chor245copy']
        batch_raw = [np.load(x + '.npy') for x in batch_names]
        batch_lengths = np.array([x.shape[1] for x in batch_raw])
        # Get a mapping of ESC end times (inclusive) to ESC IDs
        batch_ESC = [read_ESC(x) for x in batch_names]
        batch_tensor = batch_raw_to_tensor(batch_raw)

        lc = model.forward(batch_tensor, batch_lengths, batch_ESC)

        print model.loss

        optimizer.zero_grad()
        model.backward()
        optimizer.step()
    res = model.fake_generate(lc)

    pdb.set_trace()

    song1 = res[:112, 1, :].T
    song1_orig = batch_raw[1][:, :112]

    np.savetxt('song1_gen', song1.astype(int), fmt="%i", delimiter=" ")
    np.savetxt('song1_orig', song1_orig.astype(int), fmt="%i", delimiter=" ")




