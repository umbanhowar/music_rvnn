from model import Model

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

import sys

import pdb

import data.data_utils

#dtype = torch.FloatTensor
dtype = torch.cuda.FloatTensor

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
    return torch.Tensor(batch_ndarray).type(dtype)

if __name__=='__main__':
    model = Model()

    model.cuda()

    learning_rate = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    save_path = 'ckpt/model/model.pt'
    mode = sys.argv[1]

    #model.load_state_dict(torch.load(save_path))
    if mode == 'train':
        lc = None
        try:
            for i in xrange(2500):
                batch_names = ['chor245', 'prelude67-14', 'chor245copy']
                batch_raw = [np.load(x + '.npy') for x in batch_names]
                batch_lengths = np.array([x.shape[1] for x in batch_raw])
                # Get a mapping of ESC end times (inclusive) to ESC IDs
                batch_ESC = [read_ESC(x) for x in batch_names]
                batch_tensor = batch_raw_to_tensor(batch_raw)

                lc = model.forward(batch_tensor, batch_lengths, batch_ESC)

                #if i%10 == 0:
                print i, model.loss, model.rec_loss

                optimizer.zero_grad()
                model.backward()
                optimizer.step()
        except KeyboardInterrupt:
            torch.save(model.state_dict(), save_path)
        torch.save(model.state_dict(), save_path)
    elif mode == 'gen':
        res = model.generate(1)
        VAE_out0 = res[:, 0, :].T
        np.savetxt('VAE_out0', VAE_out0.astype(int), fmt="%i", delimiter=" ")
        pdb.set_trace()




