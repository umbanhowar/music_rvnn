# Lots of help from https://devblogs.nvidia.com/parallelforall/recursive-neural-networks-pytorch/

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor

sequence_len = 7 # Number of steps, including reduce tokens
dim = 7 # Dimension of stack vectors
batch_sz = 20

ex = torch.from_numpy(np.identity(sequence_len)).type(dtype) # Assuming sequence length and vec dimension are the same for now

batch = ex.expand(batch_sz, sequence_len, sequence_len)



for i in range(sequence_len):
    pass