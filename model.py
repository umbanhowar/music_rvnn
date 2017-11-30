import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

escs = [torch.IntTensor(np.random.randint(2, size=[127, np.random.randint(4,10)])) 
        for _ in xrange(4)]

# HYPERPARAMETERS
CODE_SIZE = 50
EMBED_SIZE = 127

# TODO: how to best handle encoding many-hot vectors? 
embed = Variable(torch.randn(127, EMBED_SIZE))

rnn = nn.GRUCell(EMBED_SIZE, CODE_SIZE)
print(escs[0])
