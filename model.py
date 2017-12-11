import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn


# HYPERPARAMETERS
CODE_SIZE = 50
EMBED_SIZE = 127

dtype = torch.FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.reducer = nn.Linear(2 * CODE_SIZE, CODE_SIZE)
        self.relu = torch.nn.ReLU()

    def reduce(self, left, right):
        both = torch.cat([left, right], dim=1)
        reduced = self.reducer(both)
        return self.relu(reduced)

    def encode(self, batch, lengths):
        done = [[] for _ in len(batch)]
        stacks = [[] for _ in len(batch)]
        for i in range(len(batch[0])):
            for b in range(len(batch)):
                # Make sure this is on the right side (ie first elements need to be at end)
                stacks[b] = stacks[b] + batch[b].pop()

            lefts = [[] for _ in range(len(batch))]
            rights = [[] for _ in range(len(batch))]

            for b in range(len(batch)):
                try:
                    lefts[b] = stacks[b].pop()
                except IndexError:
                    # When we're already done with this batch element
                    lefts[b] = torch.zeros(1, CODE_SIZE).type(dtype)
                try:
                    rights[b] = stacks[b].pop()
                except IndexError:
                    # When we're already done with this batch element
                    rights[b] = torch.zeros(1, CODE_SIZE).type(dtype)

            left = torch.cat(lefts, dim=0)
            right = torch.cat(rights, dim=0)
            result = self.reduce(left, right)
            result_split = torch.split(result, 1, dim=0)

            for b in range(len(batch)):
                stacks[b].push(result_split[b])
                if b + 1 == lengths[b]:
                    done[b] = result_split[b]
        return done

    def forward(self, batch_start_pad, batch_end_pad, lengths):
        latent_codes = self.encode(batch_start_pad)


