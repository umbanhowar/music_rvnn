import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

# HYPERPARAMETERS
CODE_SIZE = 50
EMBED_SIZE = 127
CL_HIDDEN_SIZE = 50
LD_HIDDEN_SIZE = 50

dtype = torch.FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.enc_lstm = nn.LSTMCell(128, CODE_SIZE)

        self.reducer = nn.Linear(2 * CODE_SIZE, CODE_SIZE)
        self.splitter = nn.Linear(CODE_SIZE, 2 * CODE_SIZE)

        # Node classifier
        self.cl0 = nn.Linear(CODE_SIZE, CL_HIDDEN_SIZE)
        self.cl1 = nn.Linear(CL_HIDDEN_SIZE, 2)
        self.cl_cross_entropy = nn.CrossEntropyLoss(reduce=False)

        #Length decoder - should this be regression or classification?
        self.ld0 = nn.Linear(CODE_SIZE, LD_HIDDEN_SIZE)
        self.ld1 = nn.Linear(LD_HIDDEN_SIZE, 1)

        self.relu = torch.nn.ReLU()
        self.loss = 0

    def encoding_pad(self, hierarchies):
        max_len = max([len(x) for x in hierarchies])
        padded = [x + [[] for _ in range(max_len - len(x))] for x in hierarchies]
        for x in padded:
            x.reverse()
        return padded

    def decoding_pad(self, hierarchies):
        max_len = max([len(x) for x in hierarchies])
        padded = [[[] for _ in range(max_len - len(x))] + x for x in hierarchies]
        return padded

    def reduce(self, left, right):
        both = torch.cat([left, right], dim=1)
        reduced = self.reducer(both)
        return self.relu(reduced)

    def split(self, codes):
        both = self.splitter(codes)
        left, right = torch.chunk(both, 2, dim=1)
        return (left, right)

    def classify(self, codes):
        out = self.cl0(codes)
        out = self.relu(out)
        out = self.cl1(out)
        return out

    def length_decoder(self, codes):
        out = self.ld0(codes)
        out = self.relu(codes)
        out = self.ld1(codes)
        return out

    def encoder_rnn(self, batch_tensor, batch_ESC, reset_hidden=False):
        batch_tensor = Variable(batch_tensor, requires_grad=False)
        hx = Variable(torch.randn(batch_tensor.shape[1], CODE_SIZE), requires_grad=False)
        cx = Variable(torch.randn(batch_tensor.shape[1], CODE_SIZE), requires_grad=False)

        ESC_vecs = [[] for b in range(batch_tensor.shape[1])]

        for i in range(batch_tensor.shape[0]):
            hx, cx = self.enc_lstm(batch_tensor[i], (hx, cx))
            for b in range(hx.shape[0]):
                # Iterate over the batch
                if i in batch_ESC[b]:
                    ESC_vecs[b].append(hx[b].view(1, -1))
                    if reset_hidden:
                        # TODO: should these have zero inits?
                        hx = Variable(torch.randn(batch_tensor.shape[1], CODE_SIZE), requires_grad=False)
                        cx = Variable(torch.randn(batch_tensor.shape[1], CODE_SIZE), requires_grad=False)
        return ESC_vecs

    def encode(self, batch, lengths):
        done = [[] for _ in range(len(batch))]
        stacks = [[] for _ in range(len(batch))]
        for i in range(len(batch[0])):
            for b in range(len(batch)):
                # Make sure this is on the right side (ie first elements need to be at end)
                stacks[b] = stacks[b] + batch[b].pop()

            rights = [[] for _ in range(len(batch))]
            lefts = [[] for _ in range(len(batch))]

            for b in range(len(batch)):
                try:
                    rights[b] = stacks[b].pop()
                except IndexError:
                    # When we're already done with this batch element
                    rights[b] = Variable(torch.zeros(1, CODE_SIZE).type(dtype), requires_grad=False)
                try:
                    lefts[b] = stacks[b].pop()
                except IndexError:
                    # When we're already done with this batch element
                    lefts[b] = Variable(torch.zeros(1, CODE_SIZE).type(dtype), requires_grad=False)

            left = torch.cat(lefts, dim=0)
            right = torch.cat(rights, dim=0)
            result = self.reduce(left, right)
            result_split = torch.split(result, 1, dim=0)

            for b in range(len(batch)):
                stacks[b].append(result_split[b])
                if i + 1 == lengths[b]:
                    done[b] = result_split[b]
        return done

    def decode(self, latent_codes, batch=None, lengths=None):
        if batch:
            # Then we are in training mode
            assert lengths is not None

            # Initialize the stacks with all the latent codes
            # TODO: handle the case where there is no structure
            stacks = [[c] for c in latent_codes]
            bufs = [[] for _ in range(len(batch))]
            done = [[] for _ in range(len(batch))]

            for i in range(len(batch[0])):
                tops = []
                for b in range(len(batch)):
                    try:
                        tops.append(stacks[b].pop())
                    except IndexError:
                        tops.append(Variable(torch.zeros(1, CODE_SIZE).type(dtype), requires_grad=False))

                top = torch.cat(tops, dim=0)

                logits = self.classify(top)
                # Say lengths = 2. Then when i=0, we want label 1 mask 1
                # when i=1, we want label 0 mask 1
                # when i=2, we don't care what the label is, we want mask 0
                # We want to label anything that's getting split as one and
                # everything else as zero.
                # We want to mask out anything where we have already done one zero mask
                labels = (i < (lengths - 1)).astype(int)
                labels = Variable(torch.Tensor(labels).type(torch.LongTensor), requires_grad=False)
                labels = labels.view(len(batch))

                mask = (i < lengths).astype(int)
                mask = Variable(torch.Tensor(mask), requires_grad=False)
                mask = mask.view(len(batch))

                losses = self.cl_cross_entropy(logits, labels)
                self.loss += torch.sum(torch.dot(losses, mask))

                left, right = self.split(top)
                lefts = torch.split(left, 1, dim=0)
                rights = torch.split(right, 1, dim=0)

                for b in range(len(batch)):
                    stacks[b].append(lefts[b])
                    stacks[b].append(rights[b])
                    # read batch backwards to determine how many to move
                    # from stack to buffers
                    num_to_unshift = len(batch[b][len(batch[0]) -1 - i])
                    group = [stacks[b].pop() for _ in range(num_to_unshift)]
                    # Note that this will build the bufs in reverse order
                    bufs[b].append(group)

                    if i + 1 == lengths[b]:
                        done[b] = bufs[b]
            return done
        else:
            pass

    def transform_ESC_dict_to_lengths(self, batch_ESC):
        batch_ESC_lengths = []
        for b in range(len(batch_ESC)):
            ESC_list = sorted(batch_ESC[b].items(), key=lambda pair: pair[0])
            ESC_ends = [pair[0] for pair in ESC_list]
            ESC_lengths = []
            start = 0
            for num in ESC_ends:
                ESC_lengths.append(num - start + 1)
                start = (num + 1)
            batch_ESC_lengths.append(ESC_lengths)
        return batch_ESC_lengths

    def transform_ESC_lengths_to_vec(self, ESC_lengths):
        lengths_flat = reduce(lambda x, y: x + y, ESC_lengths)
        lengths_flat = np.array(lengths_flat)
        lengths_flat = Variable(torch.Tensor(lengths_flat).type(dtype), requires_grad=False)
        return lengths_flat

    def decode_lengths(self, batch_reconstructed):
        # Reverse the batch to get the natural ordering
        batch_reconstructed = [reversed(x) for x in batch_reconstructed]
        # Flatten
        batch_reconstructed = [[x for sub in elt for x in sub] for elt in batch_reconstructed]
        # Really flatten
        batch_flat = reduce(lambda x, y: x + y, batch_reconstructed)
        batch_flat_tensor = torch.cat(batch_flat, dim=0)
        return self.length_decoder(batch_flat_tensor)


    def forward(self, batch_tensor, batch_ESC):
        self.loss = 0
        ESC_vecs = self.encoder_rnn(batch_tensor, batch_ESC)

        ########################################
        # This part will be solved by random hierarchy
        hierarchies = [[], []]
        hierarchies[0].append(ESC_vecs[0][:2])
        hierarchies[0].append([ESC_vecs[0][2]])
        hierarchies[1].append(ESC_vecs[1][:3])
        hierarchies[1].append([])
        hierarchies[1].append([ESC_vecs[1][3]])
        lengths = np.array([[2],[3]])
        ########################################

        batch_encoding_padded = self.encoding_pad(hierarchies)
        batch_decoding_padded = self.decoding_pad(hierarchies)

        latent_codes = self.encode(batch_encoding_padded, lengths)
        batch_reconstructed = self.decode(latent_codes, batch=batch_decoding_padded, lengths=lengths)

        # Find the lengths of each essential structure component
        # TODO: use this representation in the beginning anyway?
        ESC_lengths = self.transform_ESC_dict_to_lengths(batch_ESC)
        ESC_lengths_vec = self.transform_ESC_lengths_to_vec(ESC_lengths)
        decoded_lengths = self.decode_lengths(batch_reconstructed)
        print ESC_lengths_vec, decoded_lengths
        self.loss += torch.sum(torch.sqrt((ESC_lengths_vec - decoded_lengths) ** 2))

    def backward(self):
        self.loss.backward()


