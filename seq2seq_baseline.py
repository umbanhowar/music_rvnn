import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb
import random

EMBED_SIZE = 150
HIDDEN_SIZE = 200

LD_HIDDEN_SIZE = 50
MAX_LEN = 2500

dtype = torch.FloatTensor

class Seq2Seq(nn.Module):
    def __init__(self):
        super(Seq2Seq, self).__init__()

        # LSTM cell
        self.enc_LSTM = nn.LSTMCell(EMBED_SIZE, HIDDEN_SIZE)
        # Embedding layer for encoder LSTM
        self.enc_embedding = nn.Linear(128, EMBED_SIZE)

        # LSTM cell
        self.dec_LSTM = nn.LSTMCell(EMBED_SIZE, HIDDEN_SIZE)
        # Embedding layer for encoder LSTM
        self.dec_embedding = nn.Linear(128, EMBED_SIZE)

        # Output layer to transform RNN output to distribution over possible notes
        self.output_layer = nn.Linear(HIDDEN_SIZE, 3 * 128)

        self.ld0 = nn.Linear(HIDDEN_SIZE, LD_HIDDEN_SIZE)
        self.ld1 = nn.Linear(LD_HIDDEN_SIZE, MAX_LEN)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.relu = nn.ReLU()

        self.loss = None

    def encoder(self, batch_tensor, batch_lengths):
        batch_tensor = Variable(batch_tensor, requires_grad=False)

        done = [None for _ in range(batch_tensor.shape[1])]

        hx = Variable(torch.zeros(batch_tensor.shape[1], HIDDEN_SIZE).type(dtype), requires_grad=False)
        cx = Variable(torch.zeros(batch_tensor.shape[1], HIDDEN_SIZE).type(dtype), requires_grad=False)

        for i in range(batch_tensor.shape[0]):
            ipt = self.enc_embedding(batch_tensor[i])
            hx, cx = self.enc_LSTM(ipt, (hx, cx))
            for b in range(batch_tensor.shape[1]):
                if i+1 == batch_lengths[b]:
                    done[b] = hx[b]

        return torch.stack(done, dim=0)




    def decoder(self, latent_codes, batch_tensor=None):
        batch_tensor = Variable(batch_tensor, requires_grad=False)

        hx = latent_codes
        cx = Variable(torch.zeros(batch_tensor.shape[1], HIDDEN_SIZE).type(dtype), requires_grad=False)

        ipt = Variable(torch.zeros(batch_tensor.shape[1], EMBED_SIZE).type(dtype), requires_grad=False)
        for i in range(batch_tensor.shape[0]):
            hx, cx = self.dec_LSTM(ipt, (hx, cx))

            out = self.relu(hx)
            out = self.output_layer(out)

            # Transform to 3-class logits for each note
            note_output_logits = out.view(batch_tensor.shape[1], 128, 3)
            note_output_logits = note_output_logits.view(-1, 3)

            # Labels for the corresponding logits.
            labels = batch_tensor[i].contiguous().view(-1).type(torch.LongTensor)

            mask = (i < batch_lengths).astype(int)
            mask = np.repeat(mask, 128)
            mask = Variable(torch.Tensor(mask), requires_grad=False)
            self.loss += torch.sum(self.cross_entropy(note_output_logits, labels) * mask)

            ipt = self.dec_embedding(batch_tensor[i])

    def length_decoder(self, latent_codes, true_lengths=None):
        scores = self.ld0(latent_codes)
        scores = self.relu(scores)
        scores = self.ld1(scores)
        if true_lengths is not None:
            true_lengths = Variable(torch.Tensor(true_lengths).type(torch.LongTensor), requires_grad=False)
            self.loss += torch.sum(self.cross_entropy(scores, true_lengths))
        else:
            probs = nn.functional.softmax(scores, dim=1)
            _, predicted_lengths = torch.max(probs, dim=1)
            return predicted_lengths

    def forward(self, batch_tensor, batch_lengths):
        self.loss = 0
        latent_codes = self.encoder(batch_tensor, batch_lengths)
        self.length_decoder(latent_codes, batch_lengths)
        self.decoder(latent_codes, batch_tensor)

    def backward(self):
        self.loss.backward()

################# Run utilities

from run import batch_raw_to_tensor

if __name__ == '__main__':
    model = Seq2Seq()
    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batch_names = ['chor245', 'prelude67-14', 'chor245copy']
    batch_raw = [np.load(x + '.npy') for x in batch_names]
    batch_lengths = np.array([x.shape[1] for x in batch_raw])
    batch_tensor = batch_raw_to_tensor(batch_raw)

    for i in xrange(1000):
        model.forward(batch_tensor, batch_lengths)
        print i, model.loss
        optimizer.zero_grad()
        model.backward()
        optimizer.step()



