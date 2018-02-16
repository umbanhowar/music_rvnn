import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb
import random

EMBED_SIZE = 150
HIDDEN_SIZE = 200

dtype = torch.FloatTensor

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # LSTM cell
        self.LSTM = nn.LSTMCell(EMBED_SIZE, HIDDEN_SIZE)
        # Embedding layer for encoder LSTM
        self.embedding = nn.Linear(128, EMBED_SIZE)

        # Output layer to transform RNN output to distribution over possible notes
        self.output_layer = nn.Linear(HIDDEN_SIZE, 3 * 128)

        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        self.relu = nn.ReLU()

        self.loss = None

    def forward(self, batch_tensor, batch_lengths):
        self.loss = 0

        batch_tensor = Variable(batch_tensor, requires_grad=False)

        hx = Variable(torch.zeros(batch_tensor.shape[1], HIDDEN_SIZE).type(dtype), requires_grad=False)
        cx = Variable(torch.zeros(batch_tensor.shape[1], HIDDEN_SIZE).type(dtype), requires_grad=False)

        ipt = Variable(torch.zeros(batch_tensor.shape[1], EMBED_SIZE).type(dtype), requires_grad=False)
        for i in range(batch_tensor.shape[0]):

            hx, cx = self.LSTM(ipt, (hx, cx))
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

            ipt = self.embedding(batch_tensor[i])



    def sample(self, num_to_sample, timesteps):
        # generation mode
        outputs = []

        hx = Variable(torch.zeros(num_to_sample, HIDDEN_SIZE).type(dtype), requires_grad=False)
        cx = Variable(torch.zeros(num_to_sample, HIDDEN_SIZE).type(dtype), requires_grad=False)

        ipt = Variable(torch.zeros(num_to_sample, EMBED_SIZE).type(dtype), requires_grad=False)

        for i in range(timesteps):
            hx, cx = self.LSTM(ipt, (hx, cx))
            out = self.relu(hx)
            out = self.output_layer(out)

            # Transform to 3-class logits for each note
            note_output_logits = out.view(num_to_sample, 128, 3)

            # Convert to probs
            note_output_probs = nn.functional.softmax(note_output_logits, dim=-1)

            # Sample output for this time step. (go back to numpy land for a little)
            note_output_probs = note_output_probs.data.numpy()

            samples = np.zeros((num_to_sample, 128))
            for b in range(num_to_sample):
                for note in range(128):
                    # Sample one of the three different note states
                    val = np.random.choice(3, p=note_output_probs[b, note, :])
                    samples[b, note] = val

            outputs.append(samples)

            # Use this output as the next piano roll input
            back_to_pytorch = Variable(torch.Tensor(samples), requires_grad=False)
            ipt = self.embedding(back_to_pytorch)

        return np.stack(outputs, axis=0)



    def backward(self):
        self.loss.backward()

################# Run utilities

from run import batch_raw_to_tensor

if __name__ == '__main__':
    model = RNN()
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
    res = model.sample(2, 129)
    out1 = res[:, 0, :].T
    out2 = res[:, 1, :].T
    np.savetxt('rnn_out1', out1.astype(int), fmt="%i", delimiter=" ")
    np.savetxt('rnn_out2', out2.astype(int), fmt="%i", delimiter=" ")
