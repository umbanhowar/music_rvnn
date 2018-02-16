import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import pdb
import random

# HYPERPARAMETERS
CODE_SIZE = 300
CL_HIDDEN_SIZE = 50
LD_HIDDEN_SIZE = 50
ENC_EMBED_SIZE = 150
DEC_EMBED_SIZE = 150
MAX_ESC_LEN = 500 # This might need to be bigger

dtype = torch.FloatTensor

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # LSTM cell for encoder side
        self.enc_lstm = nn.LSTMCell(ENC_EMBED_SIZE, CODE_SIZE)
        # Embedding layer for encoder LSTM
        self.enc_embedding = nn.Linear(128, ENC_EMBED_SIZE)

        # Decoder LSTM cell and LSTM output layer mapping output codes to
        # ternary piano roll vectors. 3 possible states (note on, hold, no note)
        # x 128 possible notes.
        self.dec_lstm = nn.LSTMCell(DEC_EMBED_SIZE + CODE_SIZE, CODE_SIZE)
        self.dec_lstm_output_layer = nn.Linear(CODE_SIZE, 3 * 128)
        # Embedding layer for decoder LSTM
        self.dec_embedding = nn.Linear(128, DEC_EMBED_SIZE)

        # Linear layers used for reducing (merging) and splitting
        self.reducer = nn.Linear(2 * CODE_SIZE, CODE_SIZE)
        self.splitter = nn.Linear(CODE_SIZE, 2 * CODE_SIZE)

        # Node classifier
        self.cl0 = nn.Linear(CODE_SIZE, CL_HIDDEN_SIZE)
        self.cl1 = nn.Linear(CL_HIDDEN_SIZE, 2)

        #Length decoder - should this be regression or classification?
        self.ld0 = nn.Linear(CODE_SIZE, LD_HIDDEN_SIZE)
        self.ld1 = nn.Linear(LD_HIDDEN_SIZE, MAX_ESC_LEN)

        # Shared utility functions.
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)
        # Try changing this to tanh, which is what they used in the GRASS paper.
        self.relu = torch.nn.ReLU()
        # Model batch loss - will be dynamically constructed.
        self.loss = 0

        self.rec_loss = 0

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
        out = self.relu(out)
        out = self.ld1(out)
        return out

    def encoder_rnn(self, batch_tensor, batch_ESC, reset_hidden=False):
        """
        batch_tensor: pytorch tensor having dimensions (time, batch, feature) of
        the songs in the batch.
        batch_ESC: list of dicts, where each each element in the batch has a dict
        mapping ESC end times to ESC IDs.
        reset_hidden: whether to reset the hidden state of the RNN
        """

        batch_tensor = Variable(batch_tensor, requires_grad=False)
        hx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)
        cx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)

        # Create list of lists where ESC_vecs[b] will hold Tensors representing
        # the summary codes
        ESC_vecs = [[] for b in range(batch_tensor.shape[1])]

        for i in range(batch_tensor.shape[0]):
            # Embed
            embedded = self.enc_embedding(batch_tensor[i])
            # Process this timestep
            hx, cx = self.enc_lstm(embedded, (hx, cx))
            for b in range(hx.shape[0]):
                # Iterate over the batch checking whether we're at the end of
                # an ESC.
                if i in batch_ESC[b]:
                    # Append a horizontal view of the tensor for compatibility
                    # later in the model.
                    ESC_vecs[b].append(hx[b].view(1, -1))
                    if reset_hidden:
                        hx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)
                        cx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)
        return ESC_vecs

    def encoder_rvnn(self, batch, lengths):
        """
        lengths is really the number of merges (could be zero)
        """
        # Placeholder for final root codes
        done = [None for _ in range(len(batch))]
        stacks = [[] for _ in range(len(batch))]

        for b in range(len(batch)):
            if lengths[b] == 0:
                done[b] = batch[b][-1][0]
        for i in range(len(batch[0])):
            # Move n from buffer to stack
            for b in range(len(batch)):
                stacks[b] += batch[b].pop()

            rights = [None for _ in range(len(batch))]
            lefts = [None for _ in range(len(batch))]

            # Pop the top two elements off the stack for each batch element
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

            # Merge the top two stack elements.
            left = torch.cat(lefts, dim=0)
            right = torch.cat(rights, dim=0)
            result = self.reduce(left, right)
            result = torch.split(result, 1, dim=0)

            # Put the reduced value back on the stack.
            for b in range(len(batch)):
                stacks[b].append(result[b])
                if i + 1 == lengths[b]:
                    done[b] = result[b]
        return done

    def decoder_rvnn(self, latent_codes, batch=None, lengths=None):
        """
        batch:
        [ABC][][D]
        [] [DE][F]
        """
        if batch: # then we are training
            # Initialize the stacks with all the latent codes
            stacks = [[c] for c in latent_codes]
            bufs = [[] for _ in range(len(batch))]
            done = [None for _ in range(len(batch))]

            for b in range(len(batch)):
                if lengths[b] == 0:
                    # Pass the code directly through with correct padding.
                    done[b] = [[latent_codes[b]]] + [[] for _ in range(len(batch[0]) - 1)]
            for i in range(len(batch[0])):
                tops = []
                for b in range(len(batch)):
                    try:
                        tops.append(stacks[b].pop())
                    except IndexError:
                        tops.append(Variable(torch.zeros(1, CODE_SIZE).type(dtype), requires_grad=False))

                top = torch.cat(tops, dim=0)

                # Train the classifier that decides whether to split or if leaf.
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

                losses = self.cross_entropy(logits, labels)
                self.loss += torch.sum(torch.dot(losses, mask))


                left, right = self.split(top)
                lefts = torch.split(left, 1, dim=0)
                rights = torch.split(right, 1, dim=0)

                for b in range(len(batch)):
                    stacks[b].append(lefts[b])
                    stacks[b].append(rights[b])
                    # read batch backwards to determine how many to move
                    # from stack to buffers (could also pop from batch here)
                    num_to_unshift = len(batch[b][len(batch[0]) -1 - i])
                    group = [stacks[b].pop() for _ in range(num_to_unshift)]
                    # Note that this will build the bufs in reverse order
                    bufs[b] += group
            return bufs
        else:
            # Otherwise we are in generation mode
            stacks = [[c] for c in latent_codes]
            is_done = [False for _ in range(len(latent_codes))]
            bufs = [[] for _ in range(len(latent_codes))]
            ctr = 0
            while not all(is_done) and ctr < 10:
                tops = []
                for b in range(len(latent_codes)):
                    try:
                        tops.append(stacks[b].pop())
                    except IndexError:
                        tops.append(Variable(torch.zeros(1, CODE_SIZE).type(dtype), requires_grad=False))
                top = torch.cat(tops, dim=0)

                # Run the classifier that decides whether to split or if leaf.
                logits = self.classify(top)
                probs = nn.functional.softmax(logits, dim=1)
                vals, to_split = torch.max(probs, dim=1)

                left, right = self.split(top)
                lefts = torch.split(left, 1, dim=0)
                rights = torch.split(right, 1, dim=0)

                for b in range(len(latent_codes)):
                    if not is_done[b]:
                        if to_split.data[b] == 0:
                            # If the current node is classified as a leaf
                            # Add the top node to the buffer (note this also
                            # builds in reverse order)
                            bufs[b].append(tops[b])
                            if len(stacks[b]) == 0:
                                # If the stack is empty, we are done with
                                # this sequence element.
                                is_done[b] = True
                        else:
                            # If the current node is not a leaf, we split
                            # correctly and should put the split results
                            # back on the stack.
                            stacks[b].append(lefts[b])
                            stacks[b].append(rights[b])
                ctr += 1
            return bufs







    def decoder_rnn(self, batch_leaf_codes, batch_lengths, batch_ESC, batch_tensor=None, reset_hidden=False):
        """
        batch_leaf_codes: [[Tensor]] - List of leaf codes for each batch element
        batch_tensor: Tensor - The original batch of songs in the ternary
            representation, having dimensions (time, batch, feature).
        batch_lengths: np.ndarray - Vector of lengths (in time) of each batch
            element.
        batch_ESC: [{}] - List of dicts mapping end times of ESCs to ESC IDs for
            each batch element.
        reset_hidden: bool - whether to reset the hidden state of the RNN after
            the end of each ESC.
        """
        if batch_tensor is not None:
            batch_tensor = Variable(batch_tensor, requires_grad=False)

            # We want reverse ordering so that we can pop in the correct order
            batch_leaf_codes = [list(reversed(x)) for x in batch_leaf_codes]
            # We get [[DCBA]]

            curr_ESC = [batch_leaf_codes[b].pop() for b in range(len(batch_leaf_codes))]

            hx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)
            cx = Variable(torch.zeros(batch_tensor.shape[1], CODE_SIZE).type(dtype), requires_grad=False)

            ipt_from_piano_roll = Variable(torch.zeros(batch_tensor.shape[1], DEC_EMBED_SIZE).type(dtype), requires_grad=False)

            for i in range(batch_tensor.shape[0]):
                curr_ESC_ipt = torch.cat(curr_ESC, dim=0)
                ipt = torch.cat([ipt_from_piano_roll, curr_ESC_ipt], dim=1)

                hx, cx = self.dec_lstm(ipt, (hx, cx))

                out = self.relu(hx)
                out = self.dec_lstm_output_layer(out)

                # Transform to 3-class logits for each note
                note_output_logits = out.view(batch_tensor.shape[1], 128, 3)
                note_output_logits = note_output_logits.view(-1, 3)

                # Labels for the corresponding logits.
                labels = batch_tensor[i].contiguous().view(-1).type(torch.LongTensor)

                # Now create a mask to mask out the sequences that are already
                # done in the loss function. There are 128 sets of logits for each
                # batch element (eg 0.1 0.7 0.2), if we are done with a sequence we should mask out
                # the cross entropy for each of these sets of logits.
                # How to determine which elements to mask? For example: if i=1
                # and we're processing a sequence of length 2, we want to include
                # that term because we want to predict the last sequence element.
                # But if i=2, then we want to mask it out. Hence:
                mask = (i < batch_lengths).astype(int)
                mask = np.repeat(mask, 128)
                mask = Variable(torch.Tensor(mask), requires_grad=False)
                self.loss += torch.sum(self.cross_entropy(note_output_logits, labels) * mask)
                self.rec_loss += torch.sum(self.cross_entropy(note_output_logits, labels) * mask)

                # If we're not at the end of an ESC, we just want to update
                # the piano roll input to be the note that we just tried to
                # predict.
                ipt_from_piano_roll = self.dec_embedding(batch_tensor[i])

                # Then conditionally replace the piano roll input with zeros for
                # the batch elements where we are at the end of an ESC?
                for b in range(batch_tensor.shape[1]):
                    # Iterate over the batch, checking whether we are at the end
                    # of an essential structure component.
                    if i in batch_ESC[b]:
                        # If we are, then we want to replace the current ESC code
                        # that we're inputting into the RNN.
                        try:
                            curr_ESC[b] = batch_leaf_codes[b].pop()
                        except IndexError:
                            # If there are no more ESC to pop, it means that we're
                            # already done with this element in this batch. We do
                            # nothing here and allow it to keep processing, but
                            # we mask out this batch element in the loss function.
                            pass
                        # Also, reset the piano roll input to zero, as we want to
                        # predict the note at the next timestep given the new ESC
                        # code.
                        # ipt_from_piano_roll[b] = Variable(torch.zeros(1, DEC_EMBED_SIZE).type(dtype), requires_grad=False)
        else:
            # generation mode
            outputs = []

            # We want reverse ordering so that we can pop in the correct order
            batch_leaf_codes = [list(reversed(x)) for x in batch_leaf_codes]
            # We get [[DCBA]]

            curr_ESC = [batch_leaf_codes[b].pop() for b in range(len(batch_leaf_codes))]

            hx = Variable(torch.zeros(len(batch_leaf_codes), CODE_SIZE).type(dtype), requires_grad=False)
            cx = Variable(torch.zeros(len(batch_leaf_codes), CODE_SIZE).type(dtype), requires_grad=False)

            ipt_from_piano_roll = Variable(torch.zeros(len(batch_leaf_codes), DEC_EMBED_SIZE).type(dtype), requires_grad=False)

            for i in range(max(batch_lengths)):
                curr_ESC_ipt = torch.cat(curr_ESC, dim=0)
                ipt = torch.cat([ipt_from_piano_roll, curr_ESC_ipt], dim=1)

                hx, cx = self.dec_lstm(ipt, (hx, cx))

                out = self.relu(hx)
                out = self.dec_lstm_output_layer(out)

                # Transform to 3-class logits for each note
                note_output_logits = out.view(len(batch_leaf_codes), 128, 3)

                # Convert to probs
                note_output_probs = nn.functional.softmax(note_output_logits, dim=-1)

                # Sample output for this time step. (go back to numpy land for a little)
                note_output_probs = note_output_probs.data.numpy()

                samples = np.zeros((len(batch_leaf_codes), 128))
                for b in range(len(batch_leaf_codes)):
                    for note in range(128):
                        # Sample one of the three different note states
                        val = np.random.choice(3, p=note_output_probs[b, note, :])
                        samples[b, note] = val

                outputs.append(samples)

                # Use this output as the next piano roll input
                back_to_pytorch = Variable(torch.Tensor(samples), requires_grad=False)
                ipt_from_piano_roll = self.dec_embedding(back_to_pytorch)

                # Then conditionally replace the piano roll input with zeros for
                # the batch elements where we are at the end of an ESC?
                for b in range(len(batch_leaf_codes)):
                    # Iterate over the batch, checking whether we are at the end
                    # of an essential structure component.
                    if i in batch_ESC[b]:
                        # If we are, then we want to replace the current ESC code
                        # that we're inputting into the RNN.
                        try:
                            curr_ESC[b] = batch_leaf_codes[b].pop()
                        except IndexError:
                            # If there are no more ESC to pop, it means that we're
                            # already done with this element in this batch. We do
                            # nothing here and allow it to keep processing, but
                            # we mask out this batch element in the loss function.
                            pass
                        # Also, reset the piano roll input to zero, as we want to
                        # predict the note at the next timestep given the new ESC
                        # code. TODO: should we do this or no?
                        ipt_from_piano_roll[b] = Variable(torch.zeros(1, DEC_EMBED_SIZE).type(dtype), requires_grad=False)

            pdb.set_trace()
            return np.stack(outputs, axis=0)


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
        # These are the labels.
        lengths_flat = Variable(torch.Tensor(lengths_flat).type(torch.LongTensor), requires_grad=False)
        return lengths_flat

    def decode_lengths(self, leaf_codes):
        # Flatten out the batch dimension
        batch_flat = reduce(lambda x, y: x + y, leaf_codes)
        batch_flat_tensor = torch.cat(batch_flat, dim=0)
        return self.length_decoder(batch_flat_tensor)

    def random_hierarchy(self, ESC_vecs):
        hierarchies = [None for _ in ESC_vecs]
        for b in range(len(ESC_vecs)):
            symbols_to_generate = 2 * len(ESC_vecs[b])
            symbols_remaining = symbols_to_generate
            num_open_left_parens = 0
            curr_group = []
            hierarchy = []
            for t in range(symbols_to_generate):
                num = num_open_left_parens * (symbols_remaining + num_open_left_parens + 2)
                denom = (2 * symbols_remaining) * (num_open_left_parens + 1)
                prob_right_paren = float(num) / denom
                if random.random() < prob_right_paren:
                    hierarchy.append(list(curr_group))

                    num_open_left_parens -= 1
                else:
                    #curr_group.append()
                    curr_group = []
                    num_open_left_parens += 1

                symbols_remaining -= 1

        #TODO: this is not at all done


        pdb.set_trace()



    def forward(self, batch_tensor, batch_lengths, batch_ESC):
        # Reset the loss to zero for each batch, as it will be dynamically
        # constructed.
        self.loss = 0
        self.rec_loss = 0

        ESC_vecs = self.encoder_rnn(batch_tensor, batch_ESC)

        # Organize the ESC into hierarchies. Say we have two elements in the
        # batch, having ESC vectors ABCD and DEF. Then one possible hierarchy
        # is [ABC][][D]
        #    [DE][F][] where the last empty list is padding.
        ########################################
        # This part will be solved by random hierarchy
        hierarchies = [[], [], []]
        hierarchies[0].append(ESC_vecs[0][:2])
        hierarchies[0].append([ESC_vecs[0][2]])
        hierarchies[1].append(ESC_vecs[1][:3])
        hierarchies[1].append([])
        hierarchies[1].append([ESC_vecs[1][3]])
        hierarchies[2].append([ESC_vecs[2][0]])
        lengths = np.array([[2],[3],[0]])
        ########################################
        #hierarchies = self.random_hierarchy(ESC_vecs)

        # [D][][ABC] so we can pop from the ends.
        # [][F][DE]
        batch_encoding_padded = self.encoding_pad(hierarchies)
        # [ABC][][D]
        # [] [DE][F]
        batch_decoding_padded = self.decoding_pad(hierarchies)

        latent_codes = self.encoder_rvnn(batch_encoding_padded, lengths)

        # Get the decoded leaf codes, in reverse order
        # I.E. [D][][ABC]
        leaf_codes_reversed = self.decoder_rvnn(latent_codes, batch=batch_decoding_padded, lengths=lengths)
        #batch_reconstructed = self.decoder_rvnn(latent_codes)

        leaf_codes = [list(reversed(x)) for x in leaf_codes_reversed]

        # TODO(TEMP): delete later
        lc = list(leaf_codes)

        # At this step, we are left with leaf_codes = [[ABCD]]

        # Find the lengths of each essential structure component
        ESC_lengths = self.transform_ESC_dict_to_lengths(batch_ESC)
        ESC_lengths_vec = self.transform_ESC_lengths_to_vec(ESC_lengths)
        decoded_length_logits = self.decode_lengths(leaf_codes)

        # For testing
        length_softmax = nn.functional.softmax(decoded_length_logits, dim=1)
        vals, indices = torch.max(length_softmax, dim=1)

        # TEMP: how to do this for generation?
        decoded_len_logits = self.decode_lengths(leaf_codes)
        length_softmax = nn.functional.softmax(decoded_length_logits, dim=1)
        vals, indices = torch.max(length_softmax, dim=1)
        num_esc_by_batch = [len(x) for x in leaf_codes]
        esc_lengths_organized = []
        start = 0
        for n in num_esc_by_batch:
            esc_lengths_organized.append(indices[start:start+n].data.numpy())
            start = start + n

        # analogous to the thing i called batch_ESC
        esc_ends = [set(np.cumsum(elt) - 1) for elt in esc_lengths_organized]
        time_lengths = np.array([np.sum(x) for x in esc_lengths_organized])
        # now build the array of lengths from this

        #pdb.set_trace()


        self.loss += torch.sum(self.cross_entropy(decoded_length_logits, ESC_lengths_vec))

        # Now run the decoder rnn and train using teacher-forcing
        # TODO: the model becomes considerably harder to train at this step.
        # Look for bugs / talk to DR
        self.decoder_rnn(leaf_codes, batch_lengths, batch_ESC, batch_tensor)
        #self.decoder_rnn(leaf_codes, _lengths, batch_ESC)

        return lc

    def fake_generate(self, leaf_codes):
        pdb.set_trace()

        decoded_len_logits = self.decode_lengths(leaf_codes)
        length_softmax = nn.functional.softmax(decoded_len_logits, dim=1)
        vals, indices = torch.max(length_softmax, dim=1)
        num_esc_by_batch = [len(x) for x in leaf_codes]
        esc_lengths_organized = []
        start = 0
        for n in num_esc_by_batch:
            esc_lengths_organized.append(indices[start:start+n].data.numpy())
            start = start + n

        # analogous to the thing i called batch_ESC
        esc_ends = [set(np.cumsum(elt) - 1) for elt in esc_lengths_organized]
        time_lengths = np.array([np.sum(x) for x in esc_lengths_organized])

        return self.decoder_rnn(leaf_codes, time_lengths, esc_ends)

    def backward(self):
        self.loss.backward()


