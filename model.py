import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

pad_word = "<pad>"
bos_word = "<s>"
eos_word = "</s>"
unk_word = "<unk>"
pad_id = 0
bos_id = 1
eos_id = 2
unk_id = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Seq2seqBaseline(nn.Module):
    def __init__(self, vocab, emb_dim=300, hidden_dim=500, num_layers=2, dropout=0.1):
        super().__init__()

        # Initialize your model's parameters here. To get started, we suggest
        # setting all embedding and hidden dimensions to 300, using encoder and
        # decoder GRUs with 2 layers, and using a dropout rate of 0.1.
        self.vocab = vocab
        self.num_words = vocab.num_words
        print("vocab.num_words ", vocab.num_words)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.num_words, hidden_dim)
        self.encoder = nn.GRU(hidden_dim,
                              hidden_dim,
                              num_layers,
                              dropout=(0 if num_layers == 1 else dropout),
                              bidirectional=True)
        self.decoder = nn.GRU(hidden_dim,
                              hidden_dim,
                              num_layers,
                              dropout=(0 if num_layers == 1 else dropout),
                              bidirectional=False)

        # # this is used by attention
        # self.concat = nn.Linear(hidden_dim*2, hidden_dim)
        self.out = nn.Linear(hidden_dim, self.num_words)

    # MY CODE: HELPER FUNCTION
    def init_glove(self):
        # If a word doesn't appear in GloVe vocab, then you can just use the original embedding.
        # Read GloVe embeddings.
        GloVe = {}
        filename = "glove.840B.300d.conll_filtered.txt"
        for line in open(filename).readlines():
            fields = line.strip().split(" ")
            word = fields[0]
            GloVe[word] = [float(x) for x in fields[1:]]

        # 1 Integrate Glove using Matrix
        weights_matrix = np.zeros((self.num_words, self.emb_dim))
        found = 0

        for word, index in self.vocab.word_to_id.items():
            try:
                weights_matrix[index] = GloVe[word]
                found += 1
            except KeyError:
                weights_matrix[index] = np.random.normal(scale=0.6, size=(self.emb_dim,))

        embedding = nn.Embedding.from_pretrained(torch.tensor(weights_matrix, dtype=torch.float32), padding_idx=0,
                                                 freeze=False)
        print(found)
        return embedding

    # A helper function to get 2D mask from a 2D matrix
    def binaryMatrix(self, l, value=pad_id):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == pad_id:
                    m[i].append(0)  # should be true
                else:
                    m[i].append(1)  # should be false
        return m

    def encode(self, source, hidden=None):
        # Compute a tensor containing the length of each source sequence.
        source_lengths = torch.sum(source != pad_id, axis=0).to('cpu')

        # 1. Convert word idx to embeddings
        embeded = self.embedding(source)
        # 2. Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeded, source_lengths, batch_first=False,
                                                         enforce_sorted=False)
        # 3. Forward pass through GRU
        outputs, hidden = self.encoder(packed, hidden)
        # 4. Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 5. Sum bidirectional GRU outputs
        encoder_output = outputs[:, :, :self.hidden_dim] + outputs[:, :, self.hidden_dim:]
        encoder_mask = torch.BoolTensor(self.binaryMatrix(torch.sum(encoder_output, -1)))
        encoder_hidden = hidden

        return encoder_output, encoder_mask, encoder_hidden

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        # These arguments are not used in the baseline model.
        del encoder_output
        del encoder_mask

        # 1. Get embedding of current input word
        embedded = self.embedding(decoder_input)
        # 2. Forward through unidirectional(decoder) GRU.
        rnn_output, hidden = self.decoder(embedded,
                                          last_hidden)

        # ****temporarily for this question****
        rnn_output = rnn_output.squeeze(0)
        output = self.out(rnn_output)

        # #****for actual attention****
        # output = self.out(concat_output)

        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        # return output, hidden, _
        return output, hidden, None

    def maskNLLLoss(self, inp, target, mask):
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).sum()
        loss = loss.to(device)
        return loss, nTotal.item()

    def compute_loss(self, source, target):
        max_target_len, batch_size = target.shape[0], target.shape[1]
        mask = torch.tensor(
            [[True if target[i][j] != pad_id else False for j in range(batch_size)] for i in range(max_target_len)]).to(
            device)

        # 1.Initialize variables
        loss = 0
        n_totals = 0
        teacher_forcing_ratio = 1.0

        # 2.Forward pass through encoder
        encoder_outputs, encoder_mask, encoder_hidden = self.encode(source)

        # 3.Create initial decoder input
        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[bos_id for _ in range(batch_size)]]).to(device)

        # 4.Create initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.num_layers]

        # Step through the length of the output sequence one token at a time
        # Teacher forcing is used to assist training
        teacher_force = random.random() < teacher_forcing_ratio

        ####################################
        if teacher_force:
            for t in range(1, max_target_len):
                decoder_output, decoder_hidden, attention_weights = self.decode(
                    decoder_input, decoder_hidden, encoder_outputs, encoder_mask
                )
                # Teacher forcing: next input is current target
                decoder_input = target[t].view(1, -1).to(device)

                # Calculate the loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target[t], mask[t])
                loss += mask_loss
                n_totals += nTotal
        else:
            for t in range(1, max_target_len):
                decoder_output, decoder_hidden, attention_weights = self.decode(
                    decoder_input, decoder_hidden, encoder_outputs, encoder_mask
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])

                # Calculate the loss
                mask_loss, nTotal = self.maskNLLLoss(decoder_output, target[t], mask[t])
                loss += mask_loss
                n_totals += nTotal
        losses = loss / n_totals

        return losses


# Luong attention layer
class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        attn_energies = self.general_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class Seq2seqAttention(Seq2seqBaseline):
    def __init__(self, vocab):
        super().__init__(vocab)

        # Initialize any additional parameters needed for this model that are not
        # already included in the baseline model.

        # YOUR CODE HERE
        hidden_size = self.hidden_dim

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.embedding_dropout = nn.Dropout(p=0.1)
        self.attn = Attn(hidden_size)

    def decode(self, decoder_input, last_hidden, encoder_output, encoder_mask):
        # 1. Get embedding of current input word
        embedded = self.embedding(decoder_input)
        embedded = self.embedding_dropout(embedded)
        # 1. Forward through unidirectional GRU
        rnn_output, hidden = self.decoder(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_output)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_output.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden, attn_weights
