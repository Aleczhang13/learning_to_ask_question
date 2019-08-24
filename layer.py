import torch
import torch.nn as nn
import torch.nn.functional as F
import config

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Embedding(nn.Module):
    def __init__(self, word_vectors, padding_idx, drop_prob):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.embed = nn.Embedding.from_pretrained(word_vectors, padding_idx=padding_idx, freeze=config.embedding_freeze)

    def forward(self, x):
        emb = self.embed(x)
        emb = F.dropout(emb, self.drop_prob, self.training)

        return emb


class Encoder(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 word_vectors,
                 bidirectional,
                 drop_prob=0.
                 ):
        super(Encoder, self).__init__()

        num_direction = 2 if bidirectional else 1
        hidden_size = hidden_size // num_direction

        self.embedding = Embedding(word_vectors, padding_idx=1, drop_prob=config.embedding_drop_prob)
        self.drop_prob = drop_prob
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=bidirectional,
                           dropout=drop_prob if num_layers > 1 else 0.)

    def forward(self, x, length):
        x = self.embedding(x)
        # this is a trick to reduce the memory cost
        x = pack_padded_sequence(x, length, batch_first=True)

        x, (hidden, cell) = self.rnn(x)
        # return the orign sentence
        x, _ = pad_packed_sequence(x, batch_first=True)

        return x, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()




