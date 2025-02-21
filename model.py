import torch.nn as nn
import layer
import config

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, hidden_size, num_layers, tgt_vocab, device, drop_out):
        super(Seq2Seq, self).__init__()

        self.encoder = layer.Encoder(input_size=src_vocab.vectors.size(1),
                                     hidden_size=hidden_size,
                                     num_layers=num_layers,
                                     word_vectors=src_vocab.vectors,
                                     bidirectional=True,
                                     drop_prob=drop_out if num_layers > 1 else 0.)

        self.decoder = layer.Decoder(input_size=src_vocab.vectors.size(1) + hidden_size,
                                     hidden_size=hidden_size,
                                     word_vector=tgt_vocab.vectors,
                                     tgt_vocab=tgt_vocab,
                                     num_layers=num_layers,
                                     device=device,
                                     dropout=drop_out if num_layers > 1 else 0.,
                                     attention=config.att_need
                                     )


    def forward(self, sentence, len_sentence, question = None, len_question = None):
        enc_output, enc_hidden= self.encoder(sentence, len_sentence)
        outputs = self.decoder(enc_output,enc_hidden, question)
        return outputs




