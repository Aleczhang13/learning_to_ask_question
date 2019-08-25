import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import config
from utils import BeamSearchNode, Beam


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
    def __init__(self, input_size, hidden_size, word_vector, num_layers, tgt_vocab, device, dropout, attention=None,
                 min_length=config.min_length, max_length=config.max_length,
                 top_k=config.top_k, top_p=config.top_p,
                 temperature=config.temperature, decode_type=config.decode_type
                 ):
        super(Decoder, self).__init__()
        self.output_dim = len(tgt_vocab.itos)
        self.embedding = nn.Embedding.from_pretrained(word_vector, freeze=config.embedding_freeze)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True,
                           bidirectional=False,
                           dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.att_need = config.att_need
        self.attention = None
        self.min_len_sentence = min_length
        self.max_len_sentence = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.gen = Generator(decoder_size=hidden_size, output_dim=len(tgt_vocab.itos))
        self.temperature = temperature
        self.decode_type = decode_type
        self.special_tokens_ids = [tgt_vocab.stoi[t] for t in ["<EOS>", "<PAD>"]]
        self.device = device

    def decode_rnn(self, dec_input, dec_hidden, enc_out):

        # 进行解码

        if isinstance(self.rnn, nn.GRU):
            dec_output, dec_hidden = self.rnn(dec_input, dec_hidden[0])
        else:
            dec_output, dec_hidden = self.rnn(dec_input, dec_hidden)

        # if self.att_need:
        #     dec_output, p_attn = self.attention(dec_output, enc_out)

        dec_output = self.dropout(dec_output)

        return dec_output, dec_hidden

    def beam_decode(self, decoder_hidden, encoder_out, beam_width=3, topk=3):
        # start decoding step with <SOS> token and empty input feed, stored in a Beam nodes
        bacth_size = encoder_out.size(0)
        decoder_input = torch.zeros(1, 1).fill_(2).long().to(self.device)
        input_feed = torch.zeros(1, 1, encoder_out.size(2), device=self.device)
        node = BeamSearchNode(decoder_hidden, None, decoder_input, 0, 1, input_feed)

        # Initialize Beam queue objects and an output list
        in_nodes = Beam()
        out_nodes = Beam()
        endnodes = []
        # Feed the input Beam queue with the start token
        in_nodes.put((node.eval(), node))

        # Start Beam search
        for i in range(self.max_len_sentence):
            # At each step, keep the beam_width best nodes
            for i in range(beam_width):
                # Get the best node in the input Beam queue
                score, n = in_nodes.get()
                # Collect the values of the node to decode
                dec_input = n.wordid
                dec_hidden = n.hidden
                input_feed = n.feed

                # If we find an <EOS> token, then stop the decoding for this Beam
                if n.wordid.item() in self.special_tokens_ids and n.prevnode != None:
                    endnodes.append((score, n))
                    # Break the loop if we have enough decoded sentences
                    if len(endnodes) >= topk:
                        break
                    else:
                        continue

                # Decode with the RNN
                dec_input = self.embedding(dec_input)  # (batch size, 1, emb dim)
                dec_input = torch.cat((dec_input, input_feed), 2)
                dec_output, dec_hidden = self.decode_rnn(dec_input, dec_hidden, encoder_out)
                out = self.gen(dec_output)
                # Extract the top K most likely tokens and their log probability (log softmax)
                log_prob, indexes = torch.topk(out, beam_width)

                # Create a node for each of the K outputs and score them (sum of log probs div by length of sequence)
                nextnodes = []
                for new_k in range(beam_width):
                    out_t = indexes[0][new_k].view(1, -1)
                    log_p = log_prob[0][new_k].item()
                    node = BeamSearchNode(dec_hidden, n, out_t, n.logp + log_p, n.leng + 1, dec_output)
                    score = node.eval()
                    nextnodes.append((score, node))
                # Push the nodes to the output Beam queue
                for i in range(len(nextnodes)):
                    score, nn = nextnodes[i]
                    out_nodes.put((score, nn))

                # Break the loop if the input Beam is empty (only happens with <SOS> token at first step)
                if len(in_nodes) == 0:
                    break

            # Fill the input Beam queue with the previously computed output Beam nodes
            in_nodes = out_nodes
            out_nodes = Beam()
            # Stop decoding when we have enough output sequences
            if len(endnodes) >= topk:
                break

        # In the case where we did not encounter a <EOS> token, take the most likely sequences
        if len(endnodes) == 0:
            endnodes = [in_nodes.get() for _ in range(topk)]

        # Now we unpack the sequences in reverse order to retrieve the sentences
        utterances = []
        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = [n.wordid.item()]
            while n.prevnode != None:
                n = n.prevnode
                utterance.append(n.wordid.item())
            # Reverse the sentence
            utterance = utterance[::-1]
            utterances.append(utterance)

        return utterances


    def forward(self, encoder_out, encoder_hidden, question):
        batch_size = encoder_out.size(0)

        # prepare for the decoder word
        outputs = []
        # hidden = encoder_hidden[0]

        # depends on paper describle to concat the hidden state
        encoder_hidden = tuple(
            (torch.cat((hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]), dim=2) for hidden in encoder_hidden))
        # enc_out = encoder_out[:, -1, :].unsqueeze(1) if self.att_need else encoder_out
        decoder_hidden = encoder_hidden
        if question is not None:
            q_emb = self.embedding(question)
            input_feed = torch.zeros(batch_size, 1, encoder_out.size(2), device=self.device)
            for decoder_input in q_emb[:, :-1, :].split(1, dim=1):
                decoder_input = torch.cat((decoder_input, input_feed), dim=2)
                decoder_output, decoder_hidden = self.decode_rnn(decoder_input, decoder_hidden, encoder_out)
                outputs.append(self.gen(decoder_output))
                input_feed = decoder_output

        # eval
        else:
            outputs = self.beam_decode(decoder_hidden, encoder_out)

        return outputs


class Generator(nn.Module):
    def __init__(self, decoder_size, output_dim):
        super(Generator, self).__init__()
        self.generator = nn.Linear(decoder_size, output_dim)
        self.gen_func = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        out = self.gen_func(self.generator(x)).squeeze(1)
        return out

# TO DO
class attention(object):
    pass
