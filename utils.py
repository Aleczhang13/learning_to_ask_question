import os
import torch
import spacy
import numpy as np
import math
from nltk.tokenize import word_tokenize
spacy_en = spacy.load('en')

def tokenize_sentence(sentence):
    return [token for token in word_tokenize(sentence)]

def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in spacy_en.tokenizer(text)]


class MetricReporter:
    def __init__(self, last_epoch=0, verbose=True):
        self.epoch = last_epoch
        self.verbose = verbose
        self.training = True
        self.losses = 0
        self.n_samples = 0
        self.n_correct = 0
        self.list_train_loss = [ ]
        self.list_train_accuracy = [ ]
        self.list_train_perplexity = [ ]
        self.list_valid_loss = [ ]
        self.list_valid_accuracy = [ ]
        self.list_valid_perplexity = [ ]

    def train(self):
        self.epoch += 1
        self.training = True
        self.clear_metrics()

    def eval(self):
        self.training = False
        self.clear_metrics()

    def update_metrics(self, l, n_s, n_c):
        self.losses += l
        self.n_samples += n_s
        self.n_correct += n_c

    def compute_loss(self):
        return np.round(self.losses / self.n_samples, 2)

    def compute_accuracy(self):
        return np.round(100 * (self.n_correct / self.n_samples), 2)

    def compute_perplexity(self):
        return np.round(math.exp(self.losses / float(self.n_samples)), 2)

    def report_metrics(self):
        # Compute metrics
        set_name = "Train" if self.training else "Valid"
        loss = self.compute_loss()
        accuracy = self.compute_accuracy()
        perplexity = self.compute_perplexity()
        # Print the metrics to std output if verbose is True
        if self.verbose:
            print("{} loss of the model at epoch {} is: {}".format(set_name, self.epoch, loss))
            print("{} accuracy of the model at epoch {} is: {}".format(set_name, self.epoch, accuracy))

            print("{} perplexity of the model at epoch {} is: {}".format(set_name, self.epoch, perplexity))
        # Store the metrics in lists
        if self.training == True:
            self.list_train_loss.append(loss)
            self.list_train_accuracy.append(accuracy)
            self.list_train_perplexity.append(perplexity)
        else:
            self.list_valid_loss.append(loss)
            self.list_valid_accuracy.append(accuracy)
            self.list_valid_perplexity.append(perplexity)

    def clear_metrics(self):
        self.losses = 0
        self.n_samples = 0
        self.n_correct = 0

    def log_metrics(self, log_filename):
        with open(log_filename, "w") as f:
            f.write("Epochs:" + str(list(range(len(self.list_train_loss)))) + "\n")
            f.write("Train loss:" + str(self.list_train_loss) + "\n")
            f.write("Train accuracy:" + str(self.list_train_accuracy) + "\n")
            f.write("Train perplexity:" + str(self.list_train_perplexity) + "\n")
            f.write("Valid loss:" + str(self.list_valid_loss) + "\n")
            f.write("Valid accuracy:" + str(self.list_valid_accuracy) + "\n")
            f.write("Valid perplexity:" + str(self.list_valid_perplexity) + "\n")

# get from
class BeamSearchNode:
    def __init__(self, hidden, prevnode, wordid, log_prob, length, inputfeed):
        self.hidden = hidden
        self.prevnode = prevnode
        self.wordid = wordid
        self.logp = log_prob
        self.leng = length
        self.feed = inputfeed

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

class Beam:
    def __init__(self):
        self.queue = []

    def __len__(self):
        return len(self.queue)

    def put(self, data):
        self.queue.append(data)

    def get(self):
        try:
            max = 0
            for i in range(len(self.queue)):
                if self.queue[i][0] > self.queue[max][0]:
                    max = i
            item = self.queue[max]
            del self.queue[max]
            return item
        except IndexError:
            print()
            exit()

def dress_for_loss(prediction):
    prediction = torch.stack(prediction).squeeze(0).transpose(0, 1).contiguous()
    return prediction

def correct_tokens(pred, true_tokens, padding_idx):
    pred = pred.view(-1, pred.size(2))
    pred = pred.max(1)[1]
    true_tokens = true_tokens[:, 1:].contiguous()
    non_padding = true_tokens.view(-1).ne(padding_idx)
    num_correct = pred.eq(true_tokens.view(-1)).masked_select(non_padding).sum().item()
    num_non_padding = non_padding.sum().item()
    return num_non_padding, num_correct

def save_checkpoint(state, is_best, filename="/output/checkpoint.pkl"):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best model.")
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Validation loss did not improve.")