import os
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
        self.list_train_loss = []
        self.list_train_accuracy = []
        self.list_train_perplexity = []
        self.list_valid_loss = []
        self.list_valid_accuracy = []
        self.list_valid_perplexity = []

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
        if self.train:
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

