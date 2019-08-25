# 使用相应的log进行记录
import logging
import os

import torch
from torchtext import data,vocab
from tqdm import tqdm

import config
from utils import tokenize_sentence, tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'


# Progress relate dataset
class Squadataset_process(data.Dataset):
    def __init__(self, src_path, tgt_path, fields, max_len=None):

        examples = []
        # 设置相应的fields
        fields = [('src', fields[0]), ('tgt', fields[1])]

        with open(src_path) as src_file, open(tgt_path) as tgt_file:
            src_file = src_file.readlines()[0:10]
            tgt_file = tgt_file.readlines()[0:10]

            for src_line, tgt_line in tqdm(tuple(zip(src_file, tgt_file))):
                src_sentence = tokenize_sentence(src_line)
                tgt_sentence = tokenize_sentence(tgt_line)
                if max_len is not None:
                    src_sentence = src_sentence[:max_len]
                    src_sentence = str(" ".join(src_sentence))
                    tgt_sentence = tgt_sentence[:max_len]
                    tgt_sentence = str(" ".join(tgt_sentence))

                # assert src_sentence == " " or tgt_sentence ==" "
                examples.append(data.Example.fromlist([src_sentence, tgt_sentence], fields))

        super(Squadataset_process, self).__init__(examples, fields)


class DataPreprocessor(object):
    def __init__(self):
        self.src_field, self.tgt_field = self.generate_fields()

    def preprocess(self, src_path, tgt_path, train_save_path, test_save_path, max_len=None):
        logger.info("Preprocessing dataset...")
        train_data, test_data = self.generate_dataset(src_path, tgt_path, max_len)
        logger.info("Saving train dataset...")
        self.save_data(train_save_path, train_data)
        logger.info("Saving test dataset...")
        self.save_data(test_save_path, test_data)

        # Building field vocabulary
        self.src_field.build_vocab(train_data, max_size=config.in_vocab_size)
        self.tgt_field.build_vocab(test_data, max_size=config.out_vocab_size)


        src_vocab, tgt_vocab = self.generate_vocabs()
        vocabs = {'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}

        return train_data, test_data, vocabs

    def load_data(self, train_save_path, test_save_path, glove_dir):
        # Loading saved data
        train_dataset = torch.load(train_save_path)
        train_examples = train_dataset['examples']

        test_dataset = torch.load(test_save_path)
        test_examples = test_dataset['examples']

        # Generating torchtext dataset class
        fields = [('src', self.src_field), ('tgt', self.tgt_field)]
        train_dataset = data.Dataset(fields=fields, examples=train_examples)
        test_dataset = data.Dataset(fields=fields, examples=test_examples)

        # Loading GloVE vectors
        vec = vocab.Vectors(os.path.join(glove_dir, "glove.840B.{}d.txt".format(config.word_embedding_size)))

        # Building field vocabulary
        self.src_field.build_vocab(train_dataset, vectors=vec, max_size=config.in_vocab_size)
        self.tgt_field.build_vocab(train_dataset, vectors=vec, max_size=config.out_vocab_size)

        src_vocab, tgt_vocab = self.generate_vocabs()
        vocabs = {'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}

        return train_dataset, test_dataset, vocabs

    def generate_vocabs(self):
        src_vocab = self.src_field.vocab
        tgt_vocab = self.tgt_field.vocab

        return src_vocab, tgt_vocab

    def generate_dataset(self, src_path, tgt_path, max_len=None):
        train_data = Squadataset_process(src_path=src_path[0],
                                         tgt_path=tgt_path[0],
                                         fields=(self.src_field, self.tgt_field),
                                         max_len=max_len)

        test_data = Squadataset_process(src_path=src_path[1],
                                        tgt_path=tgt_path[1],
                                        fields=(self.src_field, self.tgt_field),
                                        max_len=max_len)

        dataset = [train_data, test_data]

        return dataset


    def generate_fields(self):
        src_field = data.Field(tokenize=tokenizer,
                               sequential=True,
                               init_token=SOS_WORD,
                               eos_token=EOS_WORD,
                               pad_token=PAD_WORD,
                               batch_first=True,
                               include_lengths=True,
                               lower=True,
                               fix_length=config.max_len_context
                               )
        tgt_field = data.Field(tokenize=tokenizer,
                               sequential=True,
                               init_token=SOS_WORD,
                               eos_token=EOS_WORD,
                               pad_token=PAD_WORD,
                               include_lengths=True,
                               batch_first=True,
                               fix_length=config.max_len_question)

        return src_field, tgt_field

    def save_data(self, data_file, dataset):
        examples = vars(dataset)['examples']
        dataset = {'examples': examples}

        torch.save(dataset, data_file)


if __name__ == "__main__":
    pro = DataPreprocessor()
    pro.preprocess([config.train_src, config.test_src], [config.train_tgt, config.test_tgt],
                   os.path.join(config.out_file, "train_dataset.pt"), os.path.join(config.out_file, "test_dataset.pt"))
