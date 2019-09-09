# 使用相应的log进行记录
import logging
import os

import nltk
import torch
from torch.nn import init
from torchtext import data, vocab
from tqdm import tqdm

nltk.download('stopwords')

import config
from utils import tokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SOS_WORD = '<SOS>'
EOS_WORD = '<EOS>'
PAD_WORD = '<PAD>'

class new_vocab(vocab.Vectors):
    def __getitem__(self, token):
        if token in self.stoi:
            return self.vectors[self.stoi[token]]
        else:
            return self.unk_init(torch.Tensor(1,self.dim)).squeeze(0)


# Progress relate dataset
class Squadataset_process(data.Dataset):
    def __init__(self, src_path, tgt_path, fields, max_len=None):

        examples = []
        # 设置相应的fields
        fields = [('src', fields[0]), ('tgt', fields[1])]

        with open(src_path, encoding="utf-8") as src_file, open(tgt_path, encoding="utf-8") as tgt_file:
            src_file = src_file.readlines()
            tgt_file = tgt_file.readlines()

            for src_line, tgt_line in tqdm(tuple(zip(src_file, tgt_file))):
                # src_sentence = tokenize_sentence(src_line)
                # tgt_sentence = tokenize_sentence(tgt_line)
                # 去除尝试
                src_sentence = src_line
                tgt_sentence = tgt_line
                # 这一步为了保证之后长度输入是一样的
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
        # ---> 建立起相应的数据集
        logger.info("Preprocessing dataset...")
        train_data, test_data = self.generate_dataset(src_path, tgt_path, max_len)
        logger.info("Saving train dataset...")
        self.save_data(train_save_path, train_data)
        logger.info("Saving test dataset...")
        self.save_data(test_save_path, test_data)

        # 建立字典
        self.src_field.build_vocab(train_data, max_size=config.in_vocab_size)
        self.tgt_field.build_vocab(test_data, max_size=config.out_vocab_size)

        src_vocab, tgt_vocab = self.generate_vocabs()
        vocabs = {'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}

        return train_data, test_data, vocabs

    def load_data(self, train_save_path, test_save_path, glove_dir):
        # loading saved data
        train_dataset = torch.load(train_save_path)
        train_examples = train_dataset['examples']

        test_dataset = torch.load(test_save_path)
        test_examples = test_dataset['examples']

        # generating torchtext dataset class
        fields = [('src', self.src_field), ('tgt', self.tgt_field)]
        train_dataset = data.Dataset(fields=fields, examples=train_examples)
        test_dataset = data.Dataset(fields=fields, examples=test_examples)

        # loading GloVE vectors
        vec = new_vocab(os.path.join(glove_dir, "glove.840B.{}d.txt".format(config.word_embedding_size)))
        #
        vec.unk_init = init.xavier_uniform_
        # building field vocabulary
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

    def generate_dataset_2(self, src_path, tgt_path, max_len=None):
        train_data = Squadataset_process(src_path=src_path,
                                         tgt_path=tgt_path,
                                         fields=(self.src_field, self.tgt_field),
                                         max_len=max_len)

        dataset = train_data

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
                               fix_length=config.max_len_context,
                               # stop_words = stopwords.words("english")
                               )
        tgt_field = data.Field(tokenize=tokenizer,
                               sequential=True,
                               init_token=SOS_WORD,
                               eos_token=EOS_WORD,
                               pad_token=PAD_WORD,
                               include_lengths=True,
                               batch_first=True,
                               lower=True,
                               fix_length=config.max_len_question,
                               # stop_words = stopwords.words("english")
                               )

        return src_field, tgt_field

    def save_data(self, data_file, dataset):
        examples = vars(dataset)['examples']
        dataset = {'examples': examples}

        torch.save(dataset, data_file)




if __name__ == "__main__":
    pro = DataPreprocessor()
    pro.preprocess([config.train_src, config.dev_src], [config.train_tgt, config.dev_tgt],
                   os.path.join(config.out_file, "train_dataset.pt"), os.path.join(config.out_file, "dev_dataset.pt"))
