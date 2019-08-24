import json
import logging
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torchtext import data

import config
from dataprocessor import DataPreprocessor
from model import Seq2Seq
from utils import MetricReporter

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "answer_embedding_size": config.question_embedding_size,
    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
}

hyper_params = {
    "cuda": config.cuda,
    "batch_size": config.batch_size,
    "pretrained": config.pretrained,
    "learning_rate": config.learning_rate,
    "num_epochs":config.num_epochs,
    "start_decay_epoch":config.start_decay_epoch,
    "decay_rate":config.decay_rate,
    "drop_prob":config.drop_prob,
    "hidden_size":config.hidden_size,
    "num_layers":config.num_layers
    # TO DO
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}
# seting the cuda
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)

# seting the a path to save the experiment log
experiment_path = "./output/{}".format(config.exp)
if not os.path.exists(experiment_path):
    os.mkdir(experiment_path)

# save the preprocesisng and model parameters used for this training experiment
with open(os.path.join(experiment_path, "config_{}.json".format(config.exp)), "w") as f:
    json.dump(experiment_params, f)

# start TensorboardX writer
writer = SummaryWriter(experiment_path)

# preprocess the data
dp = DataPreprocessor()
train_dataset, test_dataset, vocabs = dp.load_data(os.path.join(config.out_file, "train_dataset.pt"),
                                                   os.path.join(config.out_file, "test_dataset.pt"),
                                                   config.glove)

# using the dataloader to feed data
train_dataloader = data.BucketIterator(train_dataset,
                                       batch_size=hyper_params["batch_size"],
                                       sort_key=lambda x: len(x.src),
                                       sort_within_batch=True,
                                       device=device,
                                       shuffle=False)

test_dataloader = data.BucketIterator(test_dataset,
                                      batch_size=hyper_params["batch_size"],
                                      sort_key=lambda x: len(x.src),
                                      sort_within_batch=True,
                                      device=device,
                                      shuffle=False)

logger.info("Length of training data loader is:", len(train_dataloader))

logger.info("Length of valid data loader is:", len(test_dataloader))

# TO DO
model = Seq2Seq(src_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                num_layers= hyper_params["num_layers"],
                tgt_vocab=vocabs['tgt_vocab'],
                device=device,
                drop_out=hyper_params["drop_prob"])


# Resume training if checkpoint
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

# set the loss function and reduce the mask loss
padding_idx = vocabs['tgt_vocab'].stoi["<PAD>"]
criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), hyper_params["learning_rate"])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=list(range(hyper_params["start_decay_epoch"],
                                                                       hyper_params["num_epochs"] + 1)),
                                                 gamma=hyper_params["decay_rate"])

# save the best model to continue learning
if hyper_params["pretrained"]:
    best_valid_loss = torch.load(os.path.join(experiment_path, "model.pkl"))["best_valid_loss"]
    epoch_checkpoint = torch.load(os.path.join(experiment_path, "model_last_checkpoint.pkl"))["epoch"]
    logger.info("Best validation loss obtained after {} epochs is: {}".format(epoch_checkpoint, best_valid_loss))
else:
    best_valid_loss = 10000  # large number
    epoch_checkpoint = 1

# Create an object to report the different metrics
mc = MetricReporter()


# Train the model
logging.info("Starting training...")
for epoch in range(hyper_params["num_epochs"]):
    logging.info("##### epoch {:2d}".format(epoch))
    # model.train()
    # scheduler.step()
    for i, batch in enumerate(train_dataloader):
        sentence, len_sentence, question, len_question = batch.src[0].to(device), batch.src[1].to(device), batch.tgt[0].to(device),batch.tgt[1].to(device)
        optimizer.zero_grad()

        enc_output, enc_hidden = model(sentence, len_sentence, question, len_question)


