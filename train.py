import json
import logging
import os

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torchtext import data
from tqdm import tqdm

import config
from dataprocessor import DataPreprocessor
from model import Seq2Seq
from utils import MetricReporter, dress_for_loss, correct_tokens, save_checkpoint

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "question_embedding_size": config.question_embedding_size,
    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
}

hyper_params = {
    "cuda": config.cuda,
    "batch_size": config.batch_size,
    "pretrained": config.pretrained,
    "learning_rate": config.learning_rate,
    "num_epochs": config.num_epochs,
    "start_decay_epoch": config.start_decay_epoch,
    "decay_rate": config.decay_rate,
    "drop_prob": config.drop_prob,
    "hidden_size": config.hidden_size,
    "num_layers": config.num_layers
    # TO DO
}

experiment_params = {"preprocessing": prepro_params, "model": hyper_params}
# seting the cuda
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(35)

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
train_dataset, dev_dataset, vocabs = dp.load_data(os.path.join(config.out_file, "train_dataset.pt"),
                                                  os.path.join(config.out_file, "dev_dataset.pt"),
                                                  config.glove)

# using the dataloader to feed data
train_dataloader = data.BucketIterator(train_dataset,
                                       batch_size=hyper_params["batch_size"],
                                       sort_key=lambda x: len(x.src),
                                       sort_within_batch=True,
                                       device=device,
                                       shuffle=True)

dev_dataloader = data.BucketIterator(dev_dataset,
                                     batch_size=hyper_params["batch_size"],
                                     sort_key=lambda x: len(x.src),
                                     sort_within_batch=True,
                                     device=device,
                                     shuffle=True)

logger.info("Length of training data loader is:{}".format(len(train_dataloader)))
logger.info("Length of valid data loader is:{}".format(len(dev_dataloader)))

# TO DO
model = Seq2Seq(src_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                num_layers=hyper_params["num_layers"],
                tgt_vocab=vocabs['tgt_vocab'],
                device=device,
                drop_out=hyper_params["drop_prob"])

model.to(device)
# resume training if checkpoint
if hyper_params["pretrained"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])

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

# create an object to report the different metrics
mc = MetricReporter()

# train the model
logging.info("Starting training...")
for epoch in range(hyper_params["num_epochs"]):
    logging.info("##### epoch {:2d}".format(epoch))
    model.train()
    mc.train()
    print("lr is", scheduler.get_lr())
    for i, batch in enumerate(tqdm(train_dataloader)):
        sentence, len_sentence, question, len_question = batch.src[0].to(device), batch.src[1].to(device), batch.tgt[
            0].to(device), batch.tgt[1].to(device)
        optimizer.zero_grad()

        pred = model(sentence, len_sentence, question, len_question)
        pred = dress_for_loss(pred)
        # calculate Loss
        loss = criterion(pred.view(-1, pred.size(2)), question[:, 1:].contiguous().view(-1))

        # update the metrics
        num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
        mc.update_metrics(loss.item(), num_non_padding, num_correct)

        loss.backward()

        clip_grad_norm_(model.parameters(), 5.)

        optimizer.step()
    scheduler.step()

    # compute the loss, accuracy and perplexity for this epoch and push them to TensorboardX
    mc.report_metrics()
    writer.add_scalars("train", {"loss": mc.list_train_loss[-1],
                                 "accuracy": mc.list_train_accuracy[-1],
                                 "perplexity": mc.list_train_perplexity[-1],
                                 "epoch": mc.epoch})

    model.eval()
    mc.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dev_dataloader)):
            # load a batch of input sentence, sentence lengths and questions
            sentence, len_sentence, question, len_question = batch.src[0].to(device), batch.src[1].to(device), \
                                                             batch.tgt[0].to(device), batch.tgt[1].to(device)

            pred = model(sentence, len_sentence, question, len_question)

            pred = dress_for_loss(pred)

            loss = criterion(pred.view(-1, pred.size(2)), question[:, 1:].contiguous().view(-1))

            num_non_padding, num_correct = correct_tokens(pred, question, padding_idx)
            mc.update_metrics(loss.item(), num_non_padding, num_correct)

        mc.report_metrics()
        writer.add_scalars("valid", {"loss": mc.list_valid_loss[-1],
                                     "accuracy": mc.list_valid_accuracy[-1],
                                     "perplexity": mc.list_valid_perplexity[-1],
                                     "epoch": mc.epoch})

    # save last model weights
    save_checkpoint({
        "epoch": mc.epoch + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": mc.list_valid_loss[-1],
    }, True, os.path.join(experiment_path, "model_last_checkpoint.pkl"))

    # save model weights with best validation error
    is_best = bool(mc.list_valid_loss[-1] < best_valid_loss)
    best_valid_loss = min(mc.list_valid_loss[-1], best_valid_loss)
    save_checkpoint({
        "epoch": mc.epoch + epoch_checkpoint,
        "state_dict": model.state_dict(),
        "best_valid_loss": best_valid_loss
    }, is_best, os.path.join(experiment_path, "model.pkl"))

# export scalar data to TXT file for external processing and analysis
mc.log_metrics(os.path.join(experiment_path, "train_log.txt"))
