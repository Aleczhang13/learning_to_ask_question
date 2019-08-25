# external libraries
import torch
import os

# internal utilities
from dataprocessor import DataPreprocessor
from nltk.translate.bleu_score import sentence_bleu
from torchtext import data
from model import Seq2Seq
import config
from tqdm import tqdm
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Preprocessing values used for training
prepro_params = {
    "word_embedding_size": config.word_embedding_size,
    "question_embedding_size": config.question_embedding_size,

    "max_len_context": config.max_len_context,
    "max_len_question": config.max_len_question,
}

# Hyper-parameters setup
hyper_params = {
    "eval_batch_size": config.eval_batch_size,
    "hidden_size": config.hidden_size,
    "num_layers": config.num_layers,
    "drop_prob": config.drop_prob,
    "cuda": config.cuda,
    "min_len_question": config.min_len_question,
    "max_len_question": config.max_len_question,
    "top_k": config.top_k,
    "top_p": config.top_p,
    "temperature": config.temperature,
    "decode_type": config.decode_type
}

# Train on GPU if CUDA variable is set to True (a GPU with CUDA is needed to do so)
device = torch.device("cuda" if hyper_params["cuda"] else "cpu")
torch.manual_seed(42)
experiment_path = "output/{}".format(config.exp)

# Preprocess the data
dp = DataPreprocessor()
_, _, vocabs = dp.load_data(os.path.join(config.out_file, "train_dataset.pt"),
                            os.path.join(config.out_file, "dev_dataset.pt"),
                            config.glove)

# Load the data into datasets of mini-batches
test_dataset = dp.generate_dataset_2(os.path.join(config.out_file, "src-test.txt"),os.path.join(config.out_file, "tgt-test.txt"),
                            )

test_dataloader = data.BucketIterator(test_dataset,
                                      batch_size=hyper_params["eval_batch_size"],
                                      sort_key=lambda x: len(x.src),
                                      shuffle=False)


# Load the model
model = Seq2Seq(src_vocab=vocabs["src_vocab"],
                hidden_size=hyper_params["hidden_size"],
                num_layers=hyper_params["num_layers"],
                tgt_vocab=vocabs['tgt_vocab'],
                device=device,
                drop_out=hyper_params["drop_prob"])

# Load the model weights resulting from training
if not hyper_params["cuda"]:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"),
                                     map_location=lambda storage, loc: storage)["state_dict"])
else:
    model.load_state_dict(torch.load(os.path.join(experiment_path, "model.pkl"))["state_dict"])
model.to(device)

# Enter evaluation loop
model.eval()
pred_question = []
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_dataloader)):
        # Load a batch of input sentence, sentence lengths and questions
        sentence, len_sentence, question, len_question = batch.src[0].to(device), batch.src[1].to(device), batch.tgt[0].to(device),batch.tgt[1].to(device)
        # answer = batch.feat.to(device) if hyper_params["use_answer"] else None
        # Forward pass to get output/logits
        pred = model(sentence, len_sentence)
        # Convert the predicted indexes to words
        qustion_orign = question.cpu().numpy().tolist()
        orign = [vocabs["tgt_vocab"].itos[i] for i in qustion_orign[0] if vocabs["tgt_vocab"].itos[i]]
        orign = orign[1:len_question-1]
        pred = [vocabs["tgt_vocab"].itos[i] for i in pred[0] if vocabs["tgt_vocab"].itos[i]]
        pred = pred[1:-1]
        pred_question.append(pred)


save_path = os.path.join("./save_pred.txt")
file = open(save_path, 'w')

for example in tqdm(pred_question):
    example = " ".join(example)
    file.write(str(example))
    file.write('\n')

file.close()




