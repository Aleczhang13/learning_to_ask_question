exp = "exp_2_att"



train_src = "./Data/src-train.txt"
train_tgt = "./Data/tgt-train.txt"

dev_src = "./Data/src-dev.txt"
dev_tgt = "./Data/tgt-dev.txt"

out_file = "/home/zhangchenbin/code2/learning_to_ask_question/Data/"
glove = "/hdd2/zhangchenbin/data/embedding"


max_len_context = 100
max_len_question = 30
min_len_question = 5
in_vocab_size = 45000
out_vocab_size = 28000
word_embedding_size = 300
question_embedding_size = 300

### model params
cuda = True
batch_size = 64
learning_rate = 1.0
pretrained = False
num_epochs = 20
start_decay_epoch = 8
decay_rate = 0.5
drop_prob = 0.3
embedding_drop_prob = 0.3
embedding_freeze = False
hidden_size = 600
num_layers = 2
min_length = 5
max_length = 30
top_k = 0.
top_p = 0.9
temperature = 0.7
decode_type = "topk"
att_need = True

# eval_params
eval_batch_size = 1
