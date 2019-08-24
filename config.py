exp = "exp_0"



train_src = "./Data/src-train.txt"
train_tgt = "./Data/tgt-train.txt"
test_src = "./Data/tgt-train.txt"
test_tgt = "./Data/tgt-test.txt"
out_file = "/home/zhangchenbin/code2/learning_to_ask_question/Data/"
glove = "/hdd2/zhangchenbin/data/embedding"


max_len_context = 200
max_len_question = 100
in_vocab_size = 45000
out_vocab_size = 28000
word_embedding_size = 300
question_embedding_size = 300

### model params
cuda = True
batch_size = 2
learning_rate = 1.0
pretrained = False
num_epochs = 100
start_decay_epoch = 8
decay_rate = 0.5
drop_prob = 0.3
embedding_drop_prob = 0.2
embedding_freeze = False
hidden_size = 600
num_layers = 2