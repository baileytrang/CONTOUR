train_data_path = "data/chunked/train/train_*"
valid_data_path = "data/chunked/valid/valid_*"
test_data_path = "data/chunked/test/test_*"
test_x_data_path = "data/chunked/train_x/train_*"
vocab_path = "data/vocab"
glove_path = "data/glove.42B.300d.txt"


# Hyperparameters
hidden_dim = 256
hidden_layers = 1
emb_dim = 128
emb_type = "glove"
batch_size = 16
max_enc_steps = 800
max_dec_steps = 100
beam_size = 4
min_dec_steps = 20
vocab_size = 20000
max_epoch = 5
lr = 0.001
lamda = 1.25
intra_encoder = True
intra_decoder = True
max_first = "no"
llint = "no"
dth = 25
gth = 0.25
lc = "no"
detail = "no"
threshold = 0.15
use_focus = "yes"
spot_light = "yes"
use_filter = "yes"
focus_embed_size = 16
n_mixture = 3

save_each = 5
print_each_epoch = 5
total_doc = 20000
num_test = 200

rand_unif_init_mag = 0.02
trunc_norm_init_std = 1e-4

we = None

eps = 1e-12

model_name = "something"
save_model_path = "something"
dir_main = "something"

START_DECODING = "[CLS]"
STOP_DECODING = "[SEP]"
SENTENCE_START = "[CLS]"
SENTENCE_END = "[SEP]"
