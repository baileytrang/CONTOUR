import pytz
from tqdm import tqdm
import argparse
from numpy import random
from rouge import Rouge
from torch.distributions import Categorical
from train_util import *
from eval_epoch import Evaluate_epoch
from data_util.data import Vocab
from data_util.batcher import Batcher
from data_util import config, data
from model import Model
import torch.nn.functional as F
import torch.nn as nn
import torch as T
import time
import os
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set cuda device
device = 'cuda' if T.cuda.is_available() else 'cpu'

random.seed(123)
T.manual_seed(123)
if T.cuda.is_available():
    T.cuda.manual_seed_all(123)


def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]

    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, T.Tensor):
        B, *size = tensor.size()
        repeat_size = [1] + [K] + [1] * (tensor.dim() - 1)
        tensor = tensor.unsqueeze(1).repeat(*repeat_size).view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out


class Train(object):
    def __init__(self, opt):
        print("Start __init__ Train")
        print("Start __init__ Train vocab")
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        print("Start __init__ Train batcher")
        self.batcher = Batcher(config.train_data_path, self.vocab, mode='train', batch_size=config.batch_size, single_pass=False)
        self.opt = opt
        self.start_id = self.vocab.word2id(config.START_DECODING)
        self.end_id = self.vocab.word2id(config.STOP_DECODING)
        self.pad_id = self.vocab.word2id(data.PAD_TOKEN)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        self.best_rl = 0
        time.sleep(5)

    def save_model(self, iter, tag):
        e = round(iter / self.total_iter, 2)
        phase = "p1"
        if (self.opt.train_rl == "yes"):
            phase = "p2"

        save_path = config.save_model_path + "/models" + "/" + phase + "_best.tar"

        f = open(config.save_model_path + "/best_info.txt", "a")
        f.write(phase + "_" + str(e) + "  " + str(self.best_rl) + "\n")
        f.close()

        if os.path.exists(save_path):
            os.remove(save_path)

        T.save({
            "best_rl": self.best_rl,
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, save_path)
        return e

    def save_latest_model(self, iter):
        e = round(iter / self.total_iter, 2)
        phase = "p1"
        if (self.opt.train_rl == "yes"):
            phase = "p2"

        if config.detail == "yes":
            phase = phase + "_" + str(e)

        latest_path = config.save_model_path + "/models/" + phase + "_latest.tar"

        if os.path.exists(latest_path):
            os.rename(latest_path, config.save_model_path + "/models/" + phase + "_prev.tar")

        T.save({
            "best_rl": self.best_rl,
            "iter": iter + 1,
            "model_dict": self.model.state_dict(),
            "trainer_dict": self.trainer.state_dict()
        }, latest_path)

    def setup_train(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        self.trainer = T.optim.Adam(self.model.parameters(), lr=config.lr)
        start_iter = 0
        if self.opt.load_model is not None:
            load_model_path = os.path.join(
                config.save_model_path + "/models", self.opt.load_model)
            checkpoint = T.load(load_model_path)
            start_iter = checkpoint["iter"]
            self.model.load_state_dict(checkpoint["model_dict"])
            self.trainer.load_state_dict(checkpoint["trainer_dict"])
            print("Loaded model at " + load_model_path)

        return start_iter

    def train_batch_MLE(self, focus_mask, enc_out, enc_hidden, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, batch):
        ''' Calculate Negative Log Likelihood Loss for the given batch. In order to reduce exposure bias,
                pass the previous generated token as input with a probability of 0.25 instead of ground truth label
        Args:
        :param enc_out: Outputs of the encoder for all time steps (batch_size, length_input_sequence, 2*hidden_size)
        :param enc_hidden: Tuple containing final hidden state & cell state of encoder. Shape of h & c: (batch_size, hidden_size)
        :param enc_padding_mask: Mask for encoder input; Tensor of size (batch_size, length_input_sequence) with values of 0 for pad tokens & 1 for others
        :param ct_e: encoder context vector for time_step=0 (eq 5 in https://arxiv.org/pdf/1705.04304.pdf)
        :param extra_zeros: Tensor used to extend vocab distribution for pointer mechanism
        :param enc_batch_extend_vocab: Input batch that stores OOV ids
        :param batch: batch object
        '''
        dec_batch, max_dec_len, dec_lens, target_batch = get_dec_data(batch)  # Get input and target batchs for training decoder
        step_losses = []
        s_t = (enc_hidden[0], enc_hidden[1])  # Decoder hidden states
        x_t = get_cuda(T.LongTensor(len(enc_out)).fill_(self.start_id))  # Input to the decoder
        # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        prev_s = None
        # Used for intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        sum_temporal_srcs = None
        # predicted_ids = [[] for _ in range(0, config.batch_size)]
        for t in range(min(max_dec_len, config.max_dec_steps)):
            # Probabilities indicating whether to use ground truth labels instead of previous decoded tokens
            use_gound_truth = get_cuda((T.rand(len(enc_out)) > config.gth)).long()
            # Select decoder input based on use_ground_truth probabilities
            x_t = use_gound_truth * dec_batch[:, t] + (1 - use_gound_truth) * x_t

            id_x = x_t
            x_t = self.model.embeds(x_t)

            final_dist, s_t, ct_e, sum_temporal_srcs, prev_s = self.model.decoder(
                focus_mask, id_x, x_t, s_t, enc_out, enc_padding_mask, ct_e, extra_zeros, enc_batch_extend_vocab, sum_temporal_srcs, prev_s, dec_lens)
            target = target_batch[:, t]
            log_probs = T.log(final_dist + config.eps)
            step_loss = F.nll_loss(log_probs, target, reduction="none", ignore_index=self.pad_id)

            step_losses.append(step_loss)
            # Sample words from final distribution which can be used as input in next time step
            if (config.max_first != 'no'):
                x_t = T.multinomial(final_dist, 1).squeeze()
            else:
                _, x_t = T.max(final_dist, dim=1)
                # Mask indicating whether sampled word is OOV
            is_oov = (x_t.detach() >= config.vocab_size).detach().long()
            x_t = (1 - is_oov) * x_t.detach() + (is_oov) * self.unk_id
            # Replace OOVs with [UNK] token

        # unnormalized losses for each example in the batch; (batch_size)
        losses = T.sum(T.stack(step_losses, 1), 1)
        batch_avg_loss = losses / dec_lens  # Normalized losses; (batch_size)
        mle_loss = T.mean(batch_avg_loss)  # Average batch loss

        return mle_loss

    def train_one_batch(self, batch, iter):
        enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, context, focus_mask, focus_input = get_enc_data(
            batch)

        # -------------------------------Selector-----------------------
        source_WORD_encoding = enc_batch
        source_WORD_encoding_extended = enc_batch_extend_vocab

        # ============#
        # Focus Loss #
        # ============#
        # if config.use_focus and not config.eval_focus_oracle:
        if self.opt.load_mixture == "None":
            # ================================#
            # Hard-EM
            # 1) Select a minimum-loss SELECTOR expert (E-Step)
            # 2) Train with the selected SELECTOR expert (M-Step)
            # ================================#

            if config.n_mixture == 1:
                B = source_WORD_encoding.size(0)
                mixture_id = T.zeros(B, dtype=T.long, device=device)
            elif config.use_focus == "yes":
                # 1) Select a minimum-loss SELECTOR expert (E-Step)
                self.model.selector.eval()
                with T.no_grad():

                    # [B * n_mixture, L]
                    focus_logit = self.model.selector(source_WORD_encoding, mixture_id=None, focus_input=focus_input, train=True)

                    B, L = source_WORD_encoding.size()

                    # [B * n_mixture, L]
                    repeated_target = repeat(focus_mask.float(), config.n_mixture)

                    # [B * n_mixture, L]
                    focus_loss = F.binary_cross_entropy_with_logits(focus_logit, repeated_target, reduction='none').view(B, config.n_mixture, L)
                    pad_mask = (source_WORD_encoding == 1).view(B, 1, L)
                    mixture_id = focus_loss.masked_fill(pad_mask, 0).sum(dim=2).argmin(dim=1)

            # 2) Train with the selected SELECTOR expert (M-Step)
            self.model.selector.train()
            # [B, L]
            focus_logit = self.model.selector(
                source_WORD_encoding,
                mixture_id=mixture_id,
                focus_input=focus_input,
                train=True)

            # [B, L]
            focus_loss = F.binary_cross_entropy_with_logits(focus_logit, focus_mask.float(), reduction='none')

            pad_mask = source_WORD_encoding == 1
            valid_mask = ~pad_mask

            focus_len = valid_mask.float().sum(dim=1)

            focus_loss.masked_fill_(pad_mask, 0)
            focus_loss = focus_loss.sum(dim=1) / focus_len
            focus_loss = focus_loss.mean()

        else:
            # No need to train Selector
            focus_loss = T.zeros(1).squeeze().to(device)

        # Get embeddings for encoder input
        tmp = self.model.embeds(enc_batch)
        enc_out, enc_hidden = self.model.encoder(tmp, enc_lens, focus_mask)

        # -------------------------------Summarization-----------------------
        if self.opt.train_mle == "yes":  # perform MLE training
            mle_loss = self.train_batch_MLE(focus_mask, enc_out, enc_hidden, enc_padding_mask, context, extra_zeros, enc_batch_extend_vocab, batch)
        else:
            mle_loss = get_cuda(T.FloatTensor([0]))

    # ------------------------------------------------------------------------------------
        self.trainer.zero_grad()
        (focus_loss + mle_loss).backward()
        self.trainer.step()

        return mle_loss.item()

    def trainIters(self):
        print("Start trainIters")

        self.total_iter = config.total_doc // config.batch_size

        save_every = self.total_iter // config.save_each
        print_every = self.total_iter // config.print_each_epoch
        max_iterations = self.total_iter * config.max_epoch
        iter = self.setup_train()
        pbar = tqdm(total=self.total_iter)
        count = mle_total = r_total = 0

        if not os.path.exists(config.save_model_path + "/avg_rounge.csv"):
            f = open(config.save_model_path + "/avg_rounge.csv", 'w')
            f.write("model name, start, end, R1, R2, RL\n")
            f.close()

        if start_rl:
            iter = 0
            self.best_rl = 0
        else:
            for _ in range(iter % self.total_iter):
                batch = self.batcher.next_batch()
                pbar.update(1)

        while iter <= max_iterations:
            batch = self.batcher.next_batch()
            try:
                mle_loss = self.train_one_batch(batch, iter)
            except KeyboardInterrupt:
                print("-------------------Keyboard Interrupt------------------")
                exit(0)

            mle_total += mle_loss
            count += 1
            iter += 1

            if iter % print_every == 0:
                e = round(iter / self.total_iter, 2)
                self.save_latest_model(iter)
                mle_avg = mle_total / count
                tqdm.write("\nepoch: " + str(e) + " mle_loss: " + str(mle_avg)[:5])
                f = open(config.save_model_path + "/loss.txt", 'a')
                f.write(str(e) + " " + str(mle_avg) + '\n')
                f.close()
                count = mle_total = 0

            pbar.update(1)
            # if iter % save_every == 0:
            #     f = open(config.save_model_path + "/aresult.csv", 'a')
            #     current_time = datetime.datetime.now(pytz.timezone('Asia/Bangkok'))
            #     e = round(iter / self.total_iter, 2)
            #     f.write(str(e) + "," + str(current_time) + ",")

                # evaluate_epoch = Evaluate_epoch(config.valid_data_path)
                # r1, r2, rl = evaluate_epoch.evaluate_batch(self.model)

                # current_time = datetime.datetime.now(
                #     pytz.timezone('Asia/Bangkok'))
                # f.write(str(current_time) + "," + str(r1) + "," +
                #         str(r2) + "," + str(rl) + "\n")
                # f.close()

                # f = open(config.dir_main + "/atrainresult.csv", 'a')
                # f.write(str(current_time) + "," + ",".join(
                #     [opt.model_name, str(e), str(r1), str(r2), str(rl)])+"\n")
                # f.close()

            if (iter % self.total_iter == 0):
                pbar.n = 0


start_rl = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mle', type=str, default="yes")
    parser.add_argument('--train_rl', type=str, default="no")
    parser.add_argument('--max_first', type=str, default="no")
    parser.add_argument('--mle_weight', type=float, default=1.0)
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--lamda', type=float, default=1.25)
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--load_mixture', type=str, default="None")
    parser.add_argument('--new_lr', type=float, default=None)
    parser.add_argument('--hidden_dim', type=int, default=None)
    parser.add_argument('--hidden_layers', type=int, default=None)
    parser.add_argument('--emb_dim', type=int, default=None)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--n_mixture', type=int, default=3)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--emb_type', type=str, default=None)
    parser.add_argument('--max_enc_steps', type=int, default=None)
    parser.add_argument('--max_dec_steps', type=int, default=None)
    parser.add_argument('--min_dec_steps', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--gth', type=float, default=0.25)
    parser.add_argument('--beam_size', type=int, default=None)
    parser.add_argument('--dth', type=int, default=25)
    parser.add_argument('--model_name', type=str, default="cur_model")
    parser.add_argument('--lc', type=str, default="no")
    parser.add_argument('--detail', type=str, default="no")
    parser.add_argument('--use_focus', type=str, default="yes")
    parser.add_argument('--use_filter', type=str, default="yes")
    parser.add_argument('--spot_light', type=str, default="yes")
    parser.add_argument('--intra_encoder', type=bool, default=True)
    parser.add_argument('--intra_decoder', type=bool, default=True)

    opt = parser.parse_args()
    opt.rl_weight = 1 - opt.mle_weight
    path_root = config.save_model_path

    if opt.train_rl == "yes" and "p1" in opt.load_model:
        start_rl = True

    config.save_model_path = config.save_model_path + "/" + opt.model_name
    print(config.save_model_path)
    print("Training mle: %s, Training rl: %s, mle weight: %.2f, rl weight: %.2f" % (opt.train_mle, opt.train_rl, opt.mle_weight, opt.rl_weight))
    print("intra_encoder:", config.intra_encoder, "intra_decoder:", config.intra_decoder)

    if ("livedoor" in opt.model_name):
        config.total_doc = 6630
        config.print_each_epoch = 2
        config.save_each = 1
    elif "vntc" in opt.model_name:
        config.total_doc = 34503
        config.print_each_epoch = 5
        config.save_each = 5
    else:
        if opt.train_rl == "yes":
            config.total_doc = 287227
            config.print_each_epoch = 50
            config.save_each = 10
        else:
            config.total_doc = 287227
            config.print_each_epoch = 20
            config.save_each = 10

    if (opt.train_rl == 'no'):
        if ("livedoor" in opt.model_name):
            config.batch_size = 4
        elif "vntc" in opt.model_name:
            config.batch_size = 16
        else:
            config.batch_size = 32
    else:
        if ("livedoor" in opt.model_name):
            config.batch_size = 4
        elif "vntc" in opt.model_name:
            config.batch_size = 8
        else:
            config.batch_size = 16

    if not os.path.exists(config.save_model_path):
        os.mkdir(config.save_model_path)
    else:
        if "runable" not in opt.model_name and opt.load_model == None:
            print("Duplicated model name. Continue? [N]")
            ok = input()
            if (ok == "N" or ok == 'n'):
                exit()

    if not os.path.exists(config.save_model_path + "/models"):
        os.mkdir(config.save_model_path + "/models")

    if (opt.hidden_dim != None):
        config.hidden_dim = opt.hidden_dim

    config.max_first = opt.max_first
    config.dth = opt.dth
    config.gth = opt.gth
    config.detail = opt.detail
    config.threshold = opt.threshold
    config.use_focus = opt.use_focus
    config.n_mixture = opt.n_mixture
    config.intra_encoder = opt.intra_encoder
    config.intra_decoder = opt.intra_decoder
    config.use_filter = opt.use_filter
    config.spot_light = opt.spot_light
    config.lamda = opt.lamda
    config.model_name = opt.model_name

    if (opt.hidden_layers != None):
        config.hidden_layers = opt.hidden_layers

    if (opt.batch_size != None):
        config.batch_size = opt.batch_size

    if (opt.emb_dim != None):
        config.emb_dim = opt.emb_dim

    if (opt.emb_type != None):
        config.emb_type = opt.emb_type
        if config.emb_type in ["w2v", "glove"]:
            config.emb_dim = 300

    if (opt.max_enc_steps != None):
        config.max_enc_steps = opt.max_enc_steps
    else:
        if ("livedoor" in opt.model_name):
            config.max_enc_steps = 400
        elif "vntc" in opt.model_name:
            config.max_enc_steps = 850
        else:
            config.max_enc_steps = 800

    if (opt.max_dec_steps != None):
        config.max_dec_steps = opt.max_dec_steps
    else:
        if ("livedoor" in opt.model_name):
            config.max_dec_steps = 40
        elif "vntc" in opt.model_name:
            config.max_dec_steps = 10
        else:
            config.max_dec_steps = 100

    if (opt.beam_size != None):
        config.beam_size = opt.beam_size

    if (opt.min_dec_steps != None):
        config.min_dec_steps = opt.min_dec_steps
    else:
        if ("livedoor" in opt.model_name):
            config.min_dec_steps = 10
        elif "vntc" in opt.model_name:
            config.min_dec_steps = 4
        else:
            config.min_dec_steps = 20

    if (opt.vocab_size != None):
        config.vocab_size = opt.vocab_size

    if (opt.max_epoch != None):
        config.max_epoch = opt.max_epoch

    if config.use_focus == "no":
        config.focus_embed_size = 0

    with open(config.save_model_path + '/json_config.txt', 'w') as outfile:
        outfile.write("hidden_dim = " + str(config.hidden_dim) + '\n')
        outfile.write("hidden_layers = " + str(config.hidden_layers) + '\n')
        outfile.write("emb_dim = " + str(config.emb_dim) + '\n')
        outfile.write("emb_type = " + '"' + config.emb_type + '"' + '\n')
        outfile.write("batch_size = " + str(config.batch_size) + '\n')
        outfile.write("threshold = " + str(config.threshold) + '\n')
        outfile.write("max_enc_steps = " + str(config.max_enc_steps) + '\n')
        outfile.write("max_dec_steps = " + str(config.max_dec_steps) + '\n')
        outfile.write("min_dec_steps = " + str(config.min_dec_steps) + '\n')
        outfile.write("focus_embed_size = " + str(config.focus_embed_size) + '\n')
        outfile.write("beam_size = " + str(config.beam_size) + '\n')
        outfile.write("vocab_size = " + str(config.vocab_size) + '\n')
        outfile.write("n_mixture = " + str(config.n_mixture) + '\n')
        outfile.write("max_epoch = " + str(config.max_epoch) + '\n')
        outfile.write("mle_weight = " + str(opt.mle_weight) + '\n')
        outfile.write("lamda = " + str(opt.lamda) + '\n')
        outfile.write("max_first = " + '"' + str(opt.max_first) + '"' + '\n')
        outfile.write("model_name = " + '"' + opt.model_name + '"' + '\n')
        outfile.write("load_mixture = " + '"' + opt.load_mixture + '"' + '\n')
        outfile.write("intra_encoder = " + str(config.intra_encoder) + '\n')
        outfile.write("intra_decoder = " + str(config.intra_decoder) + '\n')
        outfile.write("use_focus = " + '"' + opt.use_focus + '"' + '\n')
        outfile.write("use_filter = " + '"' + opt.use_filter + '"' + '\n')
        outfile.write("spot_light = " + '"' + opt.spot_light + '"' + '\n')
        outfile.write("gth = " + str(config.gth) + '\n')
        outfile.write("lc = " + '"' + str(opt.lc) + '"' + '\n')

    train_processor = Train(opt)
    train_processor.trainIters()
