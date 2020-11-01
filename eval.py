import argparse
from rouge import Rouge
from beam_search import *
from train_util import *
from data_util.data import Vocab
from data_util.batcher import Batcher
from data_util import config, data
from model import Model
import torch.nn.functional as F
import torch.nn as nn
import torch as T
import time
import re
import os
from tqdm import tqdm
import random
import statistics
from rouge_score import rouge_scorer
import csv
import datetime
import pytz

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


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


def rouge_batch(a, b):
    score = [[], [], []]
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)

    for i in range(len(a)):
        scores = scorer.score(a[i].replace(". ", ".\n"), b[i].replace(". ", ".\n"))
        score[0].append(scores['rouge1'].fmeasure)
        score[1].append(scores['rouge2'].fmeasure)
        score[2].append(scores['rougeLsum'].fmeasure)

    rouge_1 = sum(score[0])/len(score[0])
    rouge_2 = sum(score[1])/len(score[1])
    rouge_l = sum(score[2])/len(score[2])

    return round(rouge_1, 4) * 100, round(rouge_2, 4) * 100, round(rouge_l, 4) * 100


class Evaluate(object):
    def __init__(self, data_path, opt, batch_size=config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval', batch_size=batch_size, single_pass=True)
        self.opt = opt
        time.sleep(5)

    def setup_valid(self):
        self.model = Model()
        self.model = get_cuda(self.model)
        checkpoint = T.load(os.path.join(config.save_model_path + "/models", self.opt.load_model))
        self.model.load_state_dict(checkpoint["model_dict"])

        if self.opt.load_mixture != None:
            model_loaded = Model()
            model_loaded = get_cuda(model_loaded)
            load_model_path = os.path.join(config.dir_main + "/focus_models/" + self.opt.load_mixture)
            checkpoint = T.load(load_model_path)
            model_loaded.load_state_dict(checkpoint["model_dict"])
            self.model.mixture_embedding = model_loaded.mixture_embedding
            self.model.selector = model_loaded.selector
            self.model.focus_embed = model_loaded.focus_embed
            print("Load:", config.dir_main + "/focus_models/" + self.opt.load_mixture)

    def evaluate_batch(self, print_sents=False):
        self.setup_valid()
        batch = self.batcher.next_batch()
        start_id = self.vocab.word2id(config.START_DECODING)
        end_id = self.vocab.word2id(config.STOP_DECODING)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        pbar = tqdm(total=self.batcher._batch_queue.qsize())

        overall = [[], [], []]
        load_file = self.opt.load_model
        filename = "/" + self.opt.task + "_" + load_file + ".txt"
        f = open(config.save_model_path + filename, 'w')
        count = 0
        if self.opt.log != "no":
            log_info = []  # inlen, tgtlen, outlen, R1, R2, RL
            f_log = open(config.save_model_path + "/" + self.opt.task + "_log_" + self.opt.load_model + ".txt", "w+")
            f_log_csv = open(config.save_model_path + "/" + self.opt.task + "_log_" + self.opt.load_model + "_csv.txt", "w+")
            f_log_csv.write(
                '=SPLIT("In len, Tgt len, Out len, R1, R2, RL", ",")\n')
        try:
            while batch is not None:
                enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e, focus_mask, focus_input = get_enc_data(
                    batch)
                config.batch_size = enc_batch.size()[0]
                source_WORD_encoding = enc_batch
                source_WORD_encoding_extended = enc_batch_extend_vocab
                with T.autograd.no_grad():
                    focus_p = self.model.selector(source_WORD_encoding, train=False)

                    generated_focus_mask = (focus_p > config.threshold).long()
                    input_mask = generated_focus_mask
                    input_mask_fusion = []
                    for i in range(config.batch_size):
                        tmp = focus_p[i*config.n_mixture]
                        for m in range(1, config.n_mixture):
                            tmp += focus_p[i*config.n_mixture + m]
                        input_mask_fusion.append(
                            (tmp > config.threshold*config.n_mixture).long())

                    input_mask_fusion = T.stack(input_mask_fusion)

                    tmp = self.model.embeds(source_WORD_encoding)
                    enc_out, enc_hidden = self.model.encoder(
                        tmp, enc_lens, input_mask_fusion)

                # -----------------------Summarization----------------------------------------------------
                with T.autograd.no_grad():
                    pred_ids = beam_search(input_mask_fusion, enc_hidden, enc_out, enc_padding_mask, ct_e, extra_zeros,
                                           source_WORD_encoding_extended, self.model, start_id, end_id, self.unk_id)

                best_ids = pred_ids

                f = open(config.save_model_path + filename, 'a')
                for i in range(len(best_ids)):
                    abstract = batch.original_abstracts[i]
                    article = batch.original_articles[i]
                    abstract = re.sub("<t> ", "", abstract)
                    abstract = re.sub(" </t>", "", abstract)
                    abstract = re.sub("<s> ", "", abstract)
                    abstract = re.sub(" </s>", "", abstract)

                    if (len(abstract) == 0):
                        abstract = "------------"
                    decoded_words = data.outputids2words(
                        best_ids[i], self.vocab, batch.art_oovs[i])
                    if len(decoded_words) < 2:
                        decoded_words = "xxx"
                    else:
                        decoded_words = " ".join(decoded_words)
                    decoded_words = re.sub("_", " ", decoded_words)
                    decoded_words = re.sub("<t> ", "", decoded_words)
                    decoded_words = re.sub(" </t>", "", decoded_words)
                    decoded_words = re.sub("<s> ", "", decoded_words)
                    decoded_words = re.sub(" </s>", "", decoded_words)

                    rouge_1, rouge_2, rouge_l = rouge_batch(
                        [decoded_words], [abstract])
                    count += 1

                    f.write("----- " + str(count) + " ------\n\n")
                    f.write("ARTICLE:   " + article + "\n\n")
                    f.write("TARGET:    " + abstract + "\n\n")
                    f.write("PREDICTED: " + decoded_words + "\n\n")
                    f.write("R1: " + str(rouge_1) + " - R2: " + str(rouge_2) + " - RL: " + str(rouge_l) + "\n\n\n")

                    overall[0].append(rouge_1)
                    overall[1].append(rouge_2)
                    overall[2].append(rouge_l)
                    if self.opt.log != "no":
                        # score = maxx
                        log_info.append(
                            [article.count(" "), abstract.count(" "), decoded_words.count(" "), rouge_1, rouge_2, rouge_l])
                        f_log.write(', '.join([str(log_info[-1][0]), str(log_info[-1][1]), str(
                            log_info[-1][2]), str(log_info[-1][3]), str(log_info[-1][4]), str(log_info[-1][5])])+'\n')
                        f_log_csv.write('=SPLIT("' + ', '.join([str(log_info[-1][0]), str(log_info[-1][1]), str(
                            log_info[-1][2]), str(log_info[-1][3]), str(log_info[-1][4]), str(log_info[-1][5]), article[:25]]) + '", ","' + ")\n")
                f.close()

                pbar.update(1)

                if pbar.n % 50 == 0:
                    rouge_1 = round(sum(overall[0])/len(overall[0]), 2)
                    rouge_2 = round(sum(overall[1])/len(overall[1]), 2)
                    rouge_l = round(sum(overall[2])/len(overall[2]), 2)
                    pbar.write("Total: " + str(len(overall[0])) + "         R1: " + str(rouge_1) + " - R2: " +
                               str(rouge_2) + " - RL: " + str(rouge_l))
                batch = self.batcher.next_batch()
        except KeyboardInterrupt:
            print("-------------------Keyboard Interrupt------------------")
            return

        if self.opt.log != "no":
            f_log.close()

        print("Total:", len(overall[0]))
        rouge_1 = round(sum(overall[0])/len(overall[0]), 2)
        rouge_2 = round(sum(overall[1])/len(overall[1]), 2)
        rouge_l = round(sum(overall[2])/len(overall[2]), 2)

        load_file = self.opt.load_model
        f = open(config.save_model_path + "/" + self.opt.task + "_rouge" + ".txt", "a+")
        f.write(self.opt.load_model + '\n')
        f.write("R1: " + str(rouge_1)[:5] + '\n')
        f.write("R2: " + str(rouge_2)[:5] + '\n')
        f.write("RL: " + str(rouge_l)[:5] + '\n\n')
        f.close()

        print("\n" + load_file, "scores:")
        print("R1:", str(rouge_1)[:5])
        print("R2:", str(rouge_2)[:5])
        print("RL:", str(rouge_l)[:5] + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="validate",
                        choices=["validate", "test", "train_x"])
    parser.add_argument("--start_from", type=str, default="0020000.tar")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument('--model_name', type=str, default="cur_model")
    parser.add_argument("--load_mixture", type=str, default=None)
    parser.add_argument('--log', type=str, default="yes")
    parser.add_argument('--threshold', type=float, default=0.15)
    parser.add_argument('--intra_encoder', type=bool, default=True)
    parser.add_argument('--intra_decoder', type=bool, default=True)
    opt = parser.parse_args()
    config.save_model_path = config.save_model_path + "/" + opt.model_name
    print(config.save_model_path)

    # from shutil import copyfile
    # copyfile(config.save_model_path + "/json_config.txt",
    #          "/content/ATS/data_util/json_config.py")

    # from data_util import json_config

    # config.hidden_dim = json_config.hidden_dim
    # config.max_first = str(json_config.max_first)
    # config.hidden_layers = json_config.hidden_layers
    # config.emb_dim = json_config.emb_dim
    # config.emb_type = str(json_config.emb_type)
    # config.max_enc_steps = json_config.max_enc_steps
    # config.max_dec_steps = 65
    # config.beam_size = json_config.beam_size
    # config.min_dec_steps = 10
    # config.lamda = json_config.lamda
    # config.vocab_size = json_config.vocab_size
    # config.batch_size = json_config.batch_size
    # config.use_focus = json_config.use_focus
    # config.n_mixture = json_config.n_mixture
    # config.focus_embed_size = json_config.focus_embed_size
    # config.intra_encoder = json_config.intra_encoder
    # config.intra_decoder = json_config.intra_decoder
    # config.use_filter = json_config.use_filter
    # config.spot_light = json_config.spot_light

    if opt.task == "validate":
        if (opt.load_model != None):
            print("Evaluating ", opt.load_model, " --------------")
            eval_processor = Evaluate(config.valid_data_path, opt)
            eval_processor.evaluate_batch()
        else:
            saved_models = os.listdir(config.save_model_path+"/models")
            saved_models.sort()
            file_idx = saved_models.index(opt.start_from)
            saved_models = saved_models[file_idx:]
            for f in saved_models:
                print("Evaluating ", f, " --------------")
                opt.load_model = f
                eval_processor = Evaluate(config.valid_data_path, opt)
                if ("livedoor" in opt.model_name):
                    eval_processor.evaluate_batch(print_sents=True)
                else:
                    eval_processor.evaluate_batch()
    elif opt.task == "test":
        print("Test ", opt.model_name, " --------------")
        eval_processor = Evaluate(config.test_data_path, opt)
        eval_processor.evaluate_batch(print_sents=True)
