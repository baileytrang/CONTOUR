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
from rouge_score import rouge_scorer

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
        scores = scorer.score(a[i].replace(". ", ".\n"),
                              b[i].replace(". ", ".\n"))
        score[0].append(scores['rouge1'].fmeasure)
        score[1].append(scores['rouge2'].fmeasure)
        score[2].append(scores['rougeLsum'].fmeasure)

    rouge_1 = sum(score[0])/len(score[0])
    rouge_2 = sum(score[1])/len(score[1])
    rouge_l = sum(score[2])/len(score[2])

    return round(rouge_1, 4) * 100, round(rouge_2, 4) * 100, round(rouge_l, 4) * 100


class Evaluate_epoch(object):
    def __init__(self, data_path, batch_size=config.batch_size):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(data_path, self.vocab, mode='eval',
                               batch_size=batch_size, single_pass=True)
        self.max_rl = 0
        time.sleep(5)

    def evaluate_batch(self, model):
        self.model = model
        self.model = get_cuda(self.model)
        batches = self.batcher
        batch = batches.next_batch()
        start_id = self.vocab.word2id(config.START_DECODING)
        end_id = self.vocab.word2id(config.STOP_DECODING)
        self.unk_id = self.vocab.word2id(data.UNKNOWN_TOKEN)
        decoded_sents = []
        ref_sents = []
        article_sents = []
        rouge = Rouge()
        count = 0

        filename = "/" + "validate" + ".txt"

        if not os.path.exists(config.save_model_path + "/avg_rounge.csv"):
            f = open(config.save_model_path + "/avg_rounge.csv", 'w')

        if not os.path.exists(config.save_model_path + filename):
            f = open(config.save_model_path + filename, 'w')

        overall = [[], [], []]

        while batch is not None:
            enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e, focus_mask, focus_input = get_enc_data(
                batch)
            source_WORD_encoding = enc_batch
            source_WORD_encoding_extended = enc_batch_extend_vocab
            with T.autograd.no_grad():
                focus_p = self.model.selector(
                    source_WORD_encoding,
                    train=False)

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

            score = 0
            n_val = 100
            # for i in range(min(n_val, len(pred_ids) // config.n_mixture)):
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
                # maxx = [0, 0, 0]
                # max_w = ""
                # for m in range(config.n_mixture):
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

                overall[0].append(rouge_1)
                overall[1].append(rouge_2)
                overall[2].append(rouge_l)

                if count % 600 == 0:
                    f.write("----- " + str(count) + " ------\n\n")
                    f.write("ARTICLE:   " + article + "\n\n")
                    f.write("TARGET:    " + abstract + "\n\n")
                    f.write("PREDICTED: " + decoded_words + "\n\n")
                    f.write("R1: " + str(rouge_1) + " - R2: " +
                            str(rouge_2) + " - RL: " + str(rouge_l) + "\n\n\n")

            batch = batches.next_batch()

        rouge_1 = round(sum(overall[0])/len(overall[0]), 2)
        rouge_2 = round(sum(overall[1])/len(overall[1]), 2)
        rouge_l = round(sum(overall[2]) / len(overall[2]), 2)
        return rouge_1, rouge_2, rouge_l
