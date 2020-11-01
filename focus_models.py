"""
This module implementation is based on FocusSeq2Seq - www.aclweb.org/anthology/D19-1308/
"""

import torch
import torch.nn as nn

from layers.encoder import FocusedEncoder
from layers.selector import ParallelSelector
from data_util import config, data

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Model(nn.Module):
    def __init__(self, seq2seq, selector=None):
        super().__init__()
        self.selector = selector
        self.seq2seq = seq2seq


class FocusSelector(nn.Module):
    """Sample focus (sequential binary masks) from source sequence"""

    def __init__(self,
                 word_embed_size: int = 300,
                 focus_embed_size: int = 16,
                 enc_hidden_size: int = 300,
                 dec_hidden_size: int = 300,
                 num_layers: int = 1,
                 dropout_p: float = 0.2,
                 rnn: str = 'GRU',
                 n_mixture: int = 1,
                 seq2seq_model: str = 'NQG',
                 threshold: float = 0.15,
                 feature_rich: bool = False):

        super().__init__()

        self.seq2seq_model = seq2seq_model
        self.feature_rich = feature_rich

        self.selector = ParallelSelector(
            word_embed_size,
            enc_hidden_size, dec_hidden_size,
            num_layers=num_layers, dropout_p=dropout_p, n_mixture=n_mixture,
            threshold=threshold)

    def add_embedding(self, word_embed):
        self.selector.encoder.word_embed = word_embed

    def forward(self,
                source_WORD_encoding,
                focus_POS_prob=None,
                mixture_id=None,
                focus_input=None,
                train=True,
                max_decoding_len=30):

        out = self.selector(
            source_WORD_encoding,
            mixture_id,
            focus_input,
            train,
            max_decoding_len)

        if train:
            focus_logit = out
            return focus_logit

        else:
            generated_focus_mask = out
            return generated_focus_mask



def repeat(tensor, K):
    """
    [B, ...] => [B*K, ...]
    #-- Important --#
    Used unsqueeze and transpose to avoid [K*B] when using torch.Tensor.repeat
    """
    if isinstance(tensor, torch.Tensor):
        B, *size = tensor.size()
        expand_size = B, K, *size
        tensor = tensor.unsqueeze(1).expand(
            *expand_size).contiguous().view(B * K, *size)
        return tensor
    elif isinstance(tensor, list):
        out = []
        for x in tensor:
            for _ in range(K):
                out.append(x.copy())
        return out
