"""
This module implementation is based on FocusSeq2Seq - www.aclweb.org/anthology/D19-1308/
"""

import torch
import torch.nn as nn
from data_util import config, data

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class FocusedEncoder(nn.Module):
    def __init__(self, word_embed_size=300,
                 focus_embed_size=16,
                 hidden_size=512, num_layers=1,
                 dropout_p=0.5, use_focus=True, model='NQG', rnn_type='GRU', feature_rich=False):
        super().__init__()

        self.model = model
        self.feature_rich = feature_rich

        rnn_input_size = word_embed_size
        rnn_input_size += config.focus_embed_size

        if rnn_type == 'GRU':
            self.rnn_type = 'GRU'
            self.rnn = nn.GRU(input_size=rnn_input_size,
                              hidden_size=hidden_size // 2,
                              bidirectional=True,
                              batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn_type = 'LSTM'
            self.rnn = nn.LSTM(input_size=rnn_input_size,
                               hidden_size=hidden_size // 2,
                               bidirectional=True,
                               batch_first=True)

        if dropout_p > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout_p)
        else:
            self.use_dropout = False

        self.use_focus = use_focus

    def forward(self,
                source_WORD_encoding,
                focus_mask=None,
                PAD_ID=1):

        pad_mask = (source_WORD_encoding == PAD_ID)

        # ---- Sort by length (decreasing order) ----#
        source_len = (~pad_mask).long().sum(dim=1)  # [B]
        source_len_sorted, idx_sorted = torch.sort(
            source_len, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sorted, dim=0)

        # source_len_sorted = source_len_sorted.tolist()

        source_WORD_encoding = source_WORD_encoding[idx_sorted]
        word_embedding = self.word_embed(
            source_WORD_encoding)  # [B, L, word_embed_size]
        enc_input = word_embedding

        if self.use_focus == "yes":
            focus_mask = focus_mask[idx_sorted]

            # # Focus Dropout
            # if self.training:
            #     drop_mask = torch.full_like(focus_mask, 0.9, dtype=torch.float).bernoulli().long()
            #     focus_mask = drop_mask * focus_mask

            focus_embedding = self.focus_embed(focus_mask)
            enc_input = torch.cat([enc_input, focus_embedding], dim=2)

        # ---- Input Dropout ----#
        if self.use_dropout:
            enc_input = self.dropout(enc_input)

        # ---- RNN ----#
        enc_input = pack_padded_sequence(
            enc_input, source_len_sorted, batch_first=True)

        enc_outputs, h = self.rnn(enc_input)

        enc_outputs, _ = pad_packed_sequence(enc_outputs, batch_first=True)

        # ---- Unsort ----#
        enc_outputs = enc_outputs[idx_unsort]
        if self.rnn_type == 'LSTM':
            h, c = h
            h = h.index_select(1, idx_unsort)
            c = c.index_select(1, idx_unsort)
            h = (h, c)
        elif self.rnn_type == 'GRU':
            h = h.index_select(1, idx_unsort)

        return enc_outputs, h
