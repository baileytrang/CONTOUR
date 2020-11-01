import numpy as np
import torch as T
from data_util import config
from torch.autograd import Variable


def get_cuda(tensor):
    if T.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def padding_tensor(sequences):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    max_len = max([s.size(0) for s in sequences])
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor


def get_enc_data(batch):
    batch_size = len(batch.enc_lens)
    enc_batch = T.from_numpy(batch.enc_batch).long()
    enc_padding_mask = T.from_numpy(batch.enc_padding_mask).float()

    enc_lens = batch.enc_lens

    ct_e = T.zeros(batch_size, 2*config.hidden_dim)

    enc_batch = get_cuda(enc_batch)
    enc_padding_mask = get_cuda(enc_padding_mask)

    ct_e = get_cuda(ct_e)

    enc_batch_extend_vocab = None
    if batch.enc_batch_extend_vocab is not None:
        enc_batch_extend_vocab = T.from_numpy(
            batch.enc_batch_extend_vocab).long()
        enc_batch_extend_vocab = get_cuda(enc_batch_extend_vocab)

    extra_zeros = None
    if batch.max_art_oovs > 0:
        extra_zeros = T.zeros(batch_size, batch.max_art_oovs)
        extra_zeros = get_cuda(extra_zeros)
    focus_input = padding_tensor(batch.focus_input)
    focus_input = T.Tensor(focus_input).long().to(device='cuda')
    return enc_batch, enc_lens, enc_padding_mask, enc_batch_extend_vocab, extra_zeros, ct_e, T.Tensor(batch.focus_mask).long().to(device='cuda'), focus_input


def get_dec_data(batch):
    dec_batch = T.from_numpy(batch.dec_batch).long()
    dec_lens = batch.dec_lens
    max_dec_len = np.max(dec_lens)
    dec_lens = T.from_numpy(batch.dec_lens).float()

    target_batch = T.from_numpy(batch.target_batch).long()

    dec_batch = get_cuda(dec_batch)
    dec_lens = get_cuda(dec_lens)
    target_batch = get_cuda(target_batch)

    return dec_batch, max_dec_len, dec_lens, target_batch
