# Most of this file is copied form https://github.com/abisee/pointer-generator/blob/master/data.py

import glob
import random
import struct
import csv
import numpy as np
from data_util import config
from tensorflow.core.example import example_pb2

PAD_TOKEN = '[PAD]'
# This has a vocab id, which is used to represent out-of-vocabulary words
UNKNOWN_TOKEN = '[UNK]'


class Vocab(object):

    def __init__(self, vocab_file, max_size):
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # keeps track of total number of words in the Vocab

        config.SENTENCE_START = '<s>'
        config.SENTENCE_END = '</s>'
        config.START_DECODING = '[START]'
        config.STOP_DECODING = '[STOP]'
        # [UNK], [PAD], [START] and [STOP] get the ids 0,1,2,3.
        for w in [UNKNOWN_TOKEN, PAD_TOKEN, config.START_DECODING, config.STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    # print ('Warning: incorrectly formatted line in vocabulary file: %s\n' % line)
                    continue
                w = pieces[0]
                if w in [config.SENTENCE_START, config.SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, config.START_DECODING, config.STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception(
                        'Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    # print ("max_size of vocab was specified as %i; we now have %i words. Stopping reading." % (max_size, self._count))
                    break


    def word2id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        return self._count

    def write_metadata(self, fpath):
        print("Writing word embedding metadata file to %s..." % (fpath))
        with open(fpath, "w") as f:
            fieldnames = ['word']
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            for i in xrange(self.size()):
                writer.writerow({"word": self._id_to_word[i]})


def example_generator(data_path, single_pass):
    while True:
        filelist = glob.glob(data_path)  # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' %
                          data_path)  # check filelist isn't empty
        if single_pass:
            filelist = sorted(filelist)
        else:
            random.shuffle(filelist)
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break  # finished reading this file
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack(
                    '%ds' % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)
        if single_pass:
            # print ("example_generator completed reading all datafiles. No more data.")
            break


def getWe(i2w, word_emb_glove):
    int2word_sorted = sorted(i2w.items())

    # Get the list of word embedding corresponding to int value in ascending order
    word_emb_list = list()
    embedding_size = len(word_emb_glove['the'])
    # Assign random vector to <s>, </s> token

    for int_val, word in int2word_sorted:
        # Add Glove embedding if it exists
        if(word in word_emb_glove):
            word_emb_list.append(word_emb_glove[word])

        # Otherwise, the value of word embedding is 0
        else:
            word_emb_list.append(np.zeros((embedding_size)))

    word_emb_list[2] = np.random.normal(0, 1, embedding_size)
    word_emb_list[3] = np.random.normal(0, 1, embedding_size)
    word_emb_list[0] = np.random.normal(0, 1, embedding_size)
    word_emb_list[1] = np.random.normal(0, 1, embedding_size)

    # the final word embedding
    word_emb = np.array(word_emb_list)
    return word_emb


def getWeWord2Vec(i2w, ft):
    int2word_sorted = sorted(i2w.items())

    # Get the list of word embedding corresponding to int value in ascending order
    word_emb_list = list()
    embedding_size = ft.get_dimension()
    # Assign random vector to <s>, </s> token

    for int_val, word in int2word_sorted:
        word_emb_list.append(ft.get_word_vector(word))

    word_emb_list[2] = np.random.normal(0, 1, embedding_size)
    word_emb_list[3] = np.random.normal(0, 1, embedding_size)
    word_emb_list[0] = np.random.normal(0, 1, embedding_size)
    word_emb_list[1] = np.random.normal(0, 1, embedding_size)

    # the final word embedding
    word_emb = np.array(word_emb_list)
    return word_emb


def wordEmbedingList(path):
    word_emb_glove = dict()
    with open(path, encoding="utf8") as f:
        for line in f:
            el = line.split()
            word = el[0]
            emb = [float(val) for val in el[1:]]
            word_emb_glove[word] = emb
    return word_emb_glove


def article2ids(article_words, vocab):
    ids = []
    oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is OOV
            if w not in oovs:  # Add to list of OOVs
                oovs.append(w)
            # This is 0 for the first article OOV, 1 for the second article OOV...
            oov_num = oovs.index(w)
            # This is e.g. 50000 for the first article OOV, 50001 for the second...
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)
    return ids, oovs


def abstract2ids(abstract_words, vocab, article_oovs):
    ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:  # If w is an OOV word
            if w in article_oovs:  # If w is an in-article OOV
                # Map to its temporary article OOV number
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:  # If w is an out-of-article OOV
                ids.append(unk_id)  # Map to the UNK token id
        else:
            ids.append(i)
    return ids


def outputids2words(id_list, vocab, article_oovs):
    words = []
    for i in id_list:
        try:
            w = vocab.id2word(i)  # might be [UNK]
        except ValueError as e:  # w is OOV
            assert article_oovs is not None, "Error: model produced a word ID that isn't in the vocabulary. This should not happen in baseline (no pointer-generator) mode"
            article_oov_idx = i - vocab.size()
            try:
                w = article_oovs[article_oov_idx]
            except ValueError as e:  # i doesn't correspond to an article oov
                raise ValueError('Error: model produced word ID %i which corresponds to article OOV %i but this example only has %i article OOVs' % (
                    i, article_oov_idx, len(article_oovs)))
        words.append(w)
    return words


def abstract2sents(abstract):
    cur = 0
    sents = []
    while True:
        try:
            start_p = abstract.index(SENTENCE_START, cur)
            end_p = abstract.index(SENTENCE_END, start_p + 1)
            cur = end_p + len(SENTENCE_END)
            sents.append(abstract[start_p+len(SENTENCE_START):end_p])
        except ValueError as e:  # no more sentences
            return sents


def show_art_oovs(article, vocab):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = article.split(' ')
    words = [("__%s__" % w) if vocab.word2id(
        w) == unk_token else w for w in words]
    out_str = ' '.join(words)
    return out_str


def show_abs_oovs(abstract, vocab, article_oovs):
    unk_token = vocab.word2id(UNKNOWN_TOKEN)
    words = abstract.split(' ')
    new_words = []
    for w in words:
        if vocab.word2id(w) == unk_token:  # w is oov
            if article_oovs is None:  # baseline mode
                new_words.append("__%s__" % w)
            else:  # pointer-generator mode
                if w in article_oovs:
                    new_words.append("__%s__" % w)
                else:
                    new_words.append("!!__%s__!!" % w)
        else:  # w is in-vocab word
            new_words.append(w)
    out_str = ' '.join(new_words)
    return out_str
