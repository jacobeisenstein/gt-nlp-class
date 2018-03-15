import torch
import torch.autograd as ag
from gtnlplib.constants import END_OF_INPUT_TOK, HAVE_CUDA

import numpy as np

if HAVE_CUDA:
    import torch.cuda as cuda

def to_scalar(var):
    """
    Wrap up the terse, obnoxious code to go from torch.Tensor to
    a python int / float (is there a better way?)
    """
    if isinstance(var, ag.Variable):
        return var.data.view(-1).tolist()[0]
    else:
        return var.view(-1).tolist()[0]


def argmax(vector):
    """
    Takes in a row vector (1xn) and returns its argmax
    """
    _, idx = torch.max(vector, 1)
    return to_scalar(idx)


def initialize_with_pretrained(pretrained_embeds, word_embedding):
    """
    Initialize the embedding lookup table of word_embedding with the embeddings
    from pretrained_embeds.
    Remember that word_embedding has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    :param pretrained_embeds dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding The network component to initialize (i.e, a VanillaWordEmbedding
        or BiLSTMWordEmbedding)
    """
    # STUDENT
    raise NotImplementedError
    # END STUDENT

def build_suff_to_ix(word_to_ix):
    """
        From a word to index vocab lookup, create a suffix-to-index vocab lookup
        For our purposes, a suffix consists of just the last two letters of a word.
            If the word is a single letter, just use the whole word
        :param word_to_ix: the vocab as a dict
        :return suff_to_ix: the suffix lookup as a dict
    """
    suffset = set()
    # STUDENT
    raise NotImplementedError
    # END STUDENT
    suff_to_ix = {c: i for i, c in enumerate(sorted(suffset))}
    return suff_to_ix


# ===----------------------------------------------------------------===
# Dummy classes that let us test parsing logic without having the
# necessary components implemented yet
# ===----------------------------------------------------------------===
class DummyCombiner:

    def __call__(self, head, modifier):
        return head


class DummyActionChooser:

    def __init__(self):
        self.counter = 0

    def __call__(self, inputs):
        self.counter += 1
        return ag.Variable(torch.Tensor([0., 0., 1.]))


class DummyWordEmbedding:

    def __init__(self):
        self.word_embeddings = lambda x: None
        self.counter = 0

    def __call__(self, sentence):
        self.counter += 1
        return [None]*len(sentence)


class DummyFeatureExtractor:

    def __init__(self):
        self.counter = 0

    def get_features(self, parser_state):
        self.counter += 1
        return []


