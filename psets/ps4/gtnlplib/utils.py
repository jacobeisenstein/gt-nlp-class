import torch
import torch.autograd as ag
from gtnlplib.constants import END_OF_INPUT_TOK, HAVE_CUDA

if HAVE_CUDA:
    import torch.cuda as cuda


def word_to_variable_embed(word, word_embeddings, word_to_ix):
    return word_embeddings(ag.Variable(torch.LongTensor([ word_to_ix[word] ])))


def sequence_to_variable(sequence, to_ix, use_cuda=False):
    if use_cuda:
        return ag.Variable( cuda.LongTensor([ to_ix[t] for t in sequence ]) )
    else:
        return ag.Variable( torch.LongTensor([ to_ix[t] for t in sequence ]) )


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


def concat_and_flatten(items):
    """
    Concatenate feature vectors together in a way that they can be handed into
    a linear layer
    :param items A list of ag.Variables which are vectors
    :return One long row vector of all of the items concatenated together
    """
    return torch.cat(items, 1).view(1, -1)


def initialize_with_pretrained(pretrained_embeds, word_embedding_component):
    """
    Initialize the embedding lookup table of word_embedding_component with the embeddings
    from pretrained_embeds.
    Remember that word_embedding_component has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    :param pretrained_embeds dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding_component The network component to initialize (i.e, a VanillaWordEmbeddingLookup
        or BiLSTMWordEmbeddingLookup)
    """
    # STUDENT
    for word, index in word_embedding_component.word_to_ix.items():
        if word in pretrained_embeds:
            word_embedding_component.word_embeddings.weight.data[index] = torch.Tensor(pretrained_embeds[word])
    # END STUDENT


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


class DummyWordEmbeddingLookup:

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
