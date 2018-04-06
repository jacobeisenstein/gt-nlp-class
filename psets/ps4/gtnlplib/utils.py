import torch
import torch.autograd as ag

UNK_TOKEN = '<UNK>'

def to_scalar(var):
    '''
    Wrap up the terse, obnoxious code to go from torch.Tensor to
    a python int / float
    '''
    if isinstance(var, ag.Variable):
        return var.data.view(-1).tolist()[0]
    else:
        return var.view(-1).tolist()[0]

def argmax(vector):
    '''
    Takes in a row vector (1xn) and returns its argmax
    '''
    _, idx = torch.max(vector, 1)
    return to_scalar(idx)

## deliverable 4.5
def initialize_with_pretrained(pretrained_embeds, word_embedding, use_cuda=False):
    '''
    Initialize the embedding lookup table of word_embedding with the embeddings
    from pretrained_embeds.
    Remember that word_embedding has a word_to_ix member you will have to use.
    For every word that we do not have a pretrained embedding for, keep the default initialization.
    NOTE: don't forget the UNK token!
    :param pretrained_embeds: dict mapping word to python list of floats (the embedding
        of that word)
    :param word_embedding: The network component to initialize (i.e, a BiLSTMWordEmbedding)
    '''
    raise NotImplementedError

