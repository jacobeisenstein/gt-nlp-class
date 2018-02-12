from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, END_TAG, OFFSET, UNK
from gtnlplib import naive_bayes, most_common 
import numpy as np
from collections import defaultdict
import torch
import torch.nn
from torch.autograd import Variable


def compute_transition_weights(trans_counts, smoothing):
    """
    Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag)] and weights

    """
    
    weights = defaultdict(float)

    raise NotImplementedError
    
    all_tags = list(trans_counts.keys())+ [END_TAG]
    
    
    
    return weights

def compute_weights_variables(nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix):
    """
    Computes autograd Variables of two weights: emission_probabilities and the tag_transition_probabilties
    parameters:
    nb_weights: -- a dictionary of emission weights
    hmm_trans_weights: -- dictionary of tag transition weights
    vocab: -- list of all the words
    word_to_ix: -- a dictionary that maps each word in the vocab to a unique index
    tag_to_ix: -- a dictionary that maps each tag (including the START_TAG and the END_TAG) to a unique index.
    
    :returns:
    emission_probs_vr: torch Variable of a matrix of size Vocab x Tagset_size
    tag_transition_probs_vr: torch Variable of a matrix of size Tagset_size x Tagset_size
    :rtype: autograd Variables of the the weights
    """
    # Assume that tag_to_ix includes both START_TAG and END_TAG
    
    
    
    tag_transition_probs = np.full((len(tag_to_ix), len(tag_to_ix)), -np.inf)
    emission_probs = np.full((len(vocab),len(tag_to_ix)), 0.0)
    
    raise NotImplementedError
    
    
    
    emission_probs_vr = Variable(torch.from_numpy(emission_probs.astype(np.float32)))
    tag_transition_probs_vr = Variable(torch.from_numpy(tag_transition_probs.astype(np.float32)))
    
    return emission_probs_vr, tag_transition_probs_vr
    
