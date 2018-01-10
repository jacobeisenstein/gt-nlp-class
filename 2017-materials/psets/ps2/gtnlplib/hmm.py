from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param i: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    raise NotImplementedError
    

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    # hint: these are your first two lines
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    # hint: call compute_transition_weights
    # hint: set weights for illegal transitions to -np.inf
    # hint: call get_tag_word_counts and estimate_nb_tagger
    # hint: Counter.update() combines two Counters

    # hint: return weights, all_tags
    raise NotImplementedError


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """

    weights = defaultdict(float)
    raise NotImplementedError
    

