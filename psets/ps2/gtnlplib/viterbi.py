import operator
from collections import defaultdict, Counter
from gtnlplib.constants import START_TAG,END_TAG, TRANS, EMIT
import numpy as np

def argmax(scores):
    """Find the key that has the highest value in the scores dict"""
    return max(scores.iteritems(),key=operator.itemgetter(1))[0]

def viterbi_step(tag, m, words, feat_func, weights, prev_scores):
    """
    Calculate the best path score and back pointer for a given node in the trellis

    :param tag: The tag for which we want to calculate the best path
    :param m: index of the token for which we want to calculate the best tag
    :param words: the list of tokens to tag
    :param feat_func: A function of (words, curr_tag, prev_tag, curr_index) that produces features 
    :param weights: A defaultdict that maps features to numeric score. Should not key error for indexing into keys that do not exist.
    :param prev_scores: a dict, in which keys are tags for token m-1 and values are viterbi scores
    :returns: tuple of (best_score, best_feature), where
        best_score   -- The highest score of any sequence of tags
        best_feature -- The feature in the previous layer of the trellis corresponding
            to the best score

    :rtype: tuple

    """
    scores = {}
    for prev_tag, prev_score in prev_scores.iteritems():
        feats = feat_func(words, tag, prev_tag, m)
        if type(prev_score) == tuple:
            prev_score = prev_score[0]
        scores[prev_tag] = prev_score
        for f,v in feats.iteritems():
            weight = weights[f]
            scores[prev_tag] += weight


    best_score = max(scores.values())
    best_tag = argmax(scores)

    return best_score, best_tag

def build_trellis(tokens,feat_func,weights,all_tags):
    """Construct a trellis for the hidden Markov model. Output is a list of dicts.

    :param tokens: list of word tokens to be tagged
    :param feat_func: feature function (words, tag, prev_tag, index)
    :param weights: defaultdict of weights
    :param all_tags: list/set of all possible tags
    :returns: list of dicts, length = len(words)
    first dict should represent score from start to token 1, 
    then score from token 1 to token 2,
    etc until token M
    :rtype: list of dicts

    """

    trellis = [None]*(len(tokens))
    prev = {}
    # build the first column separately
    for tag in all_tags:
        prev[tag] = viterbi_step(tag, 0, tokens, feat_func, weights,{START_TAG:0})
    trellis[0] = prev

    # iterate over the remaining columns
    for m in range(1,len(tokens)):
        prev = trellis[m-1]
        temp = {}
        for tag in all_tags:
            temp[tag] = viterbi_step(tag, m, tokens, feat_func, weights, prev)
        trellis[m] = temp

    return trellis


def viterbi_tagger(tokens,feat_func,weights,all_tags):
    """Tag the given words using the viterbi algorithm
        Parameters:
        words     -- A list of tokens to tag
        feat_func -- A function of (words, curr_tag, prev_tag, curr_index)
        that produces features
        weights   -- A defaultdict that maps features to numeric score. Should
        not key error for indexing into keys that do not exist.
        all_tags  -- A set of all possible tags

        Returns:
        tags       -- The highest scoring sequence of tags (list of tags s.t. tags[i]
        is the tag of words[i])
        best_score -- The highest score of any sequence of tags
    """

    trellis = build_trellis(tokens,feat_func,weights,all_tags)
    # Step 1: find last tag and best score
    final_scores = trellis[len(trellis)-1]
    for tag,v in final_scores.iteritems():
        final_scores[tag] = (v[0] + weights[END_TAG, tag, TRANS],v[1])

    last_tag = argmax(final_scores)
    best_score = max([v[0] for v in final_scores.values()])

    # Step 2: walk backwards through trellis to find best tag sequence
    output = [last_tag] # keep
    for m,v_m in enumerate(reversed(trellis[1:])): #keep
        last_tag = (v_m[last_tag])[1]
        output = [last_tag] + output

    return output,best_score

