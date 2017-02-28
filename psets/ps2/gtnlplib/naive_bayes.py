import numpy as np #hint: np.log
import sys
from collections import defaultdict,Counter
from gtnlplib import scorer, most_common,preproc
from gtnlplib.constants import OFFSET

def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for i in range(len(x)):
        if y[i] == label:
            for word, count in x[i].iteritems():
                corpus_counts[word] += count
    return corpus_counts

def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    counts = get_corpus_counts(x,y,label)
    sum_counts = sum(counts.values())
    divisor = sum_counts + len(vocab)*smoothing
    for word in vocab:
        counts[word] += smoothing
        counts[word] /= divisor
        counts[word] = np.log(counts[word])
    return counts

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict

    """
    #hint: use your solution from pset 1
    label_count = defaultdict(float)
    for label in y:
        label_count[label] += 1
    for label in label_count.iterkeys():
        label_count[label] /= len(y)
    labels = set(y)
    weights = defaultdict(float)
    vocab = set()
    for instance in x:
        vocab |= set(instance.keys())
    for label in labels:
        pxy = estimate_pxy(x, y, label, smoothing, list(vocab))
        for word in vocab:
            weights[(label, word)] = pxy[word]
        weights[(label, OFFSET)] = np.log(label_count[label])
    return weights

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    # hint: call estimate_nb, then modify the o`````````````````utput
    print counters