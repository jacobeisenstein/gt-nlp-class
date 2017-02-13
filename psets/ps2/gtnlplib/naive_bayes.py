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
    dic = defaultdict(float)
    for i in range(len(x)):
        if y[i] == label:
            for word,count in x[i].items():
                dic[word] += count
    return dic

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
    est = defaultdict(float)
    cc = get_corpus_counts(x, y, label)
    totalcount = sum(cc.values())
    for word in vocab:
        pLabels = totalcount + len(vocab)*smoothing
        pwordNlabels = cc[word] + smoothing
        est[word] = np.log(pwordNlabels/pLabels)
    return est

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict

    """
    #hint: use your solution from pset 1
    labels = set(y)
    weights = defaultdict(float)
    vocab = set()
    # get the set of vocabularies
    for i in x:
        vocab |= set(i.keys())

    py = Counter(y) # get the counter of the labels
    totalY = sum(py.values())

    #get the prior prob. of labels
    for lab, count in py.items():
        py[lab] = (float)(count)/totalY

    #prepare the weights
    for l in labels:
        weights[(l,OFFSET)] = np.log(py[l])
        pxy_i = estimate_pxy(x,y,l,smoothing,vocab)
        for word in vocab:
            weights[(l,word)] += pxy_i[word]

    return weights

def estimate_nb_tagger(counters,smoothing):
    """build a tagger based on the naive bayes classifier, which correctly accounts for the prior P(Y)

    :param counters: dict of word-tag counters, from most_common.get_tag_word_counts
    :param smoothing: value for lidstone smoothing
    :returns: classifier weights
    :rtype: defaultdict

    """
    # hint: call estimate_nb, then modify the output

    sorted_tags = sorted(counters.keys())
    weights = estimate_nb([counters[tag] for tag in sorted_tags], sorted_tags, smoothing)
    tempDict = defaultdict(float)

    for tag in sorted_tags:
        tempDict[tag] = sum(counters[tag].values())
    totalTags = sum(tempDict.values())

    for tag in sorted_tags:
        weights[tag,OFFSET] = np.log(tempDict[tag]/(totalTags + 0.0))

    return weights

    
