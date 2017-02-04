from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict, Counter

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

    labels = set(y)
    counts = defaultdict(float) #??
    doc_counts = defaultdict(float) #hint??
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

    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    best_smoother = 0
    bestacc = 0
    scores = defaultdict(float)
    for smoother in smoothers:
        theta_nb = estimate_nb(x_tr,y_tr,smoother)
        labels = set(y_dv)
        y_hat = clf_base.predict_all(x_dv,theta_nb, labels)
        acc = evaluation.acc(y_hat,y_dv)
        scores[smoother] = acc
        if acc > bestacc:
            bestacc = acc
            best_smoother = smoother

    return best_smoother, scores