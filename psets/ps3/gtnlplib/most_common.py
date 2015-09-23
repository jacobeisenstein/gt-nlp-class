''' your code '''
import operator
from  constants import *
from collections import defaultdict, Counter
from gtnlplib import preproc
import scorer
from gtnlplib import constants
from gtnlplib import clf_base

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tags(trainfile):
    """Produce a Counter of occurences of word in each tag"""
    return counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun"""
    return your_weights

def get_most_common_weights(trainfile):

    return weights

def get_class_counts(counters):
    return counts
