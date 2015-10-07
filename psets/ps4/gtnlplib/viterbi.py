import numpy as np #hint: np.log
from itertools import chain
import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conllSeqGenerator

from gtnlplib import scorer
from gtnlplib import constants
from gtnlplib import preproc
from gtnlplib.constants import START_TAG ,TRANS ,END_TAG , EMIT

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# define viterbiTagger
start_tag = constants.START_TAG
trans = constants.TRANS
end_tag = constants.END_TAG
emit = constants.EMIT


def viterbiTagger(words,feat_func,weights,all_tags,debug=False):
    """
    :param words: list of words
    :param feat_func: feature function
    :param weights: defaultdict of weights
    :param tagset: list of permissible tags
    :param debug: optional debug flag
    :returns output: tag sequence
    :returns best_score: viterbi score of best tag sequence
    """
    return output,best_score


