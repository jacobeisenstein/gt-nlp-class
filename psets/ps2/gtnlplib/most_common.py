import operator
from collections import defaultdict, Counter
from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import OFFSET, START_TAG, END_TAG

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

def get_tag_word_counts(filename):
    """build a dict of counters, one per tag, counting the words that go with each tag

    :param trainfile: training data
    :returns: dict of counters
    :rtype: dict

    """
    all_counters = defaultdict(lambda : Counter())

    # your code here
    # hint: for words, tags in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):

    raise NotImplementedError

    return all_counters

def get_noun_weights():
    """Produce weights dict mapping all words as noun

    :returns: simple weight dictionary

    """
    weights = defaultdict(float)
    weights[('NOUN'),OFFSET] = 1.
    return weights

def get_most_common_word_weights(trainfile):
    """Return a set of weights, so that each word is tagged by its most frequent tag in the training file.
    If the word does not appear in the training file, the weights should be set so that the output tag is Noun.

    :param trainfile: training file
    :returns: classification weights
    :rtype: defaultdict

    """
    weights = defaultdict(float)
    raise NotImplementedError
    return weights

def get_tag_trans_counts(trainfile):
    """compute a dict of counters for tag transitions

    :param trainfile: name of file containing training data
    :returns: dict, in which keys are tags, and values are counters of succeeding tags
    :rtype: dict

    """
    tot_counts = defaultdict(lambda : Counter())
    for _,tags in conll_seq_generator(trainfile):
        tags = [START_TAG] + tags + [END_TAG]
        tag_trans = zip(tags[:-1],tags[1:])
        for prev_tag, curr_tag in tag_trans:
            tot_counts[prev_tag][curr_tag] += 1

    return dict(tot_counts)
