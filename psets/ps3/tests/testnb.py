from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE, OFFSET
from gtnlplib import constants, clf_base, preproc, naivebayes,most_common
from collections import defaultdict, Counter
import numpy as np

weights_nb = {}
alltags = set([])
allwords = set()
def setup_module ():
    global weights_nb
    global alltags
    global allwords
    counters = most_common.get_tags(TRAIN_FILE)
    for counts in counters.values():
        allwords.update(set(counts.keys()))
    class_counts =  most_common.get_class_counts(counters)
    weights_nb = naivebayes.learnNBWeights(counters,class_counts,allwords)
    alltags = preproc.getAllTags(TRAIN_FILE)

def test_nb_one_class():
    allwords = ['football', 'spoon', 'dog']
    wordCountsByTag = Counter({
        'N': Counter({
            'football': 1,
            'spoon': 1,
            'dog': 1
        })
    })
    classCounts = Counter({'N': 3})
    weights = naivebayes.learnNBWeights(wordCountsByTag, classCounts, allwords, alpha=0)
    assert_almost_equals(0.333, np.exp(weights[('N', 'spoon')]), places=3)
    assert_almost_equals(0.333, np.exp(weights[('N', 'football')]), places=3)
    assert_almost_equals(0.333, np.exp(weights[('N', 'dog')]), places=3)


def test_nb_simple():
    '''
    Tests for the following two sentences: 
    the D
    man N
    runs V

    man V
    the D
    cannons N
    '''
    allwords = ['the', 'man', 'runs', 'the', 'cannons']
    wordCountsByTag = Counter({
        'D': Counter({'the': 2}),
        'N': Counter({'man': 1, 'cannons': 1}),
        'V': Counter({'runs': 1, 'man': 1})
    })
    classCounts = Counter({'D': 2, 'N': 2, 'V': 2})

    weights = naivebayes.learnNBWeights(wordCountsByTag, classCounts, allwords, alpha=0)
    assert_almost_equals(0.5, np.exp(weights[('N', 'man')]), places=3)
    assert_almost_equals(0.5, np.exp(weights[('V', 'man')]), places=3)
    assert_almost_equals(1.0, np.exp(weights[('D', 'the')]), places=3)

    # offsets
    assert_almost_equals(0.333, np.exp(weights[('N', OFFSET)]), places=3)
    assert_almost_equals(0.333, np.exp(weights[('V', OFFSET)]), places=3)
    assert_almost_equals(0.333, np.exp(weights[('D', OFFSET)]), places=3)

def test_nb_smoothing():
    '''
    Tests for the following two sentences, with smoothing of 0.5
    the D
    man N
    runs V

    man V
    the D
    cannons N
    '''
    allwords = ['the', 'man', 'runs', 'the', 'cannons']
    wordCountsByTag = Counter({
        'D': Counter({'the': 2}),
        'N': Counter({'man': 1, 'cannons': 1}),
        'V': Counter({'runs': 1, 'man': 1})
    })
    classCounts = Counter({'D': 2, 'N': 2, 'V': 2})

    # smoothing of 0.5 reserves 1/2 probability mass for unknown
    weights = naivebayes.learnNBWeights(wordCountsByTag, classCounts,
                                        allwords, alpha=0.5)
    assert_almost_equals(5.0 / 8.0, np.exp(weights[('D', 'the')]), places=3)
    assert_almost_equals(1.0 / 8.0, np.exp(weights[('N', 'the')]))

    assert_almost_equals(0.333, np.exp(weights[('N', OFFSET)]), places=3)
    assert_almost_equals(0.333, np.exp(weights[('V', OFFSET)]), places=3)
    # offsets unchanged
    assert_almost_equals(0.333, np.exp(weights[('D', OFFSET)]), places=3)

def test_nb_prob_mass ():
    probability_masses = defaultdict(float)
    allwords = set([])
    for words, _ in preproc.conllSeqGenerator(TRAIN_FILE):
        for word in words:
            allwords.add(word)

    for tag in alltags:
        total_prob = sum(np.exp(weights_nb[(tag, word)]) for word in allwords)
        assert_almost_equals (1.0, total_prob, places=2,
            msg="UNEQUAL Expected tag %s to have total prob of 1.0, but instead has %s" %(tag, total_prob))

def test_nb_weights_noun ():
    expected = -7.74708642426
    actual = weights_nb[('N','breakfast')]
    assert_almost_equals (expected, actual, places=3,msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_nb_weights_verb ():
    expected = -10.226404038
    actual = weights_nb[('V','breakfast')]
    assert_almost_equals (expected, actual, places=3,msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_nb_weights_offset ():
    expected = -3.58372417191
    actual = weights_nb[('!',constants.OFFSET)]
    assert_almost_equals (expected, actual, places=3,msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))
