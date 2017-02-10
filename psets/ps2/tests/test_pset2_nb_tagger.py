from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest, assert_not_equal
from gtnlplib import most_common, naive_bayes, tagger_base, constants, scorer
import numpy as np

def setup():
    global counters, theta_nb, vocab, theta_nb_fixed, sorted_tags
    
    counters = most_common.get_tag_word_counts(constants.TRAIN_FILE)

    sorted_tags = sorted(counters.keys())

    theta_nb = naive_bayes.estimate_nb([counters[tag] for tag in sorted_tags],
                                         sorted_tags,
                                         .01)

    vocab = set([word for tag,word in theta_nb.keys() if word is not constants.OFFSET])

    theta_nb_fixed = naive_bayes.estimate_nb_tagger(counters,.01)
    
# 2.3a (0.5 points)
def test_nb_sum_to_one():
    global theta_nb, vocab
    assert_almost_equals(1, sum(np.exp(theta_nb[('ADJ',word)]) for word in vocab), places=6)
    assert_almost_equals(1, sum(np.exp(theta_nb[('PRON',word)]) for word in vocab), places=6)
    assert_almost_equals(1, sum(np.exp(theta_nb[('PUNCT',word)]) for word in vocab), places=6)

# 2.3b
def test_nb_case_sensitive():
    global theta_nb
    assert_not_equal(theta_nb[('ADJ','Bad')],
                     theta_nb[('ADJ','bad')])

# 2.3c
def test_nb_no_oov():
    global theta_nb
    eq_(theta_nb[('ADJ','baaaaaaaaaaaaaaad')],0)
    
# 2.3d
def test_theta_nb():
    global theta_nb
    assert_almost_equals(theta_nb[(u'ADJ',u'bad')], -5.38657, places=3)
    assert_almost_equals(theta_nb[(u'PRON',u'.')], -14.44537, places=3)
    assert_almost_equals(theta_nb[(u'NOUN',u'.')], -15.0676, places=3)
    assert_almost_equals(theta_nb[(u'PUNCT',u'.')], -1.01587, places=3)

# 2.3e
def test_nb_offset_weights():
    global theta_nb
    assert_almost_equals(theta_nb[('VERB'),constants.OFFSET], -2.83321334406, places=3)
    assert_almost_equals(theta_nb[('NOUN'),constants.OFFSET], -2.83321334406, places=3)
    assert_almost_equals(theta_nb[('PUNCT'),constants.OFFSET], -2.83321334406, places=3)

# 2.4a (0.5 points)
def test_nb2_emissions_are_same():
    global theta_nb, theta_nb_fixed
    for feature in [('PRON','.'),
                    ('PUNCT','.'),
                    ('ADJ','okay'),
                    ('PRON','she')]:
        assert_almost_equals(theta_nb[feature],
                             theta_nb_fixed[feature],
                             places=7)

# 2.4b
def test_nb2_offsets_are_normalized():
    global theta_nb_fixed, sorted_tags
    assert_almost_equals(1,
                         sum(np.exp(theta_nb_fixed[(tag,constants.OFFSET)])
                             for tag in sorted_tags),
                         places=5)

# 2.4c
def test_nb2_offsets_are_correct():
    global theta_nb_fixed
    assert_almost_equals(-2.006, theta_nb_fixed[('VERB'),constants.OFFSET],places=2)
    assert_almost_equals(-2.965, theta_nb_fixed[('ADV'),constants.OFFSET],places=2)
    assert_almost_equals(-2.399, theta_nb_fixed[('PRON'),constants.OFFSET],places=2)

# 2.4d
def test_nb2_tagger_is_good():
    global theta_nb_fixed
    
    tagger = tagger_base.make_classifier_tagger(theta_nb_fixed)
    confusion = tagger_base.eval_tagger(tagger,'nb2')
    dev_acc = scorer.accuracy(confusion)

    expected_acc = 0.84
    
    ok_(expected_acc < dev_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, dev_acc))
