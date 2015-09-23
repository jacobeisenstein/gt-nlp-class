import numpy as np
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE, TRANS, END_TAG, START_TAG
from gtnlplib import viterbi, clf_base, preproc, scorer, constants, tagger_base


weights_nb = {}
mat = {}
defined_weights, alltags, allkeys = None, None, None

import os, errno

def setup_module ():
    global defined_weights
    global alltags
    global hmm_weights
    start_tag = constants.START_TAG
    trans = constants.TRANS
    end_tag = constants.END_TAG
    emit = constants.EMIT
    hmm_weights = viterbi.get_HMM_weights(TRAIN_FILE)
    defined_weights = {('N','they',emit):-1,('N','can',emit):-3,('N','fish',emit):-3,\
        ('V','they',emit):-10,('V','can',emit):-2,('V','fish',emit):-3,\
            ('N','N',trans):-5,('V','N',trans):-2,(end_tag,'N',trans):-3,\
                ('N','V',trans):-1,('V','V',trans):-4,(end_tag,'V',trans):-3,\
                    ('N',start_tag,trans):-1,('V',start_tag,trans):-1}
    alltags = preproc.getAllTags(TRAIN_FILE)

def test_hmm_weights():
    simple_weights = viterbi.get_HMM_weights('tests/hmm_simple.dat')
    assert_almost_equals(1.0, np.exp(simple_weights[('D', START_TAG, TRANS)]), places=3)
    assert_almost_equals(1.0, np.exp(simple_weights[('N', 'D', TRANS)]), places=3)
    assert_almost_equals(1.0, np.exp(simple_weights[('V', 'N', TRANS)]), places=3)
    assert_almost_equals(1.0, np.exp(simple_weights[('JJ', 'V', TRANS)]), places=3)
    assert_almost_equals(1.0, np.exp(simple_weights[(END_TAG, 'JJ', TRANS)]), places=3)

def test_tag_simple_seq():
    expected = (['N', 'V', 'N', 'V'])
    actual = viterbi.viterbiTagger(['they','can','can','fish'],viterbi.hmm_feats,defined_weights,['N','V'])[0]
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_tag_simple_score():
    expected = (-18)
    actual = viterbi.viterbiTagger(['they','can','can','fish'],viterbi.hmm_feats,defined_weights,['N','V'])[1]
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_tag_complex_seq():
    expected = (['N', 'V', 'N', 'V', 'N', 'V', 'N', 'V', 'N'])
    actual = viterbi.viterbiTagger('they can can can can can can can fish'.split(),viterbi.hmm_feats,defined_weights,['N','V'])[0]
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_tag_complex_score():
    expected = (-37)
    actual = viterbi.viterbiTagger('they can can can can can can can fish'.split(),viterbi.hmm_feats,defined_weights,['N','V'])[1]
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_hmm_weights_tag():
    expected = (['E', 'O', 'V', 'V', 'N', 'E'])
    actual = viterbi.viterbiTagger([':))','we','can','can','fish',':-)'],viterbi.hmm_feats,hmm_weights,alltags)[0]
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_hmm_weights_accuracy():
    confusion = tagger_base.evalTagger(lambda words, alltags : viterbi.viterbiTagger(words,viterbi.hmm_feats,hmm_weights,alltags)[0],'hmm')
    actual =  scorer.accuracy(confusion)
    expected = 0.74
    ok_ (expected < actual, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected, actual))
