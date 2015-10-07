import numpy as np
from collections import defaultdict
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE, TRANS, END_TAG, START_TAG, EMIT, OFFSET,CURR_SUFFIX, PREV_SUFFIX, DEV_FILE
from gtnlplib import viterbi, clf_base, preproc, scorer, constants, tagger_base, features,str_perceptron


alltags, allkeys = None, None

import os, errno

def setup_module ():
    global alltags
    global sent
    global tr_all
    start_tag = constants.START_TAG
    trans = constants.TRANS
    end_tag = constants.END_TAG
    emit = constants.EMIT
    
    alltags = preproc.getAllTags(TRAIN_FILE)
    tr_all = preproc.getAllData(TRAIN_FILE)
    sent = 'they can can fish'.split()


#Deliverable 5a (1 point)
def test_seq_features():
    expected = ({(EMIT, 'N', 'fish'): 1.0, (EMIT, 'V', 'can'): 2.0, (OFFSET, 'V'): 2.0, (EMIT, 'N', 'they'): 1.0, \
        (OFFSET, 'N'): 2.0, (OFFSET, END_TAG): 1.0})
    actual = features.seqFeatures(sent,['N','V','V','N'],features.wordFeatures)
    eq_(expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual) )


#Deliverable 5b (1 point)
def test_seq_trans_features():
    expected = ({(TRANS, 'N', '--START--'): 1.0, (TRANS, '--END--', 'N'): 1.0, (EMIT, 'N', 'fish'): 1.0, (EMIT, 'V', 'can'): 2.0, \
        (OFFSET, 'V'): 2.0, (EMIT, 'N', 'they'): 1.0, (TRANS, 'V', 'V'): 1.0, (TRANS, 'N', 'V'): 1.0, (OFFSET, 'N'): 2.0, (TRANS, 'V', 'N'): 1.0,\
         (OFFSET, END_TAG): 1.0})
    actual = features.seqFeatures(sent,['N','V','V','N'],features.wordTransFeatures)
    eq_(expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual) )

#Deliverable 5c (1 point)
def test_viterbi_trans():
    test_weights = defaultdict(float)
    test_tags = ['N','V','V','N']
    for i in range(len(sent)):
        for feat in features.wordFeatures(sent,test_tags[i],'X',i):
            test_weights[feat] = 1
        for feat in features.wordFeatures(sent,'X','X',i):
            test_weights[feat] = 1
    
    expected_output = test_tags
    expected_score = 8.0
    actual_output, actual_score=viterbi.viterbiTagger(sent,features.wordTransFeatures,test_weights,alltags)
    eq_(expected_output, actual_output, msg="UNEQUAL viterbi trans output Expected:%s, Actual:%s" %(expected_output, actual_output) )
    eq_(expected_score, actual_score, msg="UNEQUAL viterbi trans score Expected:%s, Actual:%s" %(expected_score, actual_score) )

#Deliverable 5d (2 points)
def test_one_it_str_perceptron():
    weights,wsum,tr_acc,i = str_perceptron.oneItAvgStructPerceptron(tr_all[:100],features.wordTransFeatures,defaultdict(float),defaultdict(float),alltags)

    assert_almost_equals(5.0, weights[(constants.TRANS,'!','@')], places = 3)
    assert_almost_equals(-75.0, wsum[(constants.TRANS,'!','@')], places = 3)

    assert_almost_equals(2.0, weights[(constants.TRANS,'&','@')], places = 3)
    assert_almost_equals(127.0, wsum[(constants.TRANS,'&','@')], places = 3)

    assert_almost_equals(3.0, weights[(constants.TRANS,',','A')], places = 3)
    assert_almost_equals(158.0, wsum[(constants.TRANS,',','A')], places = 3)

    assert_almost_equals(-11.0, weights[(constants.TRANS,'A','A')], places = 3)
    assert_almost_equals(-400.0, wsum[(constants.TRANS,'A','A')], places = 3)



#Deliverable 5e (2 points)
def test_str_perceptron_small():
    w,tr_acc,dv_acc = str_perceptron.trainAvgStructPerceptron(5,tr_all[:50],features.wordTransFeatures,alltags)
    confusion = tagger_base.evalTagger(lambda words,alltags : viterbi.viterbiTagger(words,features.wordTransFeatures,w,alltags)[0],'str_classifier_small')
    expected_acc = 0.506
    actual_acc = scorer.accuracy(confusion)
    ok_ (expected_acc < actual_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, actual_acc))

# Deliverable 5e (2 points)
def test_str_perceptron():
    # w,tr_acc,dv_acc = str_perceptron.trainAvgStructPerceptron(10,tr_all,features.wordTransFeatures,alltags)
    # confusion = tagger_base.evalTagger(lambda words,alltags : viterbi.viterbiTagger(words,features.wordTransFeatures,w,alltags)[0],'str_classifier')
    confusion = scorer.getConfusion(DEV_FILE, 'str_avg_perceptron.response')
    expected_acc = 0.749
    actual_acc = scorer.accuracy(confusion)
    ok_ (expected_acc < actual_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, actual_acc))   

#Deliverable 5f (3 points)
def test_custom_str_perceptron():
    # w,tr_acc,dv_acc = str_perceptron.trainAvgStructPerceptron(10,tr_all,features.yourHMMFeatures,alltags)
    # confusion = tagger_base.evalTagger(lambda words,alltags : viterbi.viterbiTagger(words,features.yourHMMFeatures,w,alltags)[0],'custom_str_classifier')
    confusion = scorer.getConfusion(DEV_FILE, 'str_avg_perceptron_custom.response')
    expected_acc = 0.810
    actual_acc = scorer.accuracy(confusion)
    ok_ (expected_acc < actual_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, actual_acc))    