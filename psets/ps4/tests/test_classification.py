import numpy as np
from collections import defaultdict
from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE, TRANS, END_TAG, START_TAG, EMIT, OFFSET,CURR_SUFFIX, PREV_SUFFIX
from gtnlplib import viterbi, clf_base, preproc, scorer, constants, tagger_base, features,avg_perceptron


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


# Deliverable 4a (1 point) 
def test_word_char_features():
    expected1 = ({(CURR_SUFFIX, 'V', 'n'): 1, (EMIT, 'V', 'can'): 1, (OFFSET, 'V'): 1, (PREV_SUFFIX, 'V', 'y'): 1})
    actual1  = features.wordCharFeatures(sent,'V','V',1)
    # print expected1, actual1
    eq_ (expected1, actual1, msg="UNEQUAL Expected:%s, Actual:%s" %(expected1, actual1))

    expected2 = ({(CURR_SUFFIX, 'V', 'y'): 1, (EMIT, 'V', 'they'): 1, (OFFSET, 'V'): 1})
    actual2  = features.wordCharFeatures(sent,'V','V',0)
    eq_ (expected2, actual2, msg="UNEQUAL Expected:%s, Actual:%s" %(expected2, actual2))

#Deliverable 4b (1 point)
def test_basic_classifer():
    test_weights = defaultdict(float)
    test_tags = ['N','V','V','N']
    for i in range(len(sent)):
        for feat in features.wordFeatures(sent,test_tags[i],'X',i):
            test_weights[feat] = 1
        for feat in features.wordFeatures(sent,'X','X',i):
            test_weights[feat] = 1
    expected = test_tags
    actual = tagger_base.classifierTagger(sent,features.wordFeatures,test_weights,alltags)
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual) )

    expected_acc = 0.139539705577
    confusion = tagger_base.evalTagger(lambda words,alltags : tagger_base.classifierTagger(words,features.wordFeatures,test_weights,alltags),'test')
    actual_acc =scorer.accuracy(confusion)
    assert_almost_equals(expected_acc ,actual_acc,places = 3)

#Deliverable 4c (3 points)
def test_one_it_avg_perceptron():
    weights,wsum,tr_acc,i = avg_perceptron.oneItAvgPerceptron(tr_all,features.wordFeatures,defaultdict(float),defaultdict(float),alltags)

    assert_almost_equals(16.0, weights[constants.EMIT,'D','the'], places = 3) 
    assert_almost_equals(2611.0, wsum[constants.EMIT,'D','the'], places = 3)

    assert_almost_equals(-1.0, weights[constants.EMIT,'N','the'], places = 3) 
    assert_almost_equals(-212.0, wsum[constants.EMIT,'N','the'], places = 3)
    
    assert_almost_equals(2.0, weights[constants.EMIT,'V','like'], places = 3) 
    assert_almost_equals(587.0, wsum[constants.EMIT,'V','like'], places = 3)
    
    assert_almost_equals(5.0, weights[constants.EMIT,'P','like'], places = 3) 
    assert_almost_equals(942.0, wsum[constants.EMIT,'P','like'], places = 3)

#Deliverable 4d (2 points)
def test_avg_perceptron():
    # w, tr_acc, dv_acc =  avg_perceptron.trainAvgPerceptron(10,tr_all,features.wordCharFeatures,alltags)
    # confusion = tagger_base.evalTagger(lambda words,alltags : tagger_base.classifierTagger(words,features.wordCharFeatures,w,alltags),'classifier')
    confusion = scorer.getConfusion(constants.DEV_FILE, 'avg_perceptron.response')
    expected_acc = 0.740
    actual_acc = scorer.accuracy(confusion)
    ok_ (expected_acc < actual_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, actual_acc))

#Deliverable 4e (3 points)
def test_custom_feat_avg_perceptron():
    # w, tr_acc, dv_acc =  avg_perceptron.trainAvgPerceptron(10,tr_all,features.yourFeatures,alltags)
    # confusion = tagger_base.evalTagger(lambda words,alltags : tagger_base.classifierTagger(words,features.yourFeatures,w,alltags),'classifier')
    confusion = scorer.getConfusion(constants.DEV_FILE, 'avg_perceptron_custom.response')
    expected_acc = 0.810
    actual_acc = scorer.accuracy(confusion)
    ok_ (expected_acc < actual_acc, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected_acc, actual_acc))