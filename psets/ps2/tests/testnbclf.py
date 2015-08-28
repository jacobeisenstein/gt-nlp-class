from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.clf_base import predict, evalClassifier
from gtnlplib.naivebayes import learnNBWeights, regularization_using_grid_search
from gtnlplib.constants import TRAINKEY, DEVKEY, TESTKEY, ALL_LABELS, OFFSET
from gtnlplib.preproc import getCountsAndKeys
import gtnlplib.scorer

weights_nb = {}
NB_FILE = "nb.txt"
mat = {}
counts, class_counts, allkeys = None, None, None

import os, errno

def silentremove(filename):
    """ Copied from the internet. """
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

def setup_module ():
    global weights_nb
    global counts
    global class_counts
    global allkeys
    counts, class_counts, allkeys = getCountsAndKeys (TRAINKEY)
    weights_nb = learnNBWeights (counts, class_counts, allkeys)

def teardown_module ():
    global NB_FILE
    silentremove (NB_FILE)

def test_nb_prediction_actual_label ():
    actual = predict ({'good':1,'worst':4,OFFSET:1}, weights_nb, ALL_LABELS)[0]
    expected = "NEG"
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_nb_prediction_scores_for_negative_label ():
    actual = predict ({'good':1,'worst':4,OFFSET:1}, weights_nb, ALL_LABELS)[1]["NEG"]
    expected = -31.0825
    assert_almost_equals (expected, actual, places=3, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

def test_nb_prediction_scores_for_positive_label ():
    actual = predict ({'good':1,'worst':4,OFFSET:1}, weights_nb, ALL_LABELS)[1]["POS"]
    expected = -39.8792
    assert_almost_equals (expected, actual, places=3, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@nottest
def setup_nb_testing ():
    global mat
    mat = evalClassifier (weights_nb, NB_FILE, DEVKEY)

@nottest
def teardown_nb_testing ():
    pass

@with_setup (setup_nb_testing, teardown_nb_testing)
def test_nb_dev_exact_accuracy ():
    global NB_FILE
    global weights_nb
    actual = gtnlplib.scorer.accuracy (mat)
    expected = 0.5177
    assert_almost_equals (expected, actual, places=3, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@with_setup (setup_nb_testing, teardown_nb_testing)
def test_nb_dev_almost_there_accuracy ():
    global NB_FILE
    global weights_nb
    actual = gtnlplib.scorer.accuracy (mat)
    expected = 0.5
    ok_ (expected < actual, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected, actual))

