from nose.tools import with_setup, eq_, ok_, assert_almost_equals, nottest
from gtnlplib.clf_base import evalClassifier
import gtnlplib.scorer as scorer
from gtnlplib.wordlist import loadSentimentWords, learnMCCWeights, learnWLCWeights
from gtnlplib.constants import SENTIMENT_FILE
from gtnlplib.constants import TRAINKEY, DEVKEY, TESTKEY

import os, errno

def silentremove(filename):
    """ Copied from the internet. """
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

poswords, negwords = {}, {}
MCC_FILE = "mcc.txt"
WLC_FILE = "wlc.txt"


def setup_module ():
    global poswords
    global negwords
    poswords, negwords = loadSentimentWords (SENTIMENT_FILE)

def teardown_module ():
    silentremove (MCC_FILE)
    silentremove (WLC_FILE)

@nottest
def setup_mcc_testing ():
    global weights_mcc
    weights_mcc = learnMCCWeights ()

@nottest
def teardown_wlc_testing ():
    pass

@nottest
def setup_wlc_testing ():
    global poswords
    global negwords
    global weights_wlc
    weights_wlc = learnWLCWeights (poswords, negwords)

@nottest
def teardown_mcc_testing ():
    pass

@with_setup (setup_mcc_testing, teardown_mcc_testing)
def test_mcc_dev_accuracy ():
    global weights_mcc
    global MCC_FILE
    mat = evalClassifier (weights_mcc, MCC_FILE, DEVKEY)
    actual = scorer.accuracy(mat)
    expected = 0.3756
    assert_almost_equals (expected, actual, places=4, msg="UNEQUAL Expected:%f, Actual:%f" %(expected,actual))

@with_setup (setup_wlc_testing, teardown_wlc_testing)
def test_wlc_dev_exact_accuracy ():
    global weights_wlc
    global WLC_FILE
    mat = evalClassifier (weights_wlc, WLC_FILE, DEVKEY)
    actual = scorer.accuracy(mat)
    expected = 0.4467
    assert_almost_equals (expected, actual, places=4, msg="UNEQUAL Expected:%f, Actual:%f" %(expected,actual))

@with_setup (setup_wlc_testing, teardown_wlc_testing)
def test_wlc_dev_almost_there_accuracy ():
    global weights_wlc
    global WLC_FILE
    mat = evalClassifier (weights_wlc, WLC_FILE, DEVKEY)
    actual = scorer.accuracy(mat)
    expected = 0.40
    ok_ (expected <= actual, msg="UNEQUAL Expected:%f, Actual:%f" %(expected,actual))
