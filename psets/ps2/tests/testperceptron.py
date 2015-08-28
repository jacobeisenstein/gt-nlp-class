from nose.tools import with_setup, ok_, eq_, nottest, assert_almost_equals
from gtnlplib.perceptron import trainPerceptron
from gtnlplib.avg_perceptron import trainAvgPerceptron
from gtnlplib.constants import TRAINKEY, DEVKEY, TESTKEY, ALL_LABELS, OFFSET
from gtnlplib.preproc import loadInstances
from gtnlplib.clf_base import predict

P_FILE = "p.txt"
AP_FILE = "ap.txt"
all_tr_insts, all_dev_insts = None, None
wp,wap,tr_acc, dv_acc = None,None, None, None

import os, errno

def silentremove(filename):
    """ Copied from the internet. """
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occured

def setup_module ():
    global all_tr_insts
    global all_dev_insts
    all_tr_insts, all_dev_insts = loadInstances(TRAINKEY, DEVKEY)

def teardown_module ():
    global P_FILE
    global AP_FILE
    silentremove (P_FILE)
    silentremove (AP_FILE)

@nottest
def setup_perceptron_testing ():
    global all_tr_insts
    global wp
    global tr_acc
    global dv_acc
    wp,tr_acc,dv_acc = trainPerceptron(10,all_tr_insts, ALL_LABELS, P_FILE, DEVKEY)

@nottest
def teardown_perceptron_testing ():
    pass

@with_setup (setup_perceptron_testing, teardown_perceptron_testing)
def test_perceptron_prediction_actual_label ():
    global wp
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wp, ALL_LABELS)[0]
    expected = "NEG"
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

@with_setup (setup_perceptron_testing, teardown_perceptron_testing)
def test_perceptron_prediction_scores_for_negative_label ():
    global wp
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wp, ALL_LABELS)[1]["NEG"]
    expected = 293.0
    assert_almost_equals (expected, actual, places=3, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@with_setup (setup_perceptron_testing, teardown_perceptron_testing)
def test_perceptron_prediction_scores_for_positive_label ():
    global wp
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wp, ALL_LABELS)[1]["POS"]
    expected = -190.0
    assert_almost_equals (expected, actual, places=3, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))



@with_setup (setup_perceptron_testing, teardown_perceptron_testing)
def test_perceptron_minimum_accuracy ():
    global tr_acc
    expected = 0.8
    actual = tr_acc[-1]
    ok_ (expected <= actual, msg = "UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

def test_perceptron_maximum_accuracy ():
    global tr_acc
    expected = 0.83
    actual = tr_acc[-1]
    ok_ (expected <= actual, msg = "UNEQUAL Expected:%f, Actual:%f" %(expected, actual))


@nottest
def setup_avg_perceptron_testing ():
    global all_tr_insts
    global wap
    global tr_acc
    global dv_acc
    wap,tr_acc,dv_acc = trainAvgPerceptron(10, all_tr_insts, ALL_LABELS, AP_FILE, DEVKEY)

@nottest
def teardown_avg_perceptron_testing ():
    pass

@with_setup (setup_avg_perceptron_testing, teardown_avg_perceptron_testing)
def test_avg_perceptron_prediction_actual_label ():
    global wap
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wap, ALL_LABELS)[0]
    expected = "NEG"
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

@with_setup (setup_avg_perceptron_testing, teardown_avg_perceptron_testing)
def test_avg_perceptron_prediction_scores_for_negative_label ():
    global wap
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wap, ALL_LABELS)[1]["NEG"]
    expected = 188.3729
    assert_almost_equals (expected, actual, places=1, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@with_setup (setup_avg_perceptron_testing, teardown_avg_perceptron_testing)
def test_avg_perceptron_prediction_scores_for_positive_label ():
    global wap
    actual = predict ({'good':1,'worst':4,OFFSET:1}, wap, ALL_LABELS)[1]["POS"]
    expected = -121.7464
    assert_almost_equals (expected, actual, places=1, msg="UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@with_setup (setup_avg_perceptron_testing, teardown_avg_perceptron_testing)
def test_avg_perceptron_minimum_accuracy ():
    global dv_acc
    expected = 0.55
    actual = dv_acc[-1]
    ok_ (expected <= actual, msg = "UNEQUAL Expected:%f, Actual:%f" %(expected, actual))

@with_setup (setup_avg_perceptron_testing, teardown_avg_perceptron_testing)
def test_avg_perceptron_maximum_accuracy ():
    global dv_acc
    expected = 0.57
    actual = dv_acc[-1]
    ok_ (expected <= actual, msg = "UNEQUAL Expected:%f, Actual:%f" %(expected, actual))
