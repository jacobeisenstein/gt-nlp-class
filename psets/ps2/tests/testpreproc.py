from nose.tools import with_setup, eq_, ok_, istest, nottest
from gtnlplib.constants import TRAINKEY, DEVKEY, TESTKEY
from gtnlplib.preproc import docsToBOWs, dataIterator, getAllCounts
from gtnlplib.preproc_metrics import get_token_type_ratio, type_frequency, unseen_types

ac_train, ac_dev = {}, {}

def setup_module():
    # Need to do this because the dataIterator function depends
    # on the BOW file to be generated.
    global ac_train
    global ac_dev
    docsToBOWs(TRAINKEY)
    docsToBOWs(DEVKEY)
    ac_train = getAllCounts (dataIterator (TRAINKEY))
    ac_dev = getAllCounts (dataIterator (DEVKEY))

def teardown_module ():
    pass

def test_number_of_tokens_in_training ():
    """ Number of tokens in training should exactly match with this number """
    actual = len (ac_train.keys()) - 1
    expected = 18430
    eq_ (actual, expected, msg="UNEQUAL Expected:%d, Actual:%d" %(expected, actual))

def test_token_type_ratio_for_train ():
    """ Token to type ratio must be within acceptable limits"""
    TOLERANCE = 0.1
    actual = get_token_type_ratio (ac_train)
    expected = 13.1482
    ok_ (abs (actual - expected) < TOLERANCE , msg = "OUT_OF_BOUND Expected:%f, Actual:%f" %(expected, actual))

def test_token_type_ratio_for_dev ():
    """ Token to type ratio must be within acceptable limits"""
    TOLERANCE = 0.1
    actual = get_token_type_ratio (ac_dev)
    expected = 6.923
    ok_ (abs (actual - expected) < 0.01 , msg = "OUT_OF_BOUND Expected:%f, Actual:%f" %(expected, actual))

def test_type_frequency_for_train ():
    """ Types occuring with certain frequency should match exactly"""
    actual = type_frequency (ac_train, 1)
    expected = 8758
    eq_ (actual, expected, msg = "UNEQUAL Expected:%d, Actual:%d" %(expected, actual))

def test_type_frequency_for_dev ():
    """ Types occuring with certain frequency should match exactly"""
    actual = type_frequency (ac_dev, 1)
    expected = 4738
    eq_ (actual, expected, msg = "UNEQUAL Expected:%d, Actual:%d" %(expected, actual))

def test_unseen_types ():
    """ Types not seen in training should match exactly"""
    actual = unseen_types (ac_train, ac_dev)
    expected = 2407
    eq_ (actual, expected, msg = "UNEQUAL Expected:%d, Actual:%d" %(expected, actual))

