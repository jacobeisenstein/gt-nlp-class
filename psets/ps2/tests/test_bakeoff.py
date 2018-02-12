from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
import torch
from gtnlplib.constants import * 
import numpy as np

#8.1a
def test_model_en_dev_accuracy1():
    confusion = scorer.get_confusion(DEV_FILE,'bakeoff-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .88)

#8.1b    
def test_model_en_test_accuracy1():
    confusion = scorer.get_confusion(TEST_FILE,'bakeoff-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .87)
    

#8.1c
def test_model_nr_dev_accuracy1():
    confusion = scorer.get_confusion(NR_DEV_FILE,'bakeoff-dev-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .89)
    

#8.1d  
def test_model_nr_test_accuracy1():
    confusion = scorer.get_confusion(NR_TEST_FILE,'bakeoff-te-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .88)





