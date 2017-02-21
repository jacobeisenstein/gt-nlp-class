from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest, assert_not_equal

from gtnlplib.constants import * #This is bad and I'm sorry.
from gtnlplib import preproc
import numpy as np
import os

def setup():
    pass

def test_using_right_files():
    eq_(os.path.basename(TRAIN_FILE),"en-ud-simpler-train.conllu")

def test_correct_number_of_tags():
    ## Demo
    all_tags = set()
    for i,(words, tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE,max_insts=100000)):
        for tag in tags:
            all_tags.add(tag)
    eq_(len(all_tags),10)
