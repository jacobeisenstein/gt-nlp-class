from nose.tools import with_setup, eq_, assert_almost_equal

from gtnlplib.constants import START_TAG, END_TAG, TRAIN_FILE
from gtnlplib import hmm, most_common
import numpy as np

def setup():
    global tag_trans_counts, hmm_trans_weights
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.001)


# 4.2a
def test_hmm_trans_weights_sum_to_one():
    global tag_trans_counts, hmm_trans_weights

    all_tags = list(tag_trans_counts.keys()) + [END_TAG]
    for tag in tag_trans_counts.keys():
        assert_almost_equal(sum(np.exp(hmm_trans_weights[(next_tag,'NOUN')]) for next_tag in all_tags),1,places=5)

# 4.2b
def test_hmm_trans_weights_exact_vals():
    global hmm_trans_weights

    assert_almost_equal(hmm_trans_weights[('NOUN',START_TAG)],-2.91877,places=3)
    assert_almost_equal(hmm_trans_weights[('VERB',START_TAG)],-3.44201,places=3)
    assert_almost_equal(hmm_trans_weights[('INTJ','DET')],-15.06970,places=3)
    assert_almost_equal(hmm_trans_weights[('NOUN','DET')],-0.69286,places=3)
    eq_(hmm_trans_weights[(START_TAG,'VERB')],-np.inf)
    

    
