from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal

from gtnlplib.constants import * 
from gtnlplib import hmm, viterbi, most_common, scorer, naive_bayes
import numpy as np

def setup():
    global nb_weights, hmm_trans_weights, all_tags
    nb_weights = naive_bayes.get_nb_weights(TRAIN_FILE, .01)
    tag_trans_counts = most_common.get_tag_trans_counts(TRAIN_FILE)
    hmm_trans_weights = hmm.compute_transition_weights(tag_trans_counts,.01)
    all_tags = list(tag_trans_counts.keys()) + [END_TAG]
    

# 4.2
def test_hmm_on_example_sentence():
    global nb_weights, hmm_trans_weights, all_tags
    tag_to_ix={}
    for tag in list(all_tags):
        tag_to_ix[tag]=len(tag_to_ix)
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE)
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights, \
                                                                         vocab, word_to_ix, tag_to_ix)
    
    score, pred_tags = viterbi.build_trellis(all_tags,
                                             tag_to_ix,
                                             [emission_probs[word_to_ix[w]] for w in ['they', 'can', 'can', 'fish','.']],
                                             tag_transition_probs)
    
    assert_almost_equal(score.data.numpy()[0],-32.4456, places=2)
    eq_(pred_tags,['PRON', 'AUX', 'AUX', 'NOUN','PUNCT'])

# 4.3a
def test_hmm_dev_accuracy():
    confusion = scorer.get_confusion(DEV_FILE,'hmm-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .840)

# 4.3b
def test_hmm_test_accuracy():
    confusion = scorer.get_confusion(TEST_FILE,'hmm-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .840)

# 4.4a
def test_nr_hmm_dev_accuracy():
    confusion = scorer.get_confusion(NR_DEV_FILE,'hmm-dev-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .861)

# 4.4b
def test_nr_hmm_test_accuracy():
    confusion = scorer.get_confusion(NR_TEST_FILE,'hmm-te-nr.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .853)


