from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest, assert_not_equal

from gtnlplib.constants import * #This is bad and I'm sorry.
from gtnlplib import hmm, viterbi, most_common, scorer
import numpy as np

def setup():
    global theta_hmm, all_tags
    theta_hmm, all_tags = hmm.compute_HMM_weights(TRAIN_FILE,.01)

# 4.2a
def test_hmm_weight_count():
    global theta_hmm
    eq_(len(theta_hmm),334428)
    
# 4.2b
def test_hmm_emit_weights_sum_to_one():
    global theta_hmm

    vocab = set([word for tag,word,feat_type in theta_hmm.keys() if feat_type==EMIT])
    assert_almost_equals(sum(np.exp(theta_hmm['NOUN',word,EMIT]) for word in vocab),1,places=5)
    assert_almost_equals(sum(np.exp(theta_hmm['ADP',word,EMIT]) for word in vocab),1,places=5)

# 4.2c
def test_hmm_emit_weights_exact_vals():
    global theta_hmm

    assert_almost_equals(theta_hmm['NOUN','right',EMIT],-7.41746204765,places=4)
    assert_almost_equals(theta_hmm['ADV','right',EMIT],-5.33071419765,places=4)
    assert_almost_equals(theta_hmm['PRON','she',EMIT],-4.5722924085,places=4)
    assert_almost_equals(theta_hmm['DET','she',EMIT],-14.3151646115,places=4)
    eq_(theta_hmm['ADJ','thisworddoesnotappear',EMIT],0)

# 4.3
def test_hmm_on_example_sentence():
    global theta_hmm, all_tags
    pred_tags, score = viterbi.viterbi_tagger(['they', 'can', 'can', 'fish'],hmm.hmm_features,theta_hmm,all_tags)

    assert_almost_equals(score,-31.2943, places=2)
    eq_(pred_tags,['PRON','AUX','AUX','NOUN'])

# 4.5a
def test_hmm_dev_accuracy():
    confusion = scorer.get_confusion(DEV_FILE,'hmm-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .875)

# 4.5b
def test_hmm_test_accuracy():
    confusion = scorer.get_confusion(TEST_FILE,'hmm-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .877)

# 5.2a
def test_ja_hmm_dev_accuracy():
    confusion = scorer.get_confusion(JA_DEV_FILE,'hmm-dev-ja.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .84)

# 5.2b
def test_ja_hmm_test_accuracy():
    confusion = scorer.get_confusion(JA_TEST_FILE,'hmm-test-ja.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .81)


