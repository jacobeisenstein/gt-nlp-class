from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from gtnlplib import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logreg, features
import numpy as np
import torch

def setup_module():
    global vocab, label_set, x_tr_pruned, X_tr

    y_tr,x_tr = preproc.read_data('lyrics-train.csv',preprocessor=preproc.bag_of_words)
    labels = set(y_tr)

    counts_tr = preproc.aggregate_counts(x_tr)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_tr, 10)

    X_tr = preproc.make_numpy(x_tr_pruned,vocab)
    label_set = sorted(list(set(y_tr)))

def test_d6_1_topfeat_numpy():
    top_feats_two = features.get_top_features_for_label_numpy(hand_weights.theta_hand,'2000s',3)
    eq_(top_feats_two[0],(('2000s', 'name'), 0.2))
    eq_(len(top_feats_two),3)
    
    top_feats_eighty = features.get_top_features_for_label_numpy(hand_weights.theta_hand,'1980s',3)
    eq_(top_feats_eighty[0],(('1980s', 'tonight'), 0.1))
    eq_(len(top_feats_eighty),1)

def test_d6_2_topfeat_torch():
	global vocab, label_set
	model_test = torch.load('tests/test_weights.torch')

	top_feats_two = features.get_top_features_for_label_torch(model_test, vocab, label_set,'2000s',5)
	eq_(top_feats_two, ['like', 'this', 'im', 'girl', 'up'])

	top_feats_nine = features.get_top_features_for_label_torch(model_test, vocab, label_set,'1990s',7)
	eq_(top_feats_nine, ['here', 'power', 'jam', 'saw', 'yeah', 'want', 'yall'])

def test_d7_1_token_type_ratio():
	global X_tr

	ratios = [features.get_token_type_ratio(X_tr[i]) for i in range(5)]
	assert_almost_equals(ratios[0], 5.08333, places=2)
	assert_almost_equals(ratios[1], 2.6, places=2)
	assert_almost_equals(ratios[2], 1.91139, places=2)
	assert_almost_equals(ratios[3], 2.31884, places=2)
	assert_almost_equals(ratios[4], 6.18868, places=2)

def test_d7_2_discretize():
	global X_tr

	X_tr_new = features.concat_ttr_binned_features(X_tr)
	eq_(X_tr_new.shape[1], 4882)
	eq_(X_tr_new[0][-2], 1)
	eq_(X_tr_new[4][-1], 1)
	eq_(X_tr_new[4][-2], 0)
	eq_(X_tr_new[1][98], 3)