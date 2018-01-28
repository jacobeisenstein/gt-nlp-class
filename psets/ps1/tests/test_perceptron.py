from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from gtnlplib import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logreg
import numpy as np
from collections import Counter

def setup_module():
    #global y_tr, x_tr, corpus_counts, labels, vocab
    #corpus_counts = get_corpus_counts(x_tr)


    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned, x_bl_pruned
    global labels
    global vocab

    y_tr,x_tr = preproc.read_data('lyrics-train.csv',preprocessor=preproc.bag_of_words)
    labels = set(y_tr)

    counts_tr = preproc.aggregate_counts(x_tr)

    y_dv,x_dv = preproc.read_data('lyrics-dev.csv',preprocessor=preproc.bag_of_words)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_tr, 10)
    x_dv_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dv, 10)

def test_d4_1_perc_update():
    global x_tr_pruned, y_tr

    labels = set(y_tr)

    theta_perc = Counter()
    update = perceptron.perceptron_update(x_tr_pruned[20],y_tr[20],theta_perc,labels)
    eq_(len(update),0)

    update = perceptron.perceptron_update(x_tr_pruned[110],y_tr[110],theta_perc,labels)
    eq_(len(update),122)
    eq_(update[('2000s','with')],1)
    eq_(update[('1980s','shes')],-2)
    eq_(update[('2000s',constants.OFFSET)],1)
    eq_(update[('1980s',constants.OFFSET)],-1)

def test_d4_2a_perc_estimate():
    global y_dv, x_tr_pruned, y_tr

    # run on a subset of data
    theta_perc,theta_perc_history = perceptron.estimate_perceptron(x_tr_pruned[:10],y_tr[:10],3)
    eq_(theta_perc[('2000s','its')],-1)
    eq_(theta_perc[('2000s','what')],1)
    eq_(theta_perc[('1980s','what')],4)
    eq_(theta_perc[('1980s','its')],-15)
    eq_(theta_perc_history[0][('1980s','what')],2)
    

def test_d4_2b_perc_accuracy():
    global y_dv
    # i get 43% accuracy
    y_hat_dv = evaluation.read_predictions('perc-dev.preds')
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.43)

