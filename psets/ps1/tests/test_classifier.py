from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from gtnlplib import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logreg
import numpy as np

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


def test_d2_1_featvec():
    label = '1980s'
    fv = clf_base.make_feature_vector({'test':1,'case':2},label)
    eq_(len(fv),3)
    eq_(fv[(label,'test')],1)
    eq_(fv[(label,'case')],2)
    eq_(fv[(label,constants.OFFSET)],1)

def test_d2_2_predict():
    global x_tr_pruned, x_dv_pruned, y_dv

    y_hat,scores = clf_base.predict(x_tr_pruned[0],hand_weights.theta_hand,labels)
    eq_(scores['pre-1980'],0.1)
    assert_almost_equals(scores['2000s'],1.3,places=5)
    eq_(y_hat,'2000s')
    eq_(scores['1980s'],0.0)

    y_hat = clf_base.predict_all(x_dv_pruned,hand_weights.theta_hand,labels)
    assert_almost_equals(evaluation.acc(y_hat,y_dv),.3422222, places=5)

def test_d3_1_corpus_counts():
    # public
    iama_counts = naive_bayes.get_corpus_counts(x_tr_pruned,y_tr,"1980s");
    eq_(iama_counts['today'],50)
    eq_(iama_counts['yesterday'],14)
    eq_(iama_counts['internets'],0)


def test_d3_2_pxy():
    global vocab, x_tr_pruned, y_tr
    
    # check that distribution normalizes to one
    log_pxy = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"1980s",0.1,vocab)
    assert_almost_equals(np.exp(list(log_pxy.values())).sum(),1)

    # check that values are correct
    assert_almost_equals(log_pxy['money'],-7.6896,places=3)
    assert_almost_equals(log_pxy['fly'],-8.6369,places=3)

    log_pxy_more_smooth = naive_bayes.estimate_pxy(x_tr_pruned,y_tr,"1980s",10,vocab)
    assert_almost_equals(log_pxy_more_smooth['money'],-7.8013635125541789,places=3)
    assert_almost_equals(log_pxy_more_smooth['tonight'], -6.4054072405225515,places=3)

def test_d3_3a_nb():
    global x_tr_pruned, y_tr

    theta_nb = naive_bayes.estimate_nb(x_tr_pruned,y_tr,0.1)

    y_hat,scores = clf_base.predict(x_tr_pruned[55],theta_nb,labels)
    assert_almost_equals(scores['2000s'],-1840.5064690929203,places=3)
    eq_(y_hat,'1980s')

    y_hat,scores = clf_base.predict(x_tr_pruned[155],theta_nb,labels)
    assert_almost_equals(scores['1980s'], -2153.0199277981355, places=3)
    eq_(y_hat,'2000s')

def test_d3_3b_nb():
    global y_dv
    y_hat_dv = evaluation.read_predictions('nb-dev.preds')
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.46)

def test_d3_4a_nb_best():
    global x_tr_pruned, y_tr, x_dv_pruned, y_dv
    vals = np.logspace(-3,2,11)
    best_smoother, scores = naive_bayes.find_best_smoother(x_tr_pruned,y_tr,x_dv_pruned,y_dv,[1e-3,1e-2,1e-1,1])
    assert_greater_equal(scores[.1],.46)
    assert_greater_equal(scores[.01],.45)



