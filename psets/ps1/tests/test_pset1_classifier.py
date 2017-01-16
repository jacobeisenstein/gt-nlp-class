from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from gtnlplib.preproc import *
#from gtnlplib.preproc_metrics import *
from gtnlplib import clf_base, constants, hand_weights, naive_bayes, evaluation, perceptron, logreg
import numpy as np

def setup_module():
    global y_tr, x_tr, corpus_counts, labels, vocab
    y_tr,x_tr = read_data('reddit-train.csv',
                         'subreddit',
                         preprocessor=tokenize_and_downcase)
    corpus_counts = get_corpus_counts(x_tr)
    labels = set(y_tr)
    
    global y_dv, x_dv
    y_dv,x_dv = read_data('reddit-dev.csv',
                          'subreddit',
                          preprocessor=tokenize_and_downcase)
    
    # your test set does not contain accurate labels, but ours does
    global y_te, x_te
    y_te,x_te = read_data('reddit-test.csv',
                         'subreddit',
                         preprocessor=tokenize_and_downcase)

    vocab = [word for word,count in corpus_counts.iteritems() if count > 10]

    x_tr = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_tr]
    x_dv = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_dv]
    x_te = [{key:val for key,val in x_i.iteritems() if key in vocab} for x_i in x_te]
    
# 0.2 points
def test_clf_base_d2_1():
    # public
    label = 'iama'
    fv = clf_base.make_feature_vector({'test':1,'case':2},label)
    eq_(len(fv),3)
    eq_(fv[(label,'test')],1)
    eq_(fv[(label,'case')],2)
    eq_(fv[(label,constants.OFFSET)],1)

# 0.4 points
def test_clf_base_d2_2():
    global x_tr, x_dv

    # public
    y_hat,scores = clf_base.predict(x_tr[5],hand_weights.theta_hand_original,labels)
    eq_(scores['iama'],0.1)
    eq_(scores['science'],5.0)
    eq_(y_hat,'science')
    eq_(scores['askreddit'],0.0)
    
# 0.4 points
def test_clf_base_d2_3():
    global x_dv, y_dv, y_te, labels

    y_hat = clf_base.predict_all(x_dv,hand_weights.theta_hand,labels)
    assert_greater_equal(evaluation.acc(y_hat,y_dv),.41)

    # just make sure the file is there
    y_hat_te = evaluation.read_predictions('hand-test.preds')
    eq_(len(y_hat_te),len(y_te))
    
def test_clf_base_d2_3_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy
    
    global y_te
    
    y_hat_te = evaluation.read_predictions('hand-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.35)

# 0.2 points
def test_nb_d3_1():
    # public
    iama_counts = naive_bayes.get_corpus_counts(x_tr,y_tr,unicode('iama'));
    eq_(iama_counts['four'],17)
    eq_(iama_counts['am'],255)
    eq_(iama_counts['internets'],0)

# 0.5 points
def test_nb_d3_2():
    global vocab, x_tr, y_tr
    
    # check that distribution normalizes to one
    log_pxy = naive_bayes.estimate_pxy(x_tr,y_tr,unicode('iama'),0.1,vocab)
    assert_almost_equals(np.exp(log_pxy.values()).sum(),1)

    # check that values are correct
    assert_almost_equals(log_pxy['science'],-8.5404,places=2)
    assert_almost_equals(log_pxy['world'],-7.3657,places=2)

# 0.8 points
def test_nb_d3_3():
    global x_tr, y_tr, x_dv, y_dv, x_te

    # public
    theta_nb = naive_bayes.estimate_nb(x_tr,y_tr,0.1)
    y_hat,scores = clf_base.predict(x_tr[55],theta_nb,labels)
    assert_almost_equals(scores['science'],-949.406,places=2)
    
def test_nb_d3_3_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('nb-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.7)
   
# 0.3 points
def test_nb_d3_4():
    global x_tr, y_tr, x_dv, y_dv
    best_smoother, scores = naive_bayes.find_best_smoother(x_tr,y_tr,x_dv,y_dv,[1e-3,1e-2,1e-1,1])
    assert_greater_equal(scores[.1],.72)
    assert_greater_equal(scores[.01],.73)

def test_nb_d3_4_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('nb-best-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.705)

# 0.5 points
def test_perc_d4_1():
    global x_tr, y_tr

    # public 
    labels = set(y_tr)
    theta_perc = hand_weights.theta_hand.copy()

    update = perceptron.perceptron_update(x_tr[110],y_tr[110],theta_perc,labels)
    eq_(len(update),0)

    update = perceptron.perceptron_update(x_tr[20],y_tr[20],theta_perc,labels)
    eq_(len(update),146)
    eq_(update[('science','and')],2)
    eq_(update[('iama','200')],-1)


# 0.5 points
def test_perc_d4_2():
    global y_dv, x_tr, y_tr

    # run on a subset of data
    theta_perc,theta_perc_history = perceptron.estimate_perceptron(x_tr[:10],y_tr[:10],3)
    eq_(theta_perc[('worldnews','its')],1)
    eq_(theta_perc[('science','its')],0)
    eq_(theta_perc[('science','what')],4)
    eq_(theta_perc[('worldnews','always')],-1)
    eq_(theta_perc_history[0][('science','what')],2)
    
    y_hat_dv = evaluation.read_predictions('perc-dev.preds')
    # i get 64.6% accuracy
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.62)

def test_perc_d4_2_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('perc-test.preds')
    # i get 64.0% accuracy
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.62)

# 1.0 points
def test_avp_d4_3():
    global y_dv, x_tr, y_tr

    # run on a subset of data
    theta_avp,theta_avp_history = perceptron.estimate_avg_perceptron(x_tr[:10],y_tr[:10],3)
    assert_almost_equals(theta_avp[('science','what')],3.2258,places=2)
    assert_almost_equals(theta_avp[('science','its')],0,places=2)
    assert_almost_equals(theta_avp[('worldnews','its')],0.871,places=2)
    
    y_hat_dv = evaluation.read_predictions('avp-dev.preds')
    # i get 66.4% accuracy
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.64)

def test_perc_d4_3_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('avp-test.preds')
    # i get 66.8% accuracy
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.645)

# 0.5 points
def test_lr_d5_1():
    global y_tr
    labels = set(y_tr)
    py_x = logreg.compute_py({'i':1,'am':2},
                             hand_weights.theta_hand_original,labels)
    assert_almost_equals(py_x['askreddit'],.19588,places=3)
    assert_almost_equals(py_x['iama'],.2165,places=3)

    py_x = logreg.compute_py({'i':1,'news':2,'science':1},
                             hand_weights.theta_hand_original,labels)
    assert_almost_equals(py_x['science'],.3182,places=3)
    assert_almost_equals(py_x['iama'],.1293,places=3)

# 1 point
def test_lr_d5_2():
    global x_tr, y_tr, y_dv, y_te

    # run on a subset of data
    theta_lr,theta_lr_hist = logreg.estimate_logreg(x_tr[:10],y_tr[:10],3)
    assert_almost_equals(theta_lr[('science','what')],.000402,places=4)
    assert_almost_equals(theta_lr[('iama', 'missile')],-0.00031832285759249263,places=4)
    assert_almost_equals(theta_lr[('iama',constants.OFFSET)],.00045298,places=4)
    assert_almost_equals(theta_lr[('askreddit',constants.OFFSET)],0.,places=4)

    # dev set accuracy
    y_hat_dv = evaluation.read_predictions('lr-dev.preds')
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.595)

def test_lr_d5_2_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('lr-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.55)

# 0.5 points
def test_lr_d5_3():
    global y_dv
    y_hat_dv = evaluation.read_predictions('lr-best-dev.preds')
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.66)

def test_lr_d5_3_test():
    # NOTE! This test is for the TAs to run
    # You cannot pass this test without the true test labels.
    # This is a sanity check to make sure your solution for 2.3 is not too crazy

    global y_te
    y_hat_te = evaluation.read_predictions('lr-best-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.63)

# 0.5 / 0.25 points
def test_lr_d6_1():
    top_feats_wn = clf_base.get_top_features_for_label(hand_weights.theta_hand_original,'worldnews',3)
    eq_(top_feats_wn[0],(('worldnews', 'worldnews'), 1.0))
    eq_(len(top_feats_wn),3)
    
    top_feats_ar = clf_base.get_top_features_for_label(hand_weights.theta_hand_original,'askreddit',3)
    eq_(top_feats_ar[1],(('askreddit', 'ask'), 0.5))
    eq_(len(top_feats_ar),2)

# compare against my dev results
# you are not required to pass this test
def test_feats_d7_1():
    global y_dv
    y_hat_dv = evaluation.read_predictions('bakeoff-dev.preds')
    assert_greater_equal(evaluation.acc(y_hat_dv,y_dv),.78)

# +1 extra credit if you pass this test
def test_feats_d7_1_test():
    global y_te
    y_hat_te = evaluation.read_predictions('bakeoff-test.preds')
    assert_greater_equal(evaluation.acc(y_hat_te,y_te),.722)

    
