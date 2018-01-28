from nose.tools import eq_, assert_almost_equals, assert_greater_equal
from gtnlplib import preproc, clf_base, constants, hand_weights, evaluation, naive_bayes, perceptron, logreg
import numpy as np

import torch
from torch.autograd import Variable
from torch import optim

def setup_module():
    global x_tr, y_tr, x_dv, y_dv, counts_tr, x_dv_pruned, x_tr_pruned
    global labels
    global vocab
    global X_tr, X_tr_var, X_dv_var, Y_tr, Y_dv, Y_tr_var, Y_dv_var

    y_tr,x_tr = preproc.read_data('lyrics-train.csv',preprocessor=preproc.bag_of_words)
    labels = set(y_tr)

    counts_tr = preproc.aggregate_counts(x_tr)

    y_dv,x_dv = preproc.read_data('lyrics-dev.csv',preprocessor=preproc.bag_of_words)

    x_tr_pruned, vocab = preproc.prune_vocabulary(counts_tr, x_tr, 10)
    x_dv_pruned, _ = preproc.prune_vocabulary(counts_tr, x_dv, 10)

    ## remove this, so people can run earlier tests    
    X_tr = preproc.make_numpy(x_tr_pruned,vocab)
    X_dv = preproc.make_numpy(x_dv_pruned,vocab)
    label_set = sorted(list(set(y_tr)))
    Y_tr = np.array([label_set.index(y_i) for y_i in y_tr])
    Y_dv = np.array([label_set.index(y_i) for y_i in y_dv])

    X_tr_var = Variable(torch.from_numpy(X_tr.astype(np.float32)))
    X_dv_var = Variable(torch.from_numpy(X_dv.astype(np.float32)))

    Y_tr_var = Variable(torch.from_numpy(Y_tr))
    Y_dv_var = Variable(torch.from_numpy(Y_dv))

def test_d5_1_numpy():
    global x_dv, counts_tr
    
    x_dv_pruned, vocab = preproc.prune_vocabulary(counts_tr,x_dv,10)
    X_dv = preproc.make_numpy(x_dv_pruned,vocab)
    eq_(X_dv.sum(), 137687)
    eq_(X_dv.sum(axis=1)[4], 417)
    eq_(X_dv.sum(axis=1)[144], 175)

    eq_(X_dv.sum(axis=0)[10], 3)
    eq_(X_dv.sum(axis=0)[100], 0)

def test_d5_2_logreg():
    global X_tr, Y_tr, X_dv_var

    model = logreg.build_linear(X_tr,Y_tr)
    scores = model.forward(X_dv_var)
    eq_(scores.size()[0], 450)
    eq_(scores.size()[1], 4)

def test_d5_3_log_softmax():

    scores = np.asarray([[-0.1721,-0.5167,-0.2574,0.1571],[-0.3643,0.0312,-0.4181,0.4564]], dtype=np.float32)
    ans = logreg.log_softmax(scores)
    assert_almost_equals(ans[0][0], -1.3904355, places=5)
    assert_almost_equals(ans[1][1], -1.3458145, places=5)
    assert_almost_equals(ans[0][1], -1.7350391, places=5)

def test_d5_4_nll_loss():
    global X_tr, Y_tr, X_dv_var

    torch.manual_seed(765)
    model = logreg.build_linear(X_tr,Y_tr)
    model.add_module('softmax',torch.nn.LogSoftmax(dim=1))
    loss = torch.nn.NLLLoss()
    logP = model.forward(X_tr_var)
    assert_almost_equals(logreg.nll_loss(logP.data.numpy(), Y_tr), 1.5013313, places=5)

def test_d5_5_accuracy():
    global Y_dv_var
    acc = evaluation.acc(np.load('logreg-es-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.5)

def test_d7_3_bakeoff_dev1():
    global Y_dv_var
    acc = evaluation.acc(np.load('bakeoff-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.515)

def test_d7_3_bakeoff_dev2():
    global Y_dv_var
    acc = evaluation.acc(np.load('bakeoff-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.53)

def test_d7_3_bakeoff_dev3():
    global Y_dv_var
    acc = evaluation.acc(np.load('bakeoff-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.54)

def test_d7_3_bakeoff_dev4():
    global Y_dv_var
    acc = evaluation.acc(np.load('bakeoff-dev.preds.npy'),Y_dv_var.data.numpy())
    assert_greater_equal(acc,0.55)

# todo: implement test for bakeoff rubric
