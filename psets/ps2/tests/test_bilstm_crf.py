from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
import torch
from gtnlplib.constants import * 
from gtnlplib import preproc, bilstm, hmm, viterbi, most_common, scorer
import numpy as np

def setup():
    global word_to_ix, tag_to_ix, X_tr, Y_tr, model
    
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, max_size=6500)
    tag_to_ix={}
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    
    if START_TAG not in tag_to_ix:
        tag_to_ix[START_TAG]=len(tag_to_ix)
    if END_TAG not in tag_to_ix:
        tag_to_ix[END_TAG]=len(tag_to_ix)
    
    torch.manual_seed(765);
    
    embedding_dim=30
    hidden_dim=30
    model = bilstm.BiLSTM_CRF(len(word_to_ix),tag_to_ix,embedding_dim, hidden_dim)
    
    X_tr = []
    Y_tr = []
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        X_tr.append(words)
        Y_tr.append(tags)

#6.1
def test_forward_alg():
    global model, X_tr, word_to_ix
    torch.manual_seed(765);
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    alpha = model.forward_alg(lstm_feats)
    assert_almost_equal(alpha.data.numpy()[0], 96.48747, places=4)

    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[1], word_to_ix))
    alpha = model.forward_alg(lstm_feats)
    assert_almost_equal(alpha.data.numpy()[0], 59.80174, places=4)

#6.2
def test_score_sentence():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(765);
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    score = model.score_sentence(lstm_feats, bilstm.prepare_sequence(Y_tr[0], tag_to_ix))
    assert_almost_equal(score.data.numpy()[0], -11.368162, places=4)
    
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[1], word_to_ix))
    score = model.score_sentence(lstm_feats, bilstm.prepare_sequence(Y_tr[1], tag_to_ix))
    assert_almost_equal(score.data.numpy()[0], -3.9872737, places=4)

#6.3
def test_predict():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(765);
    best_tags = model.predict(bilstm.prepare_sequence(X_tr[5], word_to_ix))
    eq_(best_tags[0:5],['SYM', 'NUM', 'INTJ', 'PART', 'SYM'])
    
    best_tags = model.predict(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    eq_(best_tags[0:5],['SYM', 'NUM', 'INTJ', 'PART', 'SYM'])

#6.4
def test_neg_log_likelihood():
    global model, X_tr, Y_tr, word_to_ix, tag_to_ix
    torch.manual_seed(765);
    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[5], word_to_ix))
    loss = model.neg_log_likelihood(lstm_feats, bilstm.prepare_sequence(Y_tr[5], tag_to_ix))
    assert_almost_equal(loss.data.numpy()[0],45.898315, places=4)

    lstm_feats = model.forward(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    loss = model.neg_log_likelihood(lstm_feats, bilstm.prepare_sequence(Y_tr[0], tag_to_ix))
    assert_almost_equal(loss.data.numpy()[0],107.71689, places=4)
    