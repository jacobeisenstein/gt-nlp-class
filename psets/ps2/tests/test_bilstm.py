from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal
import torch
from gtnlplib.constants import * 
from gtnlplib import preproc, bilstm, hmm, viterbi, most_common, scorer
import numpy as np

def setup():
    global word_to_ix, tag_to_ix, X_tr, Y_tr, model, embedding_dim
    
    vocab, word_to_ix = most_common.get_word_to_ix(TRAIN_FILE, max_size=6500)
    tag_to_ix={}
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    
    
    torch.manual_seed(765);
    
    embedding_dim=30
    hidden_dim=30
    model = bilstm.BiLSTM(len(word_to_ix),tag_to_ix,embedding_dim, hidden_dim)
    
    X_tr = []
    Y_tr = []
    for i,(words,tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE)):
        X_tr.append(words)
        Y_tr.append(tags)
    
#5.1
def test_dlmodel_init():
    global model, embedding_dim, tag_to_ix, word_to_ix
    
    eq_(model.lstm.hidden_size,embedding_dim//2)
    eq_(model.hidden2tag.out_features, len(tag_to_ix))
    eq_(model.word_embeds.embedding_dim, embedding_dim)
    eq_(model.word_embeds.num_embeddings, len(word_to_ix))
    
#5.2
def test_dlmodel_forward():
    global model, X_tr, word_to_ix
    
    torch.manual_seed(765);
    lstm_feats = model(bilstm.prepare_sequence(X_tr[0], word_to_ix))
    assert_almost_equal(lstm_feats[0].data.numpy()[0],-0.12883559, places=4)
    assert_almost_equal(lstm_feats[0].data.numpy()[1],-0.42470014, places=4)
    assert_almost_equal(lstm_feats[0].data.numpy()[2],-0.31790152, places=4)
#5.3a
def test_bilstm_dev_accuracy():
    confusion = scorer.get_confusion(DEV_FILE,'bilstm-dev-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .85) #change the no's
#5.3b
def test_bilstm_test_accuracy():
    confusion = scorer.get_confusion(DEV_FILE,'bilstm-te-en.preds')
    acc = scorer.accuracy(confusion)
    ok_(acc > .83) #change the no's
