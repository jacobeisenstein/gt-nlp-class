from nose.tools import with_setup, eq_, assert_almost_equals, ok_
from collections import defaultdict

from gtnlplib import coref_learning, neural_net, coref
from gtnlplib.coref_features import minimal_features
from gtnlplib.coref_learning import FFCoref
from gtnlplib.coref import Markable
from gtnlplib.utils import initialize_with_pretrained, UNK_TOKEN
from gtnlplib.neural_net import BiLSTMWordEmbedding, AttentionBasedMarkableEmbedding, SequentialScorer

import torch
import torch.autograd as ag
import torch.optim as optim

EMBEDDING_DIM = 64
TEST_EMBEDDING_DIM = 4
min_features = ['exact-match', 'last-token-match', 'content-match', 'crossover', 'new-entity']
COREF_FF_HIDDEN = 5
LSTM_HIDDEN = 10
LSTM_LAYERS = 1
DROPOUT = 0.1

def make_list(var_list):
    return map(lambda x: x.view(-1).data.tolist(), var_list)

def check_tensor_correctness(pairs):
    for pair in pairs:
        for pred, true in zip(pair[0], pair[1]):
            assert_almost_equals(pred, true, places=4)

def list_assert(l1, l2):
    for x1, x2 in zip(l1, l2):
        assert_almost_equals(x1, x2, places=4)
            
def setup():
    global test_doc, markables, word_to_ix, vocab
    test_doc = "I will not buy this record , it is scratched . It is by The Eagles and I hate them .".split()
    markables = [Markable([test_doc[0]], 'e1', 0, 1, ['PRP']),
                 Markable(test_doc[4:6], 'e2', 4, 6, ['DT', 'NN']),
                 Markable([test_doc[7]], 'e2', 7, 8, ['PRP']),
                 Markable([test_doc[11]], 'e2', 11, 12, ['PRP']),
                 Markable(test_doc[14:16], 'e3', 14, 16, ['NNP', 'NNP']),
                 Markable([test_doc[17]], 'e1', 17, 18, ['PRP']),
                 Markable([test_doc[19]], 'e3', 19, 20, ['PRP'])]
    vocab = set(test_doc)

    word_to_ix = { word: i for i, word in enumerate(sorted(vocab)) }


# ===-------------------------------------------------------------------------------------------===
# Section 3 tests
# ===-------------------------------------------------------------------------------------------===

## deliverable 3.2
def test_ffcoref_d3_2():
    torch.manual_seed(1)
    ff_coref = FFCoref(min_features, COREF_FF_HIDDEN)
    ltm_cm_ne = ff_coref(defaultdict(float, {'last-token-match':1, 'content-match':1, 'new-entity':1}))
    assert_almost_equals(float(ltm_cm_ne), 0.145386, places=4)
    ltm_cm = ff_coref(defaultdict(float, {'last-token-match':1, 'content-match':1}))
    assert_almost_equals(float(ltm_cm), 0.207934, places=4)
    ltm_cm = ff_coref(defaultdict(float, {'crossover':1}))
    assert_almost_equals(float(ltm_cm), -0.049335, places=4)

## deliverable 3.3
def test_ffcoref_score_instance_d3_3():
    global test_doc
    torch.manual_seed(1)
    ff_coref = FFCoref(min_features, COREF_FF_HIDDEN)
    inst_scores = ff_coref.score_instance(markables, minimal_features, 2)
    preds = list(make_list(inst_scores))[0]
    trues = [0.020467400550842285, 0.020467400550842285, -0.052015334367752075]
    list_assert(preds, trues)
    
## deliverable 3.4
def test_ffcoref_score_instance_top_scores_d3_4():
    global test_doc
    torch.manual_seed(1)
    ff_coref = FFCoref(min_features, COREF_FF_HIDDEN)
    s_t, s_f = ff_coref.instance_top_scores(markables, minimal_features, 6, 4)
    assert_almost_equals(float(s_t), 0.0204674, places=4)
    assert_almost_equals(float(s_f), 0.0204674, places=4)
    s_t2, s_f2 = ff_coref.instance_top_scores(markables, minimal_features, 0, 0)
    eq_(s_t2, None)
    eq_(s_f2, None)
    
# ===-------------------------------------------------------------------------------------------===
# Section 4 tests
# ===-------------------------------------------------------------------------------------------===

## deliverable 4.1
def test_bilstm_embedding_d4_1():
    global test_doc
    torch.manual_seed(1)
    lstm = BiLSTMWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM, LSTM_HIDDEN, LSTM_LAYERS, DROPOUT)
    pred_not = lstm(test_doc)[2].data.tolist()[0][:6]
    true_not = [0.11752596497535706,
                 0.042018793523311615,
                 0.06257987767457962,
                 -0.057494595646858215,
                 0.06428981572389603,
                 -0.16254858672618866]
    list_assert(pred_not, true_not)
    
    
## deliverable 4.2
def test_embedding_attention_d4_2():
    global test_doc
    torch.manual_seed(1)
    att = AttentionBasedMarkableEmbedding(TEST_EMBEDDING_DIM)
    dummy_embs = [ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM)) for d in test_doc]
    pred_this_record = att(dummy_embs, markables[1]).data.tolist()
    true_this_record = [0.35083580017089844,
                 0.5952423810958862,
                 0.5128775238990784,
                 0.18572629988193512]
    list_assert(pred_this_record, true_this_record)
    
## deliverable 4.3
def test_sequential_scorer_d4_3():
    global test_doc
    torch.manual_seed(1)
    seq = SequentialScorer(TEST_EMBEDDING_DIM, min_features, 2, COREF_FF_HIDDEN)
    emb5 = ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM))
    emb0 = ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM))
    pred = float(seq(emb5, emb0, ['exact-match', 'last-token-match']))
    assert_almost_equals(pred, -0.359851, places=4)
    
## deliverable 4.4a
def test_sequential_scorer_score_instance_d4_4():
    global test_doc
    torch.manual_seed(1)
    seq = SequentialScorer(TEST_EMBEDDING_DIM, min_features, 2, COREF_FF_HIDDEN)
    dummy_embs = [ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM)) for d in test_doc]
    pred_scores = seq.score_instance(dummy_embs, markables, 4, minimal_features).data.tolist()[0]
    true_scores = [-0.20210549235343933,
                 -0.20492248237133026,
                 -0.20492248237133026,
                 -0.19825465977191925,
                 -0.2360064834356308]
    list_assert(pred_scores, true_scores)
    
## deliverable 4.4b
def test_sequential_scorer_instance_top_scores_d4_4():
    global test_doc
    torch.manual_seed(1)
    seq = SequentialScorer(TEST_EMBEDDING_DIM, min_features, 2, COREF_FF_HIDDEN)
    dummy_embs = [ag.Variable(torch.rand(1, TEST_EMBEDDING_DIM)) for d in test_doc]
    pred_t, pred_f = seq.instance_top_scores(dummy_embs, markables, 3, 2, minimal_features)
    assert_almost_equals(float(pred_t), -0.229785, places=4)
    assert_almost_equals(float(pred_f), -0.214119, places=4)
    
## deliverable 4.5
def test_pretrain_embeddings_d4_5():
    torch.manual_seed(1)
    word_to_ix = { "interest": 0, "rate": 1, "swap": 2 }
    pretrained = { "interest": [ 6.1, 2.2, -3.5 ], "swap": [ 5.7, 1.6, 3.2 ], UNK_TOKEN: [8.5, -0.4, 2.0] }
    embedder = BiLSTMWordEmbedding(word_to_ix, 3, 2, 1, 0)
    initialize_with_pretrained(pretrained, embedder)

    embeddings = embedder.word_embeddings.weight.data

    pairs = []

    pairs.append((embeddings[word_to_ix["interest"]].tolist(), pretrained["interest"]))
    pairs.append((embeddings[word_to_ix["rate"]].tolist(), pretrained[UNK_TOKEN]))
    pairs.append((embeddings[word_to_ix["swap"]].tolist(), pretrained["swap"]))
    check_tensor_correctness(pairs)
