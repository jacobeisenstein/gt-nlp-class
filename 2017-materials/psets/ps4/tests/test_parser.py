from nose.tools import with_setup, eq_, assert_almost_equals, ok_
from gtnlplib.parsing import ParserState, TransitionParser, DepGraphEdge, train
from gtnlplib.utils import DummyCombiner, DummyActionChooser, DummyWordEmbeddingLookup, DummyFeatureExtractor, initialize_with_pretrained
from gtnlplib.data_tools import Dataset
from gtnlplib.constants import TRAIN_FILE, DEV_FILE, TEST_FILE, END_OF_INPUT_TOK, NULL_STACK_TOK, D3_2_DEV_FILENAME, DEV_GOLD, D4_4_DEV_FILENAME
from gtnlplib.evaluation import compute_metric, fscore, dependency_graph_from_oracle
from gtnlplib.feat_extractors import SimpleFeatureExtractor
from gtnlplib.neural_net import ActionChooserNetwork, MLPCombinerNetwork, VanillaWordEmbeddingLookup, BiLSTMWordEmbeddingLookup, LSTMCombinerNetwork

import torch
import torch.autograd as ag
import torch.optim as optim

EMBEDDING_DIM = 64
TEST_EMBEDDING_DIM = 4
NUM_FEATURES = 3

def make_dummy_parser_state(sentence):
    dummy_embeds = [ w + "-EMBEDDING" for w in sentence ] + [END_OF_INPUT_TOK + "-EMBEDDING"]
    return ParserState(sentence + [END_OF_INPUT_TOK], dummy_embeds, DummyCombiner())

def make_list(var_list):
    return map(lambda x: x.view(-1).data.tolist(), var_list)

def check_tensor_correctness(pairs):
    for pair in pairs:
        for pred, true in zip(pair[0], pair[1]):
            assert_almost_equals(pred, true, places=4)


def setup():
    global test_sent, gold, word_to_ix, vocab
    test_sent = "The man saw the dog with the telescope".split() + [END_OF_INPUT_TOK]
    gold = [ "SHIFT", "SHIFT", "REDUCE_L", "SHIFT", "REDUCE_L", "SHIFT",
                      "SHIFT", "REDUCE_L", "REDUCE_R", "SHIFT", "SHIFT", "SHIFT",
                      "REDUCE_L", "REDUCE_L", "REDUCE_R" ]
    vocab = set(test_sent)
    vocab.add(NULL_STACK_TOK)

    word_to_ix = { word: i for i, word in enumerate(vocab) }


# ===-------------------------------------------------------------------------------------------===
# Section 1 tests
# ===-------------------------------------------------------------------------------------------===

def test_stack_reduction_d1_1():
    """ 1 point(s) """

    global test_sent
    state = make_dummy_parser_state(test_sent)
    state.shift()
    state.shift()
    left_reduc = state.reduce_left()
    
    # head word should be "man"
    eq_(left_reduc[0][0], "man")
    eq_(left_reduc[0][1], 1)
    # dependent should be "The"
    eq_(left_reduc[1][0], "The")
    eq_(left_reduc[1][1], 0)

    # check the top of the stack to make sure combination was right
    eq_(state.stack[-1].headword, "man")
    eq_(state.stack[-1].headword_pos, 1)
    eq_(state.stack[-1].embedding, "man-EMBEDDING")

    state.shift()
    state.shift()
    right_reduc = state.reduce_right()

    # Head word should be "saw"
    eq_(right_reduc[0][0], "saw")
    eq_(right_reduc[0][1], 2)
    # dependent should be "the"
    eq_(right_reduc[1][0], "the")
    eq_(right_reduc[1][1], 3)

    eq_(state.stack[-1].headword, "saw")
    eq_(state.stack[-1].headword_pos, 2)
    eq_(state.stack[-1].embedding, "saw-EMBEDDING")


def test_stack_terminating_cond_d1_2():
    """ 0.5 point(s) """
    global test_sent
    state = make_dummy_parser_state(test_sent[:-1])
    
    assert not state.done_parsing()

    state.shift()
    state.shift()
    assert not state.done_parsing()

    state.shift()
    state.shift()
    state.reduce_left()
    state.reduce_left()
    state.reduce_left()

    assert not state.done_parsing()

    state.shift()
    state.shift()
    state.shift()
    state.shift()
    state.reduce_left()
    state.reduce_left()
    state.reduce_left()
    state.reduce_left()

    assert state.done_parsing()

# ////////////////////////////////////////////////////////////////////////////////////////////

# ===-------------------------------------------------------------------------------------------===
# Section 2 tests
# ===-------------------------------------------------------------------------------------------===

def test_word_embed_lookup_d2_1():
    """ 1 point(s) """

    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    embedder = VanillaWordEmbeddingLookup(word_to_ix, TEST_EMBEDDING_DIM)
    embeds = embedder(test_sent)
    assert len(embeds) == len(test_sent)
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], ag.Variable)
    assert embeds[0].size() == (1, TEST_EMBEDDING_DIM)

    embeds_list = make_list(embeds)

    true = ([-1.8661,  1.4146, -1.8781, -0.4674],
    [-0.9596,  0.5489, -0.9901, -0.3826],
    [0.5237,  0.0004, -1.2039,  3.5283],
    [0.3056,  1.0386,  0.5206, -0.5006],
    [0.4434,  0.5848,  0.8407,  0.5510],
    [-0.7576,  0.4215, -0.4827, -1.1198],
    [0.3056,  1.0386,  0.5206, -0.5006],
    [-2.9718,  1.7070, -0.4305, -2.2820],
    [0.3863,  0.9124, -0.8410,  1.2282] )
    pairs = zip(embeds_list, true)
    check_tensor_correctness(pairs)


def test_feature_extraction_d2_2():
    """ 0.5 point(s) """

    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    feat_extractor = SimpleFeatureExtractor()
    embedder = VanillaWordEmbeddingLookup(word_to_ix, TEST_EMBEDDING_DIM)
    combiner = DummyCombiner()
    embeds = embedder(test_sent)
    state = ParserState(test_sent, embeds, combiner)

    state.shift()
    state.shift()

    feats = feat_extractor.get_features(state)
    feats_list = make_list(feats)
    true = ([ -1.8661, 1.4146, -1.8781, -0.4674 ], [ -0.9596, 0.5489, -0.9901, -0.3826 ], [ 0.5237, 0.0004, -1.2039, 3.5283 ])
    pairs = zip(feats_list, true)
    check_tensor_correctness(pairs)


def test_action_chooser_d2_3():
    """ 1 point(s) """
    torch.manual_seed(1)
    act_chooser = ActionChooserNetwork(NUM_FEATURES * EMBEDDING_DIM)
    dummy_feats = [ ag.Variable(torch.randn(1, EMBEDDING_DIM)) for _ in xrange(NUM_FEATURES) ]
    out = act_chooser(dummy_feats)
    out_list = out.view(-1).data.tolist()
    true_out = [ -0.9352, -1.2393, -1.1460 ]
    check_tensor_correctness([(out_list, true_out)])


def test_combiner_d2_4():
    """ 1 point(s) """

    torch.manual_seed(1)
    combiner = MLPCombinerNetwork(6)
    head_feat = ag.Variable(torch.randn(1, 6))
    modifier_feat = ag.Variable(torch.randn(1, 6))
    combined = combiner(head_feat, modifier_feat)
    combined_list = combined.view(-1).data.tolist()
    true_out = [ -0.4897, 0.4484, -0.0591, 0.1778, 0.4223, -0.0940 ]
    check_tensor_correctness([(combined_list, true_out)])


# ===-------------------------------------------------------------------------------------------===
# Section 3 tests
# ===-------------------------------------------------------------------------------------------===

def test_parse_logic_d3_1():
    """ 0.5 point(s) """

    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    feat_extract = SimpleFeatureExtractor()
    word_embed = VanillaWordEmbeddingLookup(word_to_ix, TEST_EMBEDDING_DIM)
    act_chooser = ActionChooserNetwork(TEST_EMBEDDING_DIM * NUM_FEATURES)
    combiner = MLPCombinerNetwork(TEST_EMBEDDING_DIM)

    parser = TransitionParser(feat_extract, word_embed, act_chooser, combiner)
    output, dep_graph, actions_done = parser(test_sent[:-1], gold)
   
    assert len(output) == 15 # Made the right number of decisions

    # check one of the outputs
    checked_out = output[10].view(-1).data.tolist()
    true_out = [ -1.4737, -1.0875, -0.8350 ]
    check_tensor_correctness([(true_out, checked_out)])

    true_dep_graph = dependency_graph_from_oracle(test_sent, gold)
    assert true_dep_graph == dep_graph
    assert actions_done == [ 0, 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 0, 1, 1, 2 ]

def test_predict_after_train_d3_1():
    """ 1 point(s) """

    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)
    feat_extract = SimpleFeatureExtractor()
    word_embed = VanillaWordEmbeddingLookup(word_to_ix, TEST_EMBEDDING_DIM)
    act_chooser = ActionChooserNetwork(TEST_EMBEDDING_DIM * NUM_FEATURES)
    combiner = MLPCombinerNetwork(TEST_EMBEDDING_DIM)

    parser = TransitionParser(feat_extract, word_embed, act_chooser, combiner)

    # Train
    for i in xrange(75):
        train([ (test_sent[:-1], gold) ], parser, optim.SGD(parser.parameters(), lr=0.01), verbose=False)

    # predict
    pred = parser.predict(test_sent[:-1])
    gold_graph = dependency_graph_from_oracle(test_sent[:-1], gold)
    assert pred == gold_graph


def test_dev_d3_2():
    """ 0.5 point(s) / 0.25 point(s) (section dependent) """
    preds = open(D3_2_DEV_FILENAME)
    gold = open(DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines"
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.48
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)


# ===-------------------------------------------------------------------------------------------===
# Section 4 tests
# ===-------------------------------------------------------------------------------------------===

def test_bilstm_word_embeds_d4_1():
    """ 1 point(s) / 0.5 point(s) (section dependent) """

    global test_sent, word_to_ix, vocab
    torch.manual_seed(1)

    embedder = BiLSTMWordEmbeddingLookup(word_to_ix, TEST_EMBEDDING_DIM, TEST_EMBEDDING_DIM, 1, 0.0)
    embeds = embedder(test_sent)
    assert len(embeds) == len(test_sent)
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], ag.Variable)
    assert embeds[0].size() == (1, TEST_EMBEDDING_DIM)

    embeds_list = make_list(embeds)
    true = ( 
        [ .4916, -.0168, .1719, .6615 ],
        [ .3756, -.0610, .1851, .2604 ],
        [ -.2655, -.1289, .1009, -.0016 ],
        [ -.1070, -.3971, .2414, -.2588 ],
        [ -.1717, -.4475, .2739, -.0465 ], 
        [ 0.0684, -0.2586,  0.2123, -0.1832 ], 
        [ -0.0775, -0.4308,  0.1844, -0.1146 ], 
        [ 0.4366, -0.0507,  0.1018,  0.4015 ], 
        [ -0.1265, -0.2192,  0.0481,  0.1551 ])

    pairs = zip(embeds_list, true)
    check_tensor_correctness(pairs)


def test_pretrained_embeddings_d4_2():
    """ 0.5 point(s) """

    torch.manual_seed(1)
    word_to_ix = { "interest": 0, "rate": 1, "swap": 2 }
    pretrained = { "interest": [ -1.4, 2.6, 3.5 ], "swap": [ 1.6, 5.7, 3.2 ] }
    embedder = VanillaWordEmbeddingLookup(word_to_ix, 3)
    initialize_with_pretrained(pretrained, embedder)

    embeddings = embedder.word_embeddings.weight.data

    pairs = []
    true_rate_embed = [ -2.2820, 0.5237, 0.0004 ]

    pairs.append((embeddings[word_to_ix["interest"]].tolist(), pretrained["interest"]))
    pairs.append((embeddings[word_to_ix["rate"]].tolist(), true_rate_embed))
    pairs.append((embeddings[word_to_ix["swap"]].tolist(), pretrained["swap"]))
    check_tensor_correctness(pairs)


def test_lstm_combiner_d4_3():
    """ 1 point(s) """

    torch.manual_seed(1)
    combiner = LSTMCombinerNetwork(TEST_EMBEDDING_DIM, 1, 0.0)
    head_feat = ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM))
    modifier_feat = ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM))
    
    # Do the combination a few times to make sure they implemented the sequential
    # part right
    combined = combiner(head_feat, modifier_feat)
    combined = combiner(head_feat, modifier_feat)
    combined = combiner(head_feat, modifier_feat)

    combined_list = combined.view(-1).data.tolist()

    true_out = [ 0.0873, -0.1837, 0.1975, -0.1166 ]
    check_tensor_correctness([(combined_list, true_out)])


def test_dev_preds_d4_4():
    """ 0.5 point(s) / 0.25 point(s) (section dependent) """

    preds = open(D4_4_DEV_FILENAME)
    gold = open(DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines"
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.69
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)
