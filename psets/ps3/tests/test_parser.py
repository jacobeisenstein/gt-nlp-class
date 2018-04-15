from nose.tools import with_setup, eq_, assert_almost_equals, ok_
from gtnlplib.parsing import ParserState, TransitionParser, DepGraphEdge, train
from gtnlplib.utils import DummyCombiner, DummyActionChooser, DummyWordEmbedding, DummyFeatureExtractor, initialize_with_pretrained, build_suff_to_ix
from gtnlplib.data_tools import Dataset
from gtnlplib.constants import *
from gtnlplib.evaluation import compute_metric, fscore, dependency_graph_from_oracle
from gtnlplib.feat_extractors import SimpleFeatureExtractor
from gtnlplib.neural_net import FFActionChooser, FFCombiner, VanillaWordEmbedding, BiLSTMWordEmbedding, LSTMCombiner, LSTMActionChooser, SuffixAndWordEmbedding

import torch
import torch.autograd as ag
import torch.optim as optim

EMBEDDING_DIM = 64
TEST_EMBEDDING_DIM = 4
KERNEL_SIZE = 3
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
    gold = ["SHIFT", "ARC_L", "SHIFT", "ARC_L", "SHIFT", "SHIFT", "ARC_L",
            "ARC_R", "SHIFT", "SHIFT", "SHIFT", "ARC_L", "ARC_R", "ARC_R",
            "ARC_R", "SHIFT"]
    vocab = set(test_sent)
    vocab.add(NULL_STACK_TOK)

    word_to_ix = { word: i for i, word in enumerate(sorted(vocab)) }


# ===-------------------------------------------------------------------------------------------===
# Section 1 tests
# ===-------------------------------------------------------------------------------------------===

def test_get_arc_components_d1_1a():
    global test_sent
    state = make_dummy_parser_state(test_sent)
    #just to get stuff on the stack
    state.shift()
    state.shift()
    state.shift()
    state.shift()
    head, mod = state._get_arc_components(Actions.ARC_L)
    eq_(head.headword, "dog")
    eq_(head.headword_pos, 4)
    eq_(head.embedding, "dog-EMBEDDING")
    eq_(mod.headword, "the")
    eq_(mod.headword_pos, 3)
    eq_(mod.embedding, "the-EMBEDDING")

    head, mod = state._get_arc_components(Actions.ARC_R)
    eq_(head.headword, "saw")
    eq_(head.headword_pos, 2)
    eq_(head.embedding, "saw-EMBEDDING")
    eq_(mod.headword, "with")
    eq_(mod.headword_pos, 5)
    eq_(mod.embedding, "with-EMBEDDING")

    head, mod = state._get_arc_components(Actions.ARC_L)
    eq_(head.headword, "the")
    eq_(head.headword_pos, 6)
    eq_(head.embedding, "the-EMBEDDING")
    eq_(mod.headword, "man")
    eq_(mod.headword_pos, 1)
    eq_(mod.embedding, "man-EMBEDDING")
    

def test_create_arc_d1_1b():
    global test_sent
    state = make_dummy_parser_state(test_sent)
    state.shift()
    left_arc = state.arc_left()
    
    # head word should be "man"
    eq_(left_arc[0][0], "man")
    eq_(left_arc[0][1], 1)
    # dependent should be "The"
    eq_(left_arc[1][0], "The")
    eq_(left_arc[1][1], 0)

    # check the top of the input buffer to make sure combination was right
    eq_(state.input_buffer[0].headword, "man")
    eq_(state.input_buffer[0].headword_pos, 1)
    eq_(state.input_buffer[0].embedding, "man-EMBEDDING")

    state.shift()
    state.shift()
    right_arc = state.arc_right()

    # Head word should be "saw"
    eq_(right_arc[0][0], "saw")
    eq_(right_arc[0][1], 2)
    # dependent should be "the"
    eq_(right_arc[1][0], "the")
    eq_(right_arc[1][1], 3)

    eq_(state.input_buffer[0].headword, "saw")
    eq_(state.input_buffer[0].headword_pos, 2)
    eq_(state.input_buffer[0].embedding, "saw-EMBEDDING")


def test_stack_terminating_cond_d1_2():
    global test_sent
    state = make_dummy_parser_state(test_sent[:-1])
    
    assert not state.done_parsing()

    state.shift()
    state.arc_left()
    assert not state.done_parsing()

    state.shift()
    state.arc_left()
    state.shift()
    state.shift()
    state.arc_left()
    assert not state.done_parsing()

    state.arc_right()
    state.shift()
    state.shift()
    state.shift()
    assert not state.done_parsing()

    state.arc_left()
    state.arc_right()
    assert not state.done_parsing()

    state.arc_right()
    state.arc_right()
    assert not state.done_parsing()

    state.shift()
    assert state.done_parsing()

def test_validate_action_d1_3():
    global test_sent
    state = make_dummy_parser_state(test_sent[:-1])

    #test don't arc left to ROOT
    act_to_do = Actions.ARC_L
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.SHIFT)
    state.shift()
    state.arc_left()
    state.shift()
    state.arc_left()
    state.shift()
    state.shift()
    state.arc_left()
    state.arc_right()
    state.shift()
    state.shift()
    state.shift()

    #test don't shift when input buffer too short
    act_to_do = Actions.SHIFT
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.ARC_R)

    state.arc_left()
    state.arc_right()
    state.arc_right()
    #enforce arc-right here
    act_to_do = Actions.SHIFT
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.ARC_R)

    act_to_do = Actions.ARC_L
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.ARC_R)

    state.arc_right()
    
    #enforce shift here
    act_to_do = Actions.ARC_L
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.SHIFT)

    act_to_do = Actions.ARC_R
    valid_action = state._validate_action(act_to_do)
    eq_(valid_action, Actions.SHIFT)

    state.shift()


# ////////////////////////////////////////////////////////////////////////////////////////////

# ===-------------------------------------------------------------------------------------------===
# Section 2 tests
# ===-------------------------------------------------------------------------------------------===

def test_word_embed_lookup_d2_1():
    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    embedder = VanillaWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM)
    embeds = embedder(test_sent)
    assert len(embeds) == len(test_sent)
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], ag.Variable)
    assert embeds[0].size() == (1, TEST_EMBEDDING_DIM)

    embeds_list = make_list(embeds)

    true = ([-1.02760863, -0.56305277, -0.89229053, -0.05825018],
    [-0.42119515, -0.51069999, -1.57266521, -0.12324776],
    [ 3.5869894 , -1.83129013,  1.59870028, -1.27700698],
    [ 0.41074166, -0.98800713, -0.90807337,  0.54227364],
    [-0.19550958, -0.96563596,  0.42241532,  0.267317  ],
    [ 0.11025489, -2.2590096 ,  0.60669959, -0.13830966],
    [ 0.41074166, -0.98800713, -0.90807337,  0.54227364],
    [ 0.32550153, -0.47914493,  1.37900829,  2.5285573 ],
    [ 0.66135216,  0.26692411,  0.06167726,  0.62131733])
    pairs = zip(embeds_list, true)
    check_tensor_correctness(pairs)


def test_feature_extraction_d2_2():
    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    feat_extractor = SimpleFeatureExtractor()
    embedder = VanillaWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM)
    combiner = DummyCombiner()
    embeds = embedder(test_sent)
    state = ParserState(test_sent, embeds, combiner)

    state.shift()

    feats = feat_extractor.get_features(state)
    feats_list = make_list(feats)
    true = ([-1.0276086330413818, -0.563052773475647, -0.8922905325889587, -0.05825017765164375],
            [-0.4211951494216919, -0.510699987411499, -1.5726652145385742, -0.12324775755405426],
            [3.586989402770996, -1.8312901258468628, 1.5987002849578857, -1.277006983757019])
    pairs = zip(feats_list, true)
    check_tensor_correctness(pairs)


def test_action_chooser_d2_3():
    torch.manual_seed(1)
    act_chooser = FFActionChooser(NUM_FEATURES * TEST_EMBEDDING_DIM)
    dummy_feats = [ ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM)) for _ in range(NUM_FEATURES) ]
    out = act_chooser(dummy_feats)
    out_list = out.view(-1).data.tolist()
    true_out = [-1.24434566, -0.83229464, -1.28438509]
    check_tensor_correctness([(out_list, true_out)])


def test_combiner_d2_4():
    torch.manual_seed(1)
    combiner = FFCombiner(6)
    head_feat = ag.Variable(torch.randn(1, 6))
    modifier_feat = ag.Variable(torch.randn(1, 6))
    combined = combiner(head_feat, modifier_feat)
    combined_list = combined.view(-1).data.tolist()
    true_out = [-0.1194517, 0.31343767, 0.29966655, 0.26423377, -0.09193783, -0.56594414]
    check_tensor_correctness([(combined_list, true_out)])


# ===-------------------------------------------------------------------------------------------===
# Section 3 tests
# ===-------------------------------------------------------------------------------------------===

def test_parse_logic_d3_1():
    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)

    feat_extract = SimpleFeatureExtractor()
    word_embed = VanillaWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM)
    act_chooser = FFActionChooser(TEST_EMBEDDING_DIM * NUM_FEATURES)
    combiner = FFCombiner(TEST_EMBEDDING_DIM)

    parser = TransitionParser(feat_extract, word_embed, act_chooser, combiner)
    output, dep_graph, actions_done = parser(test_sent[:-1], gold)
   
    assert len(output) == 16 # Made the right number of decisions

    # check one of the outputs
    checked_out = output[9].view(-1).data.tolist()
    true_out = [-1.2444578409194946, -1.3128550052642822, -0.8145193457603455]
    check_tensor_correctness([(true_out, checked_out)])

    true_dep_graph = dependency_graph_from_oracle(test_sent, gold)
    assert true_dep_graph == dep_graph
    assert actions_done == [ 0, 1, 0, 1, 0, 0, 1, 2, 0, 0, 0, 1, 2, 2, 2, 0]

def test_predict_after_train_d3_1():
    global test_sent, gold, word_to_ix, vocab
    torch.manual_seed(1)
    feat_extract = SimpleFeatureExtractor()
    word_embed = VanillaWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM)
    act_chooser = FFActionChooser(TEST_EMBEDDING_DIM * NUM_FEATURES)
    combiner = FFCombiner(TEST_EMBEDDING_DIM)

    parser = TransitionParser(feat_extract, word_embed, act_chooser, combiner)

    # Train
    for i in range(75):
        train([ (test_sent[:-1], gold) ], parser, optim.SGD(parser.parameters(), lr=0.01), verbose=False)

    # predict
    pred = parser.predict(test_sent[:-1])
    gold_graph = dependency_graph_from_oracle(test_sent[:-1], gold)
    assert pred == gold_graph


def test_dev_d3_2_english():
    preds = open(EN_D3_2_DEV_FILENAME)
    gold = open(EN_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.42
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)

def test_dev_d3_3_norwegian():
    preds = open(NR_D3_3_DEV_FILENAME)
    gold = open(NR_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.42
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)



# ===-------------------------------------------------------------------------------------------===
# Section 4 tests
# ===-------------------------------------------------------------------------------------------===

def test_bilstm_word_embeds_d4_1():
    global test_sent, word_to_ix, vocab
    torch.manual_seed(1)

    embedder = BiLSTMWordEmbedding(word_to_ix, TEST_EMBEDDING_DIM, TEST_EMBEDDING_DIM, 1, 0.0)
    embeds = embedder(test_sent)
    assert len(embeds) == len(test_sent)
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], ag.Variable)
    assert embeds[0].size() == (1, TEST_EMBEDDING_DIM)

    embeds_list = make_list(embeds)
    true = (
            [0.09079286456108093, 0.06577987223863602, 0.26242679357528687, -0.004267544485628605],
            [0.16868481040000916, 0.2032647728919983, 0.23663431406021118, -0.11785736680030823],
            [0.35757705569267273, 0.3805052936077118, -0.006295515224337578, 0.0010524550452828407],
            [0.26692214608192444, 0.3241712749004364, 0.13473030924797058, -0.026079852133989334],
            [0.23157459497451782, 0.13698695600032806, 0.04000323265790939, 0.1107199415564537],
            [0.22783540189266205, -0.02211562544107437, 0.06239837780594826, 0.08553065359592438],
            [0.24633683264255524, 0.09283821284770966, 0.0987505242228508, -0.07646450400352478],
            [0.05530695244669914, -0.4060348570346832, -0.060150448232889175, -0.003920700401067734],
            [0.2099054455757141, -0.304738312959671, -0.01663055270910263, -0.05987118184566498]
            )

    pairs = zip(embeds_list, true)
    check_tensor_correctness(pairs)

def test_suff_word_embeds_d4_2():
    global test_sent, word_to_ix, vocab
    torch.manual_seed(1)
    test_suff_to_ix = build_suff_to_ix(word_to_ix)

    suff_word_embedder = SuffixAndWordEmbedding(word_to_ix, test_suff_to_ix, TEST_EMBEDDING_DIM)
    embeds = suff_word_embedder(test_sent)
    assert len(embeds) == len(test_sent)
    assert isinstance(embeds, list)
    assert isinstance(embeds[0], ag.Variable)
    assert embeds[0].size() == (1, TEST_EMBEDDING_DIM)

    embeds_list = make_list(embeds)
    true = ([-0.45190597, -0.16613023,  1.37900829,  2.5285573 ],
            [-1.02760863, -0.56305277,  1.59870028, -1.27700698],
            [-0.89229053, -0.05825018,  0.32550153, -0.47914493],
            [ 0.42241532,  0.267317  ,  1.37900829,  2.5285573 ],
            [-1.5227685 ,  0.38168392,  0.41074166, -0.98800713],
            [-0.42119515, -0.51069999,  0.11025489, -2.2590096 ],
            [ 0.42241532,  0.267317  ,  1.37900829,  2.5285573 ],
            [-0.19550958, -0.96563596, -0.90807337,  0.54227364],
            [ 0.66135216,  0.26692411,  3.5869894 , -1.83129013])

    pairs = zip(embeds_list, true)
    check_tensor_correctness(pairs)

def test_pretrained_embeddings_d4_3():
    torch.manual_seed(1)
    word_to_ix = { "interest": 0, "rate": 1, "swap": 2 }
    pretrained = { "interest": [ -1.4, 2.6, 3.5 ], "swap": [ 1.6, 5.7, 3.2 ] }
    embedder = VanillaWordEmbedding(word_to_ix, 3)
    initialize_with_pretrained(pretrained, embedder)

    embeddings = embedder.word_embeddings.weight.data

    pairs = []
    true_rate_embed = [0.62131733, -0.45190597, -0.16613023]

    pairs.append((embeddings[word_to_ix["interest"]].tolist(), pretrained["interest"]))
    pairs.append((embeddings[word_to_ix["rate"]].tolist(), true_rate_embed))
    pairs.append((embeddings[word_to_ix["swap"]].tolist(), pretrained["swap"]))
    check_tensor_correctness(pairs)


def test_lstm_combiner_d4_4():
    torch.manual_seed(1)
    combiner = LSTMCombiner(TEST_EMBEDDING_DIM, 1, 0.0)
    head_feat = ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM))
    modifier_feat = ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM))
    
    # Do the combination a few times to make sure they implemented the sequential
    # part right
    combined = combiner(head_feat, modifier_feat)
    combined = combiner(head_feat, modifier_feat)
    combined = combiner(head_feat, modifier_feat)

    combined_list = combined.view(-1).data.tolist()

    true_out = [0.059396881610155106,
               -0.3381599187850952,
                0.26394787430763245,
               -0.11590906977653503]
    check_tensor_correctness([(combined_list, true_out)])

def test_lstm_action_chooser_d4_5():
    torch.manual_seed(1)
    action_chooser = LSTMActionChooser(TEST_EMBEDDING_DIM * NUM_FEATURES, 1, 0.0)
    feats = [ag.Variable(torch.randn(1, TEST_EMBEDDING_DIM)) for _ in range(3)]
    
    # Run the action chooser a few times to make sure they implemented the sequential
    # part right
    output = action_chooser(feats)
    output = action_chooser(feats)
    output = action_chooser(feats)

    output_list = output.view(-1).data.tolist()

    true_out = [-1.0001587867736816, -1.1845550537109375, -1.119948148727417]
    check_tensor_correctness([(output_list, true_out)])


def test_dev_preds_d4_6_english():
    preds = open(EN_D4_6_DEV_FILENAME)
    gold = open(EN_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.61
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)

def test_dev_preds_d4_7_norwegian():
    preds = open(NR_D4_7_DEV_FILENAME)
    gold = open(NR_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.53
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)

def test_dev_preds_bakeoff_d5_1_english():
    preds = open(EN_BAKEOFF_FILENAME)
    gold = open(EN_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.76 
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)

def test_dev_preds_bakeoff_d5_2_norwegian():
    preds = open(NR_BAKEOFF_FILENAME)
    gold = open(NR_DEV_GOLD)

    correct = 0
    total = 0
    for p, g in zip(preds, gold):
        if p.strip() == "":
            assert g.strip() == "", "Mismatched blank lines. Check your parser's behavior when gold actions are not provided."
            continue
        p_data = p.split("\t")
        g_data = g.split("\t")
        if p_data[3] == g_data[3]:
            correct += 1
        total += 1
    acc = float(correct) / total
    exp = 0.71
    assert acc > exp, "ERROR: Expected {} Got {}".format(exp, acc)
