import os
from nose.tools import with_setup, assert_almost_equals, assert_greater, assert_equal, assert_true, assert_false
from collections import defaultdict

from nltk.tag import pos_tag

from gtnlplib import coref_rules, coref, coref_features, coref_learning, neural_net, bcm_evaluate, utils

tr_dir = os.path.join('data','wsj','train')
dev_dir = os.path.join('data','wsj','dev')
te_dir = os.path.join('data','wsj','test')

global markables_dev
global markables_tr
global all_markables, all_words
global exact_matcher

def setup_module():
    global markables_dev
    global markables_tr
    global all_markables, all_markables_te, all_markables_dev
    global exact_matcher
    
    markables_tr, _ = coref.read_data('06_wsj_0051.sty', basedir=tr_dir, tagger=pos_tag)
    markables_dev, _ = coref.read_data('25_wsj_0015.sty', basedir=dev_dir, tagger=pos_tag)
    all_markables, _ = coref.read_dataset(tr_dir, tagger=pos_tag)
    all_markables_dev, _ = coref.read_dataset(dev_dir, tagger=pos_tag)
    all_markables_te, _ = coref.read_dataset(te_dir, tagger=pos_tag)

    exact_matcher = coref_rules.make_resolver(coref_rules.exact_match)
    
# deliverable 1.1
def test_get_markables_d1_1():
    global markables_dev, markables_tr
    
    mark83 = coref.get_markables_for_entity(markables_tr,'set_3083')
    assert_equal(sorted(mark83), ["Fujitsu Ltd. 's top executive",
                                  'Fujitsu President Takuma Yamamoto',
                                  'Mr. Yamamoto',
                                  'Mr. Yamamoto',
                                  'he',
                                  'he',
                                  'he',
                                  'his',
                                  'his'])
                                   
    mark78 = coref.get_markables_for_entity(markables_dev,'set_3678')
    assert_equal(sorted(mark78), ['Judge Curry',
                                  'Judge Curry',
                                  'Judge Curry',
                                  'Judge Curry',
                                  'Judge Curry',
                                  'Judge Curry',
                                  'State court Judge Richard Curry',
                                  'he',
                                  'his',
                                  'his'])
    
# deliverable 1.2
def test_get_antecedents_d1_2():
    global markables_dev, markables_tr
    
    assert_equal(coref.get_distances(markables_tr, 'his'), [3, 2])
    
    dv_distances = coref.get_distances(markables_dev, 'the plant')
    assert_equal(dv_distances, [9, 14, 9])

###############################
    
# deliverable 2.1a
def test_recall_d2_1():
    global markables_tr, exact_matcher
    f, r, p = coref.evaluate_f(exact_matcher, markables_tr)
    assert_almost_equals(r, 0.409639, places=4)

# deliverable 2.1b
def test_precision_d2_1():
    global markables_tr, exact_matcher
    f, r, p = coref.evaluate_f(exact_matcher, markables_tr)
    assert_almost_equals(p, 0.723404, places=4)

# deliverable 2.1c
def test_fmeasure_d2_1():
    global markables_tr, exact_matcher
    f, r, p = coref.evaluate_f(exact_matcher, markables_tr)
    assert_almost_equals(f, 0.523077,places=4)

# deliverable 2.2a
def test_singleton_matcher_d2_2():
    ants = coref_rules.make_resolver(coref_rules.singleton_matcher)(markables_tr)
    assert_equal(len(coref.markables_to_entities(markables_tr, ants)[1]), len(markables_tr))

# deliverable 2.2b
def test_full_cluster_matcher_d2_2():
    ants = coref_rules.make_resolver(coref_rules.full_cluster_matcher)(markables_tr)
    assert_equal(len(coref.markables_to_entities(markables_tr, ants)[1]), 1)

# deliverable 2.3a
def test_match_nopro_d2_3():
    assert_true(coref_rules.exact_match(markables_tr[35], markables_tr[135]))
    assert_false(coref_rules.exact_match_no_pronouns(markables_tr[35], markables_tr[135]))
    assert_true(coref_rules.exact_match(markables_tr[169], markables_tr[175]))
    assert_false(coref_rules.exact_match_no_pronouns(markables_tr[169], markables_tr[175]))

# deliverable 2.3b
def test_match_nopro_f1_d2_3():
    global all_markables
    f, r, p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.exact_match_no_pronouns),
        all_markables)
    assert_almost_equals(r, 0.3028, places=4)
    assert_almost_equals(p, 0.9158, places=4)

# deliverable 2.4a
def test_match_last_tok_d2_4():
    global markables_tr
    assert_true(coref_rules.match_last_token(markables_tr[20], markables_tr[43]))
    assert_true(coref_rules.match_last_token(markables_tr[189], markables_tr[190]))
    assert_false(coref_rules.match_last_token(markables_tr[0], markables_tr[20]))
    assert_true(coref_rules.match_last_token(markables_tr[38], markables_tr[39]))

# deliverable 2.4b
def test_match_last_tok_f1_d2_4():
    global all_markables
    f, r, p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_last_token),
        all_markables)
    assert_greater(f, .399)
    assert_greater(r, .438)
    assert_greater(p, .366)

# deliverable 2.5
def test_match_no_overlap_f1_d2_5():
    global all_markables
    f, r, p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_last_token_no_overlap),
        all_markables)
    assert_greater(f, .491)
    assert_greater(r, .496)
    assert_greater(p, .485)
    
# deliverable 2.6
def test_match_content_f1_d2_6():
    global all_markables
    f, r, p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_on_content),
        all_markables)
    assert_greater(f, .556)
    assert_greater(r, .437)
    assert_greater(p, .764)

# deliverable 2.7a
def test_dev_acc_f1_d2_7():
    f, r, p = coref.eval_predictions('predictions/rules-dev.preds', all_markables_dev);
    assert_greater(f, .546)
    assert_greater(r, .395)
    assert_greater(p, .880)

# deliverable 2.7b
# students can't run this
def test_test_acc_f1_d2_7():
    f, r, p = coref.eval_predictions('predictions/rules-test.preds', all_markables_te);
    assert_greater(f, .499)
    assert_greater(r, .381)
    assert_greater(p, .723)

########################3

# deliverable 3.1
def test_minimal_features_d3_1():
    global markables_tr
    f = coref_features.minimal_features(markables_tr, 6, 17)
    assert_equal(len(f), 0)
    f = coref_features.minimal_features(markables_tr, 0, 1)
    assert(f['crossover'] == 1 and len(f) == 1)
    f = coref_features.minimal_features(markables_tr, 1, 1)
    assert(f['new-entity'] == 1 and len(f) == 1)
    f = coref_features.minimal_features(markables_tr, 88, 127)
    assert(len(f) == 1 and f['last-token-match'] == 1)
    f = coref_features.minimal_features(markables_tr, 28, 153)
    assert(len(f) == 3 and f['exact-match'] == 1 and f['last-token-match'] == 1 and f['content-match'] == 1)

# deliverable 3.5
def test_distance_features_d3_5():
    global markables_tr

    f = coref_features.distance_features(markables_tr, 0, 0)
    assert len(f) == 0

    f = coref_features.distance_features(markables_tr, 0, 3)
    assert f['token-distance-8'] == 1
    assert f['mention-distance-3'] == 1
    assert len(f) == 2
    
    f = coref_features.distance_features(markables_tr, 0, 4)
    assert f['token-distance-10'] == 1
    assert f['mention-distance-4'] == 1
    assert len(f) == 2
    
    f = coref_features.distance_features(markables_tr, 4, 14)
    assert f['token-distance-10'] == 1
    assert f['mention-distance-5'] == 1
    assert len(f) == 2

    f = coref_features.distance_features(markables_tr, 14, 14)
    assert len(f) == 0

# deliverable 3.6
def test_feature_union_d3_6():
    global markables_tr
    joint_feats1 = coref_features.make_feature_union([coref_features.minimal_features,
                                                      coref_features.distance_features])
    f = joint_feats1(markables_tr, 1, 3)
    print(f)
    assert len(f) == 2 and f['token-distance-5'] == 1
    f = joint_feats1(markables_tr, 23, 47)
    assert len(f) == 3 and f['mention-distance-5'] == 1 and f['last-token-match'] == 1
    f = joint_feats1(markables_tr, 10, 10)
    assert len(f) == 1 and f['new-entity'] == 1

### bakeoff tests

