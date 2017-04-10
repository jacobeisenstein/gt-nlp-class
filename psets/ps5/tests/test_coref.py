import os
from nose.tools import with_setup, assert_almost_equals, assert_greater, assert_equal, assert_true, assert_false
from collections import defaultdict

from nltk.tag import pos_tag

from gtnlplib import coref_rules, coref, coref_features, coref_learning

# still need these?
data_path = os.path.join('data','dev','*')
te_data_path = os.path.join('data','te','*')

tr_dir = os.path.join('data','tr')
te_dir = os.path.join('data','te')
dev_dir = os.path.join('data','dev')

global markables_dev
global markables_tr
global all_markables,all_words
global exact_matcher

def setup_module():
    global markables_dev
    global markables_tr
    global all_markables,all_markables_te,all_markables_dev
    global exact_matcher
    
    markables_dev,_ = coref.read_data('Siege of Chaves',basedir=dev_dir)
    markables_tr,_ = coref.read_data('Johnston Atoll',basedir=tr_dir)
    all_markables,_ = coref.read_dataset(tr_dir,tagger=pos_tag)
    all_markables_te,_ = coref.read_dataset(te_dir)
    all_markables_dev,_ = coref.read_dataset(dev_dir)

    exact_matcher = coref_rules.make_resolver(coref_rules.exact_match)
    
# deliverable 1.1 (0.5 pts)
def test_get_markables_d1_1():
    global markables_dev, markables_tr
    mark104 = coref.get_markables_for_entity(markables_dev,'set_104')
    assert_equal(sorted(mark104), ['Spain','Spain','the Spanish'])

    mark100 = coref.get_markables_for_entity(markables_tr,'set_100')
    assert_equal(sorted(mark100), ['Johnston and Sand Island',
                                   'Johnston and Sand islands',
                                   'The islands',
                                   'the area',
                                   'the islands',
                                   'them'])
    
# deliverable 1.2 (0.5 pts)
def test_get_antecedents_d1_2():
    global markables_dev, markables_tr
    distances = coref.get_distances(markables_dev,'the Portuguese')
    assert_equal(distances,[102,6,1])
    assert_equal(coref.get_distances(markables_tr,'the area'),
                 [4])

###############################
    
# deliverable 2.1a (0.4 pt)
def test_recall_d2_1():
    global markables_tr, exact_matcher
    f,r,p = coref.evaluate(exact_matcher,markables_tr)
    assert_almost_equals(r,.47305,places=4)

# deliverable 2.1b (0.4 pt)
def test_precision_d2_1():
    global markables_tr, exact_matcher
    f,r,p = coref.evaluate(exact_matcher,markables_tr)
    assert_almost_equals(p,.89773,places=4)

# deliverable 2.1c (0.2 pt)
def test_fmeasure_d2_1():
    global markables_tr, exact_matcher
    f,r,p = coref.evaluate(exact_matcher,markables_tr)
    assert_almost_equals(f,.61961,places=4)

# deliverable 2.2 (0.5/0.25 pts)
def test_match_nopro_d2_2():
    assert_true(coref_rules.exact_match(markables_tr[17],markables_tr[71]))
    assert_false(coref_rules.exact_match_no_pronouns(markables_tr[17],markables_tr[71]))
    assert_true(coref_rules.exact_match_no_pronouns(markables_tr[21],markables_tr[28]))
    assert_false(coref_rules.exact_match_no_pronouns(markables_tr[21],markables_tr[29]))

def test_match_nopro_f1_d2_2():
    global all_markables
    f,r,p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.exact_match_no_pronouns),
        all_markables)
    assert_greater(f,.64)
    assert_greater(r,.48)
    assert_greater(p,.94)

# deliverable 2.3 (0.5/0.25 pts)
def test_match_last_tok_d2_3():
    global markables_tr
    assert_true(coref_rules.match_last_token(markables_tr[0],markables_tr[0]))
    assert_true(coref_rules.match_last_token(markables_tr[0],markables_tr[3]))
    assert_false(coref_rules.match_last_token(markables_tr[0],markables_tr[2]))
    assert_true(coref_rules.match_last_token(markables_tr[3],markables_tr[10]))

def test_match_last_tok_f1_d2_3():
    global all_markables
    f,r,p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_last_token),
        all_markables)
    assert_greater(f,.64)
    assert_greater(r,.59)
    assert_greater(p,.71)

# deliverable 2.4 (0.5/0.25 pts)
def test_match_no_overlap_f1_d2_4():
    global all_markables
    f,r,p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_last_token_no_overlap),
        all_markables)
    assert_greater(f,.67)
    assert_greater(r,.61)
    assert_greater(p,.74)
    
# deliverable 2.5 (0.5/0.25 pts)
def test_match_content_f1_d2_5():
    global all_markables
    f,r,p = coref.eval_on_dataset(
        coref_rules.make_resolver(coref_rules.match_on_content),
        all_markables)
    assert_greater(f,.68)
    assert_greater(r,.57)
    assert_greater(p,.85)

# deliverable 2.6a (0.25 points)
def test_dev_acc_f1_d2_6():
    f,r,p = coref.eval_predictions('predictions/rules-dev.preds',all_markables_dev);
    assert_greater(f,.68)
    assert_greater(r,.58)
    assert_greater(p,.81)

# deliverable 2.6b (0.25 points)
def test_test_acc_f1_d2_6():
    f,r,p = coref.eval_predictions('predictions/rules-test.preds',all_markables_te);
    assert_greater(f,.675)
    assert_greater(r,.57)
    assert_greater(p,.82)

########################3

# deliverable 3.1 (1 point)
def test_minimal_features_d3_1():
    global all_markables
    f=coref_features.minimal_features(all_markables[14],0,1)
    assert(len(f)==0)
    f=coref_features.minimal_features(all_markables[14],1,1)
    assert(f['new-entity']==1 and len(f)==1)
    f = coref_features.minimal_features(all_markables[14],0,3)
    assert(len(f)==1 and f['last-token-match']==1)
    f = coref_features.minimal_features(all_markables[14],6,7)
    assert(len(f)==1 and f['crossover']==1)
    f = coref_features.minimal_features(all_markables[14],3,14)
    assert(len(f)==3 and f['exact-match']==1 and f['last-token-match']==1 and f['content-match']==1)

# deliverable 3.2 (1 point)
def test_mention_rank_d3_2():
    global all_markables
    hand_weights = defaultdict(float,
                               {'new-entity':0.5,
                                'last-token-match':0.6,
                                'content-match':0.7,
                                'exact-match':1.}
    )
    assert(coref_learning.mention_rank(all_markables[12],
                                       1,
                                       coref_features.minimal_features,
                                       hand_weights)
           ==1)
    assert(coref_learning.mention_rank(all_markables[12],
                                       7,
                                       coref_features.minimal_features,
                                       hand_weights)
           ==0)

# deliverable 3.3 (0.5 points)
def test_compute_instance_update_d3_3():
    global all_markables
    hand_weights = defaultdict(float,
                               {'new-entity':0.5,
                                'last-token-match':0.6,
                                'content-match':0.7,
                                'exact-match':1.}
    )
    update=coref_learning.compute_instance_update(
        all_markables[14],14,3,coref_features.minimal_features,hand_weights)
    assert(len(update)==0)
    update=coref_learning.compute_instance_update(
        all_markables[14],14,10,coref_features.minimal_features,hand_weights)
    assert(len(update)==0)
    update=coref_learning.compute_instance_update(
        all_markables[14],14,12,coref_features.minimal_features,hand_weights)
    assert(len(update)==3 and\
           update['exact-match']==-1 and\
           update['last-token-match']==-1 and\
           update['content-match']==-1)
    update=coref_learning.compute_instance_update(
        all_markables[14],14,1,coref_features.minimal_features,hand_weights)
    assert(len(update)==3 and\
           update['exact-match']==-1 and\
           update['last-token-match']==-1 and\
           update['content-match']==-1)

# deliverable 3.4 (1 point)
def test_average_perceptron_d3_4a():
    global all_markables
    theta_simple = coref_learning.train_avg_perceptron([all_markables[3][:10]],coref_features.minimal_features,N_its=2)
    assert(theta_simple[-1]['content-match']==0.6)
    assert(theta_simple[-1]['crossover']==0.0)
    assert(theta_simple[-1]['new-entity']==0.2)
    assert(theta_simple[-1]['exact-match']==0.6)

def test_average_perceptron_d3_4b():
    f,r,p = coref.eval_predictions('predictions/minimal-dev.preds',all_markables_dev);
    assert(f>.66)
    assert(r>.57)
    assert(p>.78)


# deliverable 3.5
def test_distance_features_d3_5():
    global all_markables

    f = coref_features.distance_features(all_markables[9],0,0)
    assert len(f)==0

    f = coref_features.distance_features(all_markables[9],0,1)
    assert f['token-distance-8'] == 1
    assert f['mention-distance-1'] == 1
    assert len(f) == 2
    
    f = coref_features.distance_features(all_markables[9],0,2)
    assert f['token-distance-10'] == 1
    assert f['mention-distance-2'] == 1
    assert len(f) == 2
    
    f = coref_features.distance_features(all_markables[9],1,3)
    assert f['token-distance-6'] == 1
    assert f['mention-distance-2'] == 1
    assert len(f) == 2

    f = coref_features.distance_features(all_markables[9],11,11)
    assert len(f) == 0

    f = coref_features.distance_features(all_markables[9],0,30)
    assert len(f) == 2
    assert f['token-distance-10'] == 1
    assert f['mention-distance-10'] == 1

# 0.25 points
def test_feature_union_d3_6():
    global all_markables
    joint_feats1 = coref_features.make_feature_union([coref_features.minimal_features,
                                                      coref_features.distance_features])
    f = joint_feats1(all_markables[12],1,3)
    assert len(f) == 2 and f['token-distance-6']==1
    f = joint_feats1(all_markables[12],0,7)
    assert len(f) == 3 and f['mention-distance-7']==1 and f['last-token-match']==1
    f = joint_feats1(all_markables[12],10,10)
    assert len(f) == 1 and f['new-entity']==1

# 0.25 points
def test_feature_product_d3_7():
    prod_feats1 = coref_features.make_feature_cross_product(coref_features.minimal_features,
                                                            coref_features.distance_features)
    f = prod_feats1(all_markables[14],3,14)
    assert len(f) == 6\
        and f['content-match-mention-distance-10'] == 1\
        and f['exact-match-mention-distance-10']==1\
        and f['content-match-token-distance-10']==1\
        and f['last-token-match-mention-distance-10']==1\
        and f['last-token-match-token-distance-10']==1\
        and f['exact-match-token-distance-10']==1

    f = prod_feats1(all_markables[12],0,7)
    assert len(f) == 2\
        and f['last-token-match-mention-distance-7']==1\
        and f['last-token-match-token-distance-10']==1


# +0.25 points
def test_bakeoff_dev_d3_9a():
    global all_markables_dev
    f,r,p = coref.eval_predictions('predictions/bakeoff-dev.preds',all_markables_dev)
    assert f > .71

# +0.25 points
def test_bakeoff_dev_d3_9b():
    global all_markables_dev
    f,r,p = coref.eval_predictions('predictions/bakeoff-dev.preds',all_markables_dev)
    assert f > .72

# +0.25 points
def test_bakeoff_dev_d3_9c():
    global all_markables_dev
    f,r,p = coref.eval_predictions('predictions/bakeoff-dev.preds',all_markables_dev)
    assert f > .73
    
# +0.25
def test_bakeoff_test_d3_9():
    global all_markables_te
    # students can't run this
    f,r,p = coref.eval_predictions('predictions/bakeoff-te.preds',all_markables_te)
    assert f > .7

# +0.5 if you beat my score
def test_bakeoff_vs_prof_d3_9():
    global all_markables_te
    # students can't run this
    f,r,p = coref.eval_predictions('predictions/bakeoff-te.preds',all_markables_te)
    assert f > .723



