from nose.tools import with_setup, eq_, assert_almost_equals, ok_, assert_greater
from collections import defaultdict

from gtnlplib.constants import * #This is bad and I'm sorry.
from gtnlplib import features, preproc, tagger_base, structure_perceptron, viterbi
from gtnlplib import scorer


def setup():
    global all_tags, theta_clf_hand, toy_data
    all_tags = set()
    for i,(words, tags) in enumerate(preproc.conll_seq_generator(TRAIN_FILE,max_insts=100000)):
        for tag in tags:
            all_tags.add(tag)

    theta_clf_hand = defaultdict(float,
                                 {('NOUN',OFFSET):0.1,
                                  ('PRON',CURR_WORD_FEAT,'They'):1,
                                  ('PRON',CURR_WORD_FEAT,'can'):-1,
                                  ('NOUN',CURR_WORD_FEAT,'fish'):1,
                                  ('VERB',CURR_WORD_FEAT,'fish'):0.5})

    toy_data = [('They can fish'.split(),['PRON','AUX','VERB']),
                ('the old man the boat'.split(),['DET','NOUN','VERB','DET','NOUN'])]
    

# 0.7 points    
def test_word_feats_d1_1():
    fv1 = features.word_feats(['The','old','man','the','boat'],'NOUN','ADJ',2)
    eq_(len(fv1),2)
    eq_(fv1['NOUN',OFFSET],1)
    eq_(fv1['NOUN',CURR_WORD_FEAT,'man'],1)

    fv2 = features.word_feats(['The','old','man','the','boat'],'VERB','NOUN',3)
    eq_(len(fv2),2)
    eq_(fv2['VERB',OFFSET],1)
    eq_(fv2['VERB',CURR_WORD_FEAT,'the'],1)

    fv3 = features.word_feats(['The','old','man','the','boat'],'NOUN','ADJ',5)
    eq_(len(fv3),1)
    eq_(fv3,{('NOUN',OFFSET):1})

# 0.7 points
def test_classifier_tagger_d1_2():
    global all_tags, theta_clf_hand

    w = 'They can fish'.split()
    y_hat,score = tagger_base.classifier_tagger(w,
                                                features.word_feats,
                                                theta_clf_hand,
                                                all_tags)
    eq_(y_hat, ['PRON','NOUN','NOUN'])
    eq_(score, sum([1,0.1,1.1]))

    theta_clf_hand2 = theta_clf_hand.copy()
    theta_clf_hand2['AUX',CURR_WORD_FEAT,'can'] = 1.5

    y_hat,score = tagger_base.classifier_tagger(w,
                                                features.word_feats,
                                                theta_clf_hand2,
                                                all_tags)
    eq_(y_hat,['PRON','AUX','NOUN'])
    eq_(score, sum([1,1.5,1.1]))
    
    y_hat,score = tagger_base.classifier_tagger("They can fish efficiently".split(),
                                                features.word_feats,
                                                theta_clf_hand2,
                                                all_tags)
    eq_(y_hat,['PRON','AUX','NOUN','NOUN'])
    eq_(score, sum([1,1.5,1.1,0.1]))

# 0.7 points
def test_compute_features_d1_3():
    fv1 = tagger_base.compute_features('the old man the boat'.split(),
                                       ['DET','NOUN','VERB','DET','NOUN'],
                                       features.word_feats)
    eq_(len(fv1),8)
    eq_(fv1[('DET',CURR_WORD_FEAT,'the')],2.0)
    eq_(fv1[('DET',OFFSET)],2.0)
    eq_(fv1[(END_TAG,OFFSET)],1.0) # don't forget to cover this edge case!
    eq_(fv1[('NOUN',CURR_WORD_FEAT,'boat')],1.0)

# 0.7 points
def test_sp_update_d1_4():
    global theta_clf_hand

    # predicted tags should be: [PRON,NOUN,NOUN]
    update = structure_perceptron.sp_update('They can fish'.split(),
                                            ['PRON','AUX','VERB'],
                                            theta_clf_hand,
                                            features.word_feats,
                                            tagger_base.classifier_tagger,
                                            all_tags)
    eq_(sum(update.values()),0)
    eq_(update['NOUN',CURR_WORD_FEAT,'fish'],-1)
    eq_(update['VERB',CURR_WORD_FEAT,'fish'],1)
    eq_(update['PRON',CURR_WORD_FEAT,'They'],0)
    eq_(update['NOUN',OFFSET],-2)
    eq_(update['AUX',CURR_WORD_FEAT,'can'],1)
    eq_(update['NOUN',CURR_WORD_FEAT,'can'],-1)
    eq_(update['VERB',OFFSET],1)
    eq_(update['AUX',OFFSET],1)
    
    theta_clf_hand2 = theta_clf_hand.copy()
    theta_clf_hand2['AUX',CURR_WORD_FEAT,'can'] = 1.5

    # predicted tags should be: [PRON,AUX,NOUN]
    update = structure_perceptron.sp_update('They can fish'.split(),
                                            ['PRON','AUX','VERB'],
                                            theta_clf_hand2,
                                            features.word_feats,
                                            tagger_base.classifier_tagger,
                                            all_tags)
    eq_(sum(update.values()),0)
    eq_(update['NOUN',CURR_WORD_FEAT,'fish'],-1)
    eq_(update['VERB',CURR_WORD_FEAT,'fish'],1)
    eq_(update['PRON',CURR_WORD_FEAT,'They'],0)
    eq_(update['NOUN',OFFSET],-1)
    eq_(update['AUX',CURR_WORD_FEAT,'can'],0)
    eq_(update['NOUN',CURR_WORD_FEAT,'can'],0)
    eq_(update['VERB',OFFSET],1)
    eq_(update['AUX',OFFSET],0)

# this test only looks at a single instance, so weight averaging should be trivial
# 0.35 points
def test_sp_estimate_d1_5a():
    global all_tags, toy_data
    
    theta_toy_one_inst,_ = structure_perceptron.estimate_perceptron(toy_data[:1],
                                                                    features.word_feats,
                                                                    tagger_base.classifier_tagger,
                                                                    1,
                                                                    all_tags)
    eq_(theta_toy_one_inst['NOUN',CURR_WORD_FEAT,'fish'],-1)
    eq_(theta_toy_one_inst['VERB',CURR_WORD_FEAT,'fish'],1)
    eq_(theta_toy_one_inst['NOUN',OFFSET],-2.999) # not -3 because of the tie-breaking initialization
    eq_(theta_toy_one_inst['VERB',OFFSET],1.0)
    eq_(theta_toy_one_inst['AUX',CURR_WORD_FEAT,'can'],1)
    eq_(theta_toy_one_inst['NOUN',CURR_WORD_FEAT,'can'],-1)
    
    # nothing should change after 10 iterations, because the tagger should get it right
    # after the first iteration, so the weights stay the same
    theta_toy_one_inst,_ = structure_perceptron.estimate_perceptron(toy_data[:1],
                                                                    features.word_feats,
                                                                    tagger_base.classifier_tagger,
                                                                    10,
                                                                    all_tags)
        
    eq_(theta_toy_one_inst['NOUN',CURR_WORD_FEAT,'fish'],-1)
    eq_(theta_toy_one_inst['VERB',CURR_WORD_FEAT,'fish'],1)
    eq_(theta_toy_one_inst['NOUN',OFFSET],-2.999) # not -3 because of the tie-breaking initialization
    eq_(theta_toy_one_inst['VERB',OFFSET],1.0)
    eq_(theta_toy_one_inst['AUX',CURR_WORD_FEAT,'can'],1)
    eq_(theta_toy_one_inst['NOUN',CURR_WORD_FEAT,'can'],-1)
    
# now let's look at two instances
# 0.35 points
def test_sp_estimate_d1_5b():
    global all_tags, toy_data
    theta_toy,_ = structure_perceptron.estimate_perceptron(toy_data,
                                                           features.word_feats,
                                                           tagger_base.classifier_tagger,
                                                           1)
    eq_(theta_toy['PRON',CURR_WORD_FEAT,'They'],1)
    eq_(theta_toy['NOUN',CURR_WORD_FEAT,'They'],-1)
    eq_(theta_toy['AUX',CURR_WORD_FEAT,'can'],1)
    eq_(theta_toy['VERB',CURR_WORD_FEAT,'man'],0.5) #averaging
    eq_(theta_toy['NOUN',CURR_WORD_FEAT,'boat'],0.5) #averaging
    eq_(theta_toy['DET',OFFSET],1) #appears twice in second instance, gets averaged

# 0.25 points
def test_sp_score_d1_6():
    confusion = scorer.get_confusion(DEV_FILE,'avp-words.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.805) # should be .8129

# 0.25 points
def test_sp_score_d1_6_test():
    confusion = scorer.get_confusion(TEST_FILE,'avp-words-te.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc, .815) # should be .8229

# 0.25 points for 4650, 0.125 points for 7650
def test_sp_score_d1_7():
    confusion = scorer.get_confusion(JA_DEV_FILE,'avp-words.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc, .78) # should be .7902

# 0.25 points for 4650, 0.125 points for 7650
def test_sp_score_d1_7_test():
    confusion = scorer.get_confusion(JA_TEST_FILE,'avp-words-te.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc, .741) # should be .7514

# 0.5 points
def test_suff_feats_d2_1():
    w = 'The old man a boat'.split()
    tag = 'DET'
    fv0 = features.word_suff_feats(w,tag,'IGNORE',0)
    eq_(len(fv0),3)
    eq_(fv0[tag,CURR_WORD_FEAT,'The'],1)
    eq_(fv0[tag,OFFSET],1)
    eq_(fv0[tag,SUFFIX_FEAT,'he'],1)

    fv1 = features.word_suff_feats(w,tag,'IGNORE',1)
    eq_(len(fv0),3)
    eq_(fv1[tag,CURR_WORD_FEAT,'old'],1)
    eq_(fv1[tag,OFFSET],1)
    eq_(fv1[tag,SUFFIX_FEAT,'ld'],1)

    fv3 = features.word_suff_feats(w,tag,'IGNORE',3)
    eq_(len(fv0),3)
    eq_(fv3[tag,CURR_WORD_FEAT,'a'],1)
    eq_(fv3[tag,OFFSET],1)
    eq_(fv3[tag,SUFFIX_FEAT,'a'],1)

#0.125 points for 4650, 0.0625 points for 7650
def test_suff_feats_acc_d2_2_en_dev():
    confusion = scorer.get_confusion(DEV_FILE,'avp-words-suff.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.834) # should be .844

#0.125 points for 4650, 0.0625 points for 7650
def test_suff_feats_acc_d2_2_en_test():
    confusion = scorer.get_confusion(TEST_FILE,'avp-words-suff-te.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.838) # should be .848

#0.125 points for 4650, 0.0625 points for 7650
def test_suff_feats_acc_d2_2_ja_dev():
    confusion = scorer.get_confusion(JA_DEV_FILE,'avp-words-suff.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.872) # should be .882

#0.125 points for 4650, 0.0625 points for 7650
def test_suff_feats_acc_d2_2_ja_test():
    confusion = scorer.get_confusion(JA_TEST_FILE,'avp-words-suff-te.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.834) # should be .844

# 0.5 points
def test_neighbor_feats_d2_4():
    feats = features.word_neighbor_feats(['The','old','man','a','boat'],'TAG','IGNORE',0)
    eq_(len(feats),4)
    eq_(feats['TAG',CURR_WORD_FEAT,'The'],1)
    eq_(feats['TAG',PREV_WORD_FEAT,PRE_START_TOKEN],1)
    eq_(feats['TAG',NEXT_WORD_FEAT,'old'],1)

    feats = features.word_neighbor_feats(['The','old','man','a','boat'],'TAG','IGNORE',4)
    eq_(len(feats),4)
    eq_(feats['TAG',CURR_WORD_FEAT,'boat'],1)
    eq_(feats['TAG',PREV_WORD_FEAT,'a'],1)
    eq_(feats['TAG',NEXT_WORD_FEAT,POST_END_TOKEN],1)

    feats = features.word_neighbor_feats(['The','old','man','a','boat'],'TAG','IGNORE',5)
    eq_(len(feats),2)
        
# 0.25 / 0.125 points
def test_neighbor_acc_d2_5_en():
    confusion = scorer.get_confusion(DEV_FILE,'avp-words-neighbor.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.848) # should be .858

# 0.25 / 0.125 points
def test_neighbor_acc_d2_5_ja():
    confusion = scorer.get_confusion(JA_DEV_FILE,'avp-words-neighbor.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.792) # should be .802

# 0.25 points
def test_bakeoff_acc_d2_6_en_half_credit():
    acc = scorer.accuracy(scorer.get_confusion(DEV_FILE,'avp-words-best.preds'))
    assert_greater(acc,.87) 

# 0.25 points
def test_bakeoff_acc_d2_6_en_full_credit():
    acc = scorer.accuracy(scorer.get_confusion(DEV_FILE,'avp-words-best.preds'))
    assert_greater(acc,.88) 

# +0.1 points bonus
def test_bakeoff_acc_d2_6_en_beat_the_prof():
    acc = scorer.accuracy(scorer.get_confusion(TEST_FILE,'avp-words-best-te.preds'))
    assert_greater(acc,.88735) 

# 0.25 points
def test_bakeoff_acc_d2_6_ja_half_credit():
    acc = scorer.accuracy(scorer.get_confusion(JA_DEV_FILE,'avp-words-best.ja.preds'))
    assert_greater(acc,.89) 

# 0.25 points
def test_bakeoff_acc_d2_6_ja_full_credit():
    acc = scorer.accuracy(scorer.get_confusion(JA_DEV_FILE,'avp-words-best.ja.preds'))
    assert_greater(acc,.90) 

# +0.1 points bonus
def test_bakeoff_acc_d2_6_ja_beat_the_prof():
    acc = scorer.accuracy(scorer.get_confusion(JA_TEST_FILE,'avp-words-best-te.ja.preds'))
    assert_greater(acc,.87882)

# 0.5 points
def test_viterbi_is_same_d3_1():
    global toy_data, theta_clf_hand, all_tags

    theta_toy_one_inst_classifier,_ = structure_perceptron.estimate_perceptron(toy_data,
                                                                               features.word_feats,
                                                                               tagger_base.classifier_tagger,
                                                                               3,
                                                                               all_tags)

    theta_toy_one_inst_viterbi,_ = structure_perceptron.estimate_perceptron(toy_data,
                                                                features.word_feats,
                                                                viterbi.viterbi_tagger,
                                                                3,
                                                                all_tags)
    eq_(theta_toy_one_inst_viterbi,theta_toy_one_inst_classifier)

# 0.5 points
def test_hmm_features_d3_2():
    global toy_data

    fv1 = features.hmm_feats(toy_data[0][0],'PRON',START_TAG,0)
    eq_(fv1,{('PRON', '--CURR-WORD--', 'They'): 1, ('PRON', '--PREV-TAG--', '--START--'): 1})
    
    fv2 = features.hmm_feats(toy_data[0][0],'AUX','PRON',1)
    eq_(fv2,{('AUX', '--CURR-WORD--', 'can'): 1, ('AUX', '--PREV-TAG--', 'PRON'): 1})
    
    fv3 = features.hmm_feats(toy_data[0][0],'VERB','AUX',2)
    eq_(fv3,{('VERB', '--CURR-WORD--', 'fish'): 1, ('VERB', '--PREV-TAG--', 'AUX'): 1})
    
    fv4 = features.hmm_feats(toy_data[0][0],END_TAG,'VERB',3)
    eq_(fv4,{('--END--', '--PREV-TAG--', 'VERB'): 1})


# 0.25 / 0.125
def test_hmm_feat_acc_d3_3_en():
    confusion = scorer.get_confusion(DEV_FILE,'sp-hmm.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.862) # should be .872

# 0.25 / 0.125
def test_hmm_feat_acc_d3_3_ja():
    confusion = scorer.get_confusion(JA_DEV_FILE,'sp-hmm.ja.preds')
    acc = scorer.accuracy(confusion)
    assert_greater(acc,.797) # should be .807

# 0.25 points
def test_bakeoff_acc_d3_4_en_half_credit():
    acc = scorer.accuracy(scorer.get_confusion(DEV_FILE,'sp-best.preds'))
    assert_greater(acc,.885) 

# 0.25 points
def test_bakeoff_acc_d3_4_en_full_credit():
    acc = scorer.accuracy(scorer.get_confusion(DEV_FILE,'sp-best.preds'))
    assert_greater(acc,.895) 

# +0.1 bonus points
def test_bakeoff_acc_d3_4_en_beat_the_prof():
    acc = scorer.accuracy(scorer.get_confusion(TEST_FILE,'sp-best-te.preds'))
    assert_greater(acc,.88735) # same as with the classification-based tagger!

# 0.25 points
def test_bakeoff_acc_d3_4_ja_half_credit():
    acc = scorer.accuracy(scorer.get_confusion(JA_DEV_FILE,'sp-best.ja.preds'))
    assert_greater(acc,.90) 

# 0.25 points
def test_bakeoff_acc_d3_4_ja_full_credit():
    acc = scorer.accuracy(scorer.get_confusion(JA_DEV_FILE,'sp-best.ja.preds'))
    assert_greater(acc,.91) 

# +0.1 bonus points
def test_bakeoff_acc_d3_4_ja_beat_the_prof():
    acc = scorer.accuracy(scorer.get_confusion(JA_TEST_FILE,'sp-best-te.ja.preds'))
    assert_greater(acc,.879926) 


