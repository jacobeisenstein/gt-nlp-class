from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE, DEV_FILE
from gtnlplib import most_common, clf_base, preproc, scorer, tagger_base

def setup():
    global all_tags, theta_mc, tagger_mc

    all_tags = preproc.get_all_tags(TRAIN_FILE)

    theta_mc = most_common.get_most_common_word_weights(TRAIN_FILE)
    tagger_mc = tagger_base.make_classifier_tagger(theta_mc)

## when there are multiple tests for a single question, must pass *both* tests for credit
    
#1.1a (0.5 points)
def test_get_top_noun_tags():
    expected = [('time', 385), ('people', 233), ('way', 187)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["NOUN"].most_common(3)
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

#1.1a
def test_get_top_verb_tags():
    expected = [('is', 1738), ('was', 808), ('have', 748)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["VERB"].most_common(3)
    eq_(expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

#2.1 (0.5 points)
def test_classifier_tagger():
    global all_tags
    
    expected = 0.1668919993637665

    noun_weights = most_common.get_noun_weights()
    noun_tagger = tagger_base.make_classifier_tagger(noun_weights)

    confusion = tagger_base.eval_tagger(noun_tagger,'all_nouns.preds',all_tags=all_tags)
    actual  = scorer.accuracy(confusion)

    assert_almost_equals(expected, actual,places=3, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

#2.2a (0.5 points)
def test_mcc_tagger_output():
    global tagger_mc, all_tags
    
    tags = tagger_mc(['They','can','can','fish'],all_tags)
    eq_(tags,['PRON','AUX','AUX','NOUN'])

    tags = tagger_mc(['The','old','man','the','boat','.'],all_tags)
    eq_(tags,['DET', 'ADJ', 'NOUN', 'DET', 'NOUN','PUNCT'])
        
#2.2b
def test_mcc_tagger_accuracy():
    global tagger_mc, all_tags
        
    expected = 0.848

    confusion = tagger_base.eval_tagger(tagger_mc,'most-common.preds',all_tags=all_tags)
    actual = scorer.accuracy(confusion)
    
    ok_(expected < actual, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected, actual))
