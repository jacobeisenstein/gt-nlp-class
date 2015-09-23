from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtnlplib.constants import TRAIN_FILE
from gtnlplib import most_common, clf_base,preproc,scorer,tagger_base

def test_get_top_noun_tags():
    expected = [('day', 19), ('time', 18), ('people', 17)]
    tag_counter = most_common.get_tags(TRAIN_FILE)
    actual = tag_counter["N"].most_common(3)
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_get_top_verb_tags():
    expected = [('is', 105), ('are', 52), ('have', 48)]
    tag_counter = most_common.get_tags(TRAIN_FILE)
    actual = tag_counter["V"].most_common(3)
    eq_(expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_classifier_tagger():
    expected = 0.136844287788
    noun_weights = most_common.get_noun_weights()
    noun_tagger = tagger_base.makeClassifierTagger(noun_weights)
    
    confusion = tagger_base.evalTagger(noun_tagger,'nouns')
    actual  = scorer.accuracy(confusion)

    assert_almost_equals(expected, actual,places=3, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

def test_get_most_common_tag():
    expected = 0.63
    weights = most_common.get_most_common_weights(TRAIN_FILE)
    confusion = tagger_base.evalTagger(tagger_base.makeClassifierTagger(weights),'mcc')
    actual = scorer.accuracy(confusion)
    
    ok_(expected < actual, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected, actual))
