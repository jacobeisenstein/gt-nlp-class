from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest
from gtnlplib.constants import TRAIN_FILE, DEV_FILE
from gtnlplib import most_common, clf_base, preproc, scorer, tagger_base

def setup():
    global all_tags

    all_tags = preproc.get_all_tags(TRAIN_FILE)


#2.1 
def test_classifier():
    global all_tags
    
    expected = 0.1527613022274944

    noun_weights = most_common.get_noun_weights()
    noun_tagger = tagger_base.make_classifier_tagger(noun_weights)

    confusion = tagger_base.eval_tagger(noun_tagger,'all_nouns.preds',all_tags=all_tags)
    actual  = scorer.accuracy(confusion)

    assert_almost_equal(expected, actual,places=3, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

