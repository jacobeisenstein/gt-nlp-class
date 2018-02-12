from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest
from gtnlplib.constants import TRAIN_FILE, DEV_FILE
from gtnlplib import most_common, clf_base, preproc, scorer, tagger_base

def setup():
    global all_tags, theta_mc, tagger_mc

    all_tags = preproc.get_all_tags(TRAIN_FILE)

    theta_mc = most_common.get_most_common_word_weights(TRAIN_FILE)
    tagger_mc = tagger_base.make_classifier_tagger(theta_mc)

## when there are multiple tests for a single question, must pass *both* tests for credit

#2.2a 
def test_mcc_tagger_output():
    global tagger_mc, all_tags
    
    tags = tagger_mc(['They','can','can','fish'],all_tags)
    eq_(tags,['PRON','AUX','AUX','NOUN'])

    tags = tagger_mc(['The','old','man','the','boat','.'],all_tags)
    eq_(tags,['DET', 'ADJ', 'NOUN', 'DET', 'PROPN', 'PUNCT'])
        
#2.2b
def test_mcc_tagger_accuracy():
    global tagger_mc, all_tags
        
    expected = 0.811124

    confusion = tagger_base.eval_tagger(tagger_mc,'most-common.preds',all_tags=all_tags)
    actual = scorer.accuracy(confusion)
    
    ok_(expected < actual, msg="NOT_IN_RANGE Expected:%f, Actual:%f" %(expected, actual))
