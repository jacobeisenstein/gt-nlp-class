from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest
from gtnlplib.constants import TRAIN_FILE, DEV_FILE
from gtnlplib import most_common, clf_base, preproc, scorer, tagger_base

#1.1a (0.5 points)
def test_get_top_noun_tags():
    expected = [('people', 53), ('time', 48), ('world', 46)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["NOUN"].most_common(3)
    eq_ (expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))

#1.1b
def test_get_top_verb_tags():
    expected = [('is', 335), ('was', 128), ('have', 110)]
    tag_word_counts = most_common.get_tag_word_counts(TRAIN_FILE)
    actual = tag_word_counts["VERB"].most_common(3)
    eq_(expected, actual, msg="UNEQUAL Expected:%s, Actual:%s" %(expected, actual))