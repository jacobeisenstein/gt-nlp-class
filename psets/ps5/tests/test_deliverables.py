from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest
from gtparsing import dependency_parser, dependency_features, custom_features
from gtparsing import dependency_reader
from gtparsing import utilities

from score import accuracy

import os, errno

DIR = "data/deppars/"

ENGLISH = "english"
GERMAN  = "german"
SPANISH = "spanish"
ITALIAN = "italian"
FRENCH  = "french"
PORTO   = "portuguese"
ENGLISH_UNIVTAGS = "english_univtags"

ENGLISH_KEYFILE = os.path.join (DIR, "english_dev.conll")
GERMAN_KEYFILE  = os.path.join (DIR, "german_dev.conll")
SPANISH_KEYFILE = os.path.join (DIR, "spanish_dev.conll")
ITALIAN_KEYFILE = os.path.join (DIR, "italian_dev.conll")
FRENCH_KEYFILE  = os.path.join (DIR, "french_dev.conll")
PORTO_KEYFILE   = os.path.join (DIR, "portuguese_dev.conll")
UNIVERSAL_ENGLISH_KEYFILE   = os.path.join (DIR, "english_univtags_dev.conll")

KEYFILES = {
             ENGLISH:ENGLISH_KEYFILE,
             GERMAN :GERMAN_KEYFILE,
             SPANISH:SPANISH_KEYFILE,
             ITALIAN:ITALIAN_KEYFILE,
             FRENCH :FRENCH_KEYFILE,
             PORTO  :PORTO_KEYFILE,
             ENGLISH_UNIVTAGS : UNIVERSAL_ENGLISH_KEYFILE
           }

ENGLISH_TEST_FILE = "tests/english_test.conll"

DELIVERABLE1a = os.path.join (DIR, "deliverable1a.conll")
DELIVERABLE1b = os.path.join (DIR, "deliverable1b.conll")
DELIVERABLE1c = os.path.join (DIR, "deliverable1c.conll")

instances = None

def silentremove (filename):
    """
    Remove all temporary files created.
    """
    try:
        os.remove (filename)
    except OSError as e:
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory.
            raise # raise error if different error occurred.

@nottest
def getForeignLanguage (directory, suffix):
    for filename in os.listdir (directory):
        if filename.endswith (suffix):
            lang = filename.split (".")[0]
            print lang
            break
    return lang, filename

def setup_module ():
    global instances
    dr = dependency_reader.DependencyReader ()
    word_id, pos_id = 0,0
    conll_file = open(ENGLISH_TEST_FILE)
    
    dr.word_dict["__START__"] = word_id # Start symbol
    word_id+=1
    dr.word_dict["__STOP__"] = word_id # Stop symbol
    word_id+=1
    dr.pos_dict["__START__"] = pos_id # Start symbol
    pos_id+=1
    dr.pos_dict["__STOP__"] = pos_id # Stop symbol
    pos_id+=1

    for line in conll_file:
        line = line.rstrip()
        if len(line) == 0:
            continue
        fields = line.split("\t")
        word = fields[1]
        pos = fields[3]
        if word not in dr.word_dict:
            dr.word_dict[word] = word_id
            word_id+=1
        if pos not in dr.pos_dict:
            dr.pos_dict[pos] = pos_id
            pos_id+=1
    conll_file.close()
    instances = dr.loadInstances (ENGLISH_TEST_FILE)

def teardown_module ():
    pass

def test_features_for_deliverable1a ():
    global instances
    f = custom_features.LexDistFeats()
    expected_feature_dict = {(1, 5, 4): 7, (1, 0, 5): 9, (1, 5, 3): 4, (1, 3, 2): 1, 
                (0, 5, 6): 11, (0, 5, 3): 3, (2, 1): 2, (1, 5, 6): 12, 
                (0, 0, 5): 8, (2, -1): 13, (2, 2): 5, (0, 3, 2): 0, 
                (0, 5, 4): 6, (2, -4): 10}
    f.create_dictionary(instances)
    expected = 0
    actual = cmp (expected_feature_dict, f.feat_dict)
    eq_ (expected, actual, msg="Features Mismatch for 1a: Expected %s, Actual %s" %(str(expected_feature_dict), str(f.feat_dict))) 

def test_accuracy_for_deliverable1a ():
    expected = 0.694
    actual   = accuracy (KEYFILES[ENGLISH], DELIVERABLE1a)
    assert_almost_equals (expected, actual, places=3, msg="Accuracy Incorrect for 1a: Expected %f, Actual %f" %(expected, actual))


def test_features_for_deliverable1b ():
    global instances
    f = custom_features.LexDistFeats2()
    expected_feature_dict = {(1, 5, 4): 9, (1, 0, 5): 12, (1, 5, 3): 5, (1, 3, 2): 1, (3, 6, 5): 18, 
                             (3, 5, 0): 14, (0, 5, 6): 15, (0, 5, 3): 4, (2, 1): 2, (1, 5, 6): 16, 
                             (3, 4, 5): 10, (3, 3, 5): 7, (2, -1): 17, (2, 2): 6, (3, 2, 3): 3, 
                             (0, 3, 2): 0, (0, 0, 5): 11, (0, 5, 4): 8, (2, -4): 13}
    f.create_dictionary(instances)
    expected = 0
    actual = cmp (expected_feature_dict, f.feat_dict)
    eq_ (expected, actual, msg="Features Mismatch for 1a: Expected %s, Actual %s" %(str(expected_feature_dict), str(f.feat_dict))) 

def test_accuracy_for_deliverable1b ():
    expected = 0.729 
    actual   = accuracy (KEYFILES[ENGLISH], DELIVERABLE1b)
    assert_almost_equals (expected, actual, places=3, msg="Accuracy Incorrect for 1b: Expected %f, Actual %f" %(expected, actual))

def test_features_for_deliverable1c ():
    global instances
    f = custom_features.ContextFeats()
    expected_feat_dict =  {(4, 3, 2, 2): 4, (0, 5, 3): 5, (1, 5, 3): 6, (1, 3, 2): 1, (0, 5, 6): 18, 
                           (2, 1): 2, (4, 5, 4, 6): 22, (2, -4): 16, (1, 0, 5): 15, (3, 3, 5): 8, 
                           (0, 0, 5): 14, (2, 2): 7, (3, 2, 3): 3, (0, 3, 2): 0, (3, 6, 5): 21, (2, -1): 20, 
                           (1, 5, 4): 11, (3, 5, 0): 17, (4, 5, 4, 3): 9, (4, 5, 4, 4): 13, (1, 5, 6): 19, 
                           (3, 4, 5): 12, (0, 5, 4): 10}

    f.create_dictionary(instances)
    expected = set ([key[1:] for key in expected_feat_dict])
    actual = set ([key[1:] for key in f.feat_dict])
    
    ok_ (expected <= actual, msg="Features Mismatch for 1c: Expected %s, Actual %s" %(str(expected_feat_dict), str(f.feat_dict)))

def test_accuracy_for_deliverable1c ():
    expected = 0.82
    actual   = accuracy (KEYFILES[ENGLISH], DELIVERABLE1c)
    ok_(expected < (actual + 0.002), msg="Accuracy is lesser than expected for 1c: Expected %f, Actual %f" %(expected, actual))

def test1_CPT_for_deliverable2a ():
    dp = dependency_parser.DependencyParser(feature_function=dependency_features.DependencyFeatures())
    dp.read_data(ENGLISH)
    verb_distribution = utilities.CPT (dp.reader.train_instances, dp.reader.pos_dict['VB'])
    assert_almost_equals (verb_distribution[dp.reader.pos_dict['IN']], 0.1562, places=3, msg="Incorrect for 2a: Expected CPT incorrect")
    assert_almost_equals (sum(verb_distribution.values()), 1.0, places=2, msg="Incorrect for 2a: Not a probability distribution")

def test2_CPT_for_deliverable2a ():
    dp = dependency_parser.DependencyParser(feature_function=dependency_features.DependencyFeatures())
    dp.read_data(ENGLISH_UNIVTAGS)
    verb_distribution = utilities.CPT (dp.reader.train_instances, dp.reader.pos_dict['VERB'])
    assert_almost_equals (verb_distribution[dp.reader.pos_dict['NOUN']], 0.2736, places=3, msg="Incorrect for 2a: Expected CPT incorrect")
    assert_almost_equals (sum(verb_distribution.values()), 1.0, places=2, msg="Incorrect for 2a: Not a probability distribution")

def test1_entropy_for_deliverable2b ():
    dp = dependency_parser.DependencyParser(feature_function=dependency_features.DependencyFeatures())
    dp.read_data(ENGLISH)
    verb_distribution = utilities.CPT (dp.reader.train_instances, dp.reader.pos_dict['VB'])
    expected = 2.653
    actual = utilities.entropy (verb_distribution)
    ok_ (expected < (actual + 0.002), msg="Entropy incorrect for 2b: Expected %f, Actual %f" %(expected, actual))

def test2_entropy_for_deliverable2b ():
    dp = dependency_parser.DependencyParser(feature_function=dependency_features.DependencyFeatures())
    dp.read_data(ENGLISH_UNIVTAGS)
    verb_distribution = utilities.CPT (dp.reader.train_instances, dp.reader.pos_dict['VERB'])
    expected = 1.9
    actual = utilities.entropy (verb_distribution)
    ok_ (expected < (actual + 0.002), msg="Entropy incorrect for 2b: Expected %f, Actual %f" %(expected, actual))

def test_accuracy_for_deliverable2c ():
    expected = {GERMAN: 0.432, SPANISH:0.365 , ITALIAN: 0.311, FRENCH: 0.372, PORTO: 0.305}
    SUFFIX = "deliverable2c.conll"
    lang, filename = getForeignLanguage (DIR, SUFFIX)
    actual   = accuracy (KEYFILES[lang], os.path.join (DIR, filename))
    ok_ (expected[lang] <= (actual + 0.002), msg="Accuracy Incorrect for 2c: Expected %f, Actual %f" %(expected[lang], actual)) # adding some tolerance.

def test_accuracy_for_deliverable2d ():
    expected = {GERMAN: 0.440, SPANISH: 0.497, ITALIAN: 0.317, FRENCH: 0.277, PORTO: 0.44}
    SUFFIX = "deliverable2d.conll"
    lang, filename = getForeignLanguage (DIR, SUFFIX)
    actual   = accuracy (KEYFILES[lang], os.path.join (DIR, filename))
    ok_ (expected[lang] <= (actual + 0.002), msg="Accuracy Incorrect for 2d: Expected %f, Actual %f" %(expected[lang], actual))

def test_accuracy_for_deliverable2e ():
    expected = {GERMAN: 0.532, SPANISH: 0.614, ITALIAN: 0.439, FRENCH: 0.327, PORTO: 0.572}
    SUFFIX = "deliverable2e.conll"
    lang, filename = getForeignLanguage (DIR, SUFFIX)
    actual   = accuracy (KEYFILES[lang], os.path.join (DIR, filename))
    ok_ (expected[lang]<= (actual + 0.002), msg="Accuracy Incorrect for 2e: Expected %f, Actual %f" %(expected[lang], actual))


def test_accuracy_for_deliverable2f_almost_there ():
    expected = {GERMAN: 0.562, SPANISH: 0.644, ITALIAN: 0.469, FRENCH: 0.357, PORTO: 0.602}
    SUFFIX = "deliverable2f.conll"
    lang, filename = getForeignLanguage (DIR, SUFFIX)
    actual   = accuracy (KEYFILES[lang], os.path.join (DIR, filename))
    ok_ (expected[lang]<= (actual + 0.002), msg="Accuracy Incorrect for 2f: Expected %f, Actual %f" %(expected[lang], actual))

def test_accuracy_for_deliverable2f ():
    expected = {GERMAN: 0.612, SPANISH: 0.662, ITALIAN: 0.657, FRENCH: 0.582, PORTO: 0.624}
    SUFFIX = "deliverable2f.conll"
    lang, filename = getForeignLanguage (DIR, SUFFIX)
    actual   = accuracy (KEYFILES[lang], os.path.join (DIR, filename))
    ok_ (expected[lang]<= (actual + 0.002), msg="Accuracy Incorrect for 2f: Expected %f, Actual %f" %(expected[lang], actual))


