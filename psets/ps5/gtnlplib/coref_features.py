import itertools
import coref_rules
from nltk import wordnet

# useful?
pronoun_list=['it','he','she','they','this','that']
poss_pronoun_list=['its','his','her','their']
oblique_pronoun_list=['him','her','them']
def_list=['the','this','that','these','those']
indef_list=['a','an','another']

# d3.1
def minimal_features(markables,a,i):
    """Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: features
    :rtype: dict

    """
    f = dict()
    ## your code here
    ## use functions from coref_rules
    return f

# deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=10,
                      max_token_distance=10):
    """compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: feature dict
    :rtype: dict

    """
    f = dict()
    ## your code here
    return f
    
###### Feature combiners

# deliverable 3.6
def make_feature_union(feat_func_list):
    """return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function

    """
    def f_out(x,a,i):
        # your code here
        return None
    return f_out

# deliverable 3.7
def make_feature_cross_product(feat_func1,feat_func2):
    """return a feature function that is the cross-product of the two feature functions

    :param feat_func1: a feature function
    :param feat_func2: a feature function
    :returns: another feature function
    :rtype: function

    """
    def f_out(x,a,i):
        # your code here
        return None
    return f_out




