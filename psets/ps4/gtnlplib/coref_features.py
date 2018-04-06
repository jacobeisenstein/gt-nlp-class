import itertools
from . import coref_rules
from collections import defaultdict

## deliverable 3.1
def minimal_features(markables,a,i):
    '''
    Compute a minimal set of features for antecedent a and mention i

    :param markables: list of markables for the document
    :param a: index of antecedent
    :param i: index of mention
    :returns: dict of features
    :rtype: defaultdict
    '''
    
    f = defaultdict(float)
    # STUDENT
    
    # END STUDENT
    return f

## deliverable 3.5
def distance_features(x,a,i,
                      max_mention_distance=5,
                      max_token_distance=10):
    '''
    compute a set of distance features for antecedent a and mention i

    :param x: markable list for document
    :param a: antecedent index
    :param i: mention index
    :param max_mention_distance: upper limit on mention distance
    :param max_token_distance: upper limit on token distance
    :returns: dict of features
    :rtype: defaultdict
    '''
    
    f = defaultdict(float)
    # STUDENT
    
    # END STUDENT
    return f

## deliverable 3.6
def make_feature_union(feat_func_list):
    '''
    return a feature function that is the union of the feature functions in the list

    :param feat_func_list: list of feature functions
    :returns: feature function
    :rtype: function
    '''
    raise NotImplementedError

## deliverable 6
def make_bakeoff_features():
    raise NotImplementedError

