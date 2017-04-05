from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag 
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    d_weights = defaultdict(float)
    y_hats, score = tagger(tokens, feat_func, weights, all_tags)
    fv_y_hat = tagger_base.compute_features(tokens,y_hats,feat_func)
    fv_y = tagger_base.compute_features(tokens,tags,feat_func)

    for k,v in fv_y_hat.iteritems():
        d_weights[k] -= v
    for k,v in fv_y.iteritems():
        d_weights[k] += v

    return d_weights
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    # compute all_tags if it's not provided
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    # this initialization should make sure there isn't a tie for the first prediction
    # this makes it easier to test your code
    weights = defaultdict(float,
                          {('NOUN',constants.OFFSET):1e-3})

    weight_history = []
    w_sum = defaultdict(float)
    t = 0.0
    for it in xrange(N_its):
    	print it,
        for tokens, tags in labeled_instances:
            updates = sp_update(tokens, tags, weights, feat_func, tagger, all_tags)
            for k,v in updates.items():
                weights[k] += v
                w_sum[k] += t*v
            t += 1.0
        avg_weights = defaultdict(float)
        for k,v in weights.items():
            avg_weights[k] = weights[k] - w_sum[k]/t
        weight_history.append(avg_weights.copy())

    return avg_weights, weight_history