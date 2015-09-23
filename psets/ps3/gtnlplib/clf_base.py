import operator
# use this to find the highest-scoring label
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# hide inner code
# should return two outputs: the highest-scoring label, and the scores for all labels
def predict(instance,weights,labels):
    """Predict the best label for the instance given weights

    Parameters:
    instance -- An iterable of (feature, count) pairs
    weights -- Weights dict mapping (label, feature) pairs to numeric score.
    labels -- List of all possible labels

    Returns:
    tuple of (best_label, scores) where
    best_label -- The label with the highest score for this instance
    scores -- dict of scores for each label
    """
    scores = {label:sum([weights[(label,feat)]*count for feat,count in instance.iteritems()]) for label in labels}
    return argmax(scores),scores

