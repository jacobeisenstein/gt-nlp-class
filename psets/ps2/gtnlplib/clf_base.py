from gtnlplib.constants import OFFSET
import numpy as np

# hint! use this.
argmax = lambda x : max(x.iteritems(),key=lambda y : y[1])[0]

def make_feature_vector(base_features,label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    fv = dict()
    fv[(label,OFFSET)] = 1
    for k,v in base_features.items():
        fv[(label,k)] = v
    return fv

def predict(base_features,weights,labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = dict()
    #initialize the dictionary
    for label in labels:
        scores[label] = 0

    # update scores in the dictionary
    for label in labels:
        fv = make_feature_vector(base_features, label)
        for feature, value in fv.items():
            weight = weights[feature]
            scores [label] += weight*(value)

    return argmax(scores),scores

def predict_all(x,weights,labels):
    """Predict the label for all instances in a dataset

    :param x: base instances
    :param weights: defaultdict of weights
    :returns: predictions for each instance
    :rtype: numpy array

    """
    y_hat = np.array([predict(x_i,weights,labels)[0] for x_i in x])
    return y_hat

def get_top_features_for_label(weights,label,k=5):
    """Return the five features with the highest weight for a given label.

    :param weights: the weight dictionary
    :param label: the label you are interested in 
    :returns: list of tuples of features and weights
    :rtype: list
    """
    filtered_dict = { k:v for k,v in weights.iteritems() if label in k and v != 0.0}
    top_5_weights = sorted(filtered_dict.iteritems(),key=lambda y : y[1], reverse=True)[:k]

    return top_5_weights
