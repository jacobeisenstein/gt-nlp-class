from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    update = defaultdict(float)
    # getting y head
    y_head, scores = predict(x, weights, labels)
 

    if y != y_head:
        fv_y = make_feature_vector(x, y)
        fv_y_head = make_feature_vector(x, y_head)
        for key, val in fv_y.items():
            update[key] += val
        for key, val in fv_y_head.items():
            update[key] -= val

    return update


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            updates = perceptron_update(x_i, y_i, weights, labels)
            for k,v in updates.items():
                weights[k] += v
        weight_history.append(weights.copy())
    return weights, weight_history

def estimate_avg_perceptron(x,y,N_its):
    """estimate averaged perceptron classifier

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    w_sum = defaultdict(float) #hint
    weights = defaultdict(float)
    weight_history = []

    t=1.0 #hint
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            # YOUR CODE GOES HERE
            updates = perceptron_update(x_i, y_i, weights, labels)
            for k,v in updates.items():
                weights[k] += v
                w_sum[k] += t*v
            t += 1.0
        avg_weights = defaultdict(float)
        for k,v in w_sum.items():
            avg_weights[k] = weights[k] - w_sum[k]/t
        weight_history.append(avg_weights.copy())

    return avg_weights, weight_history
