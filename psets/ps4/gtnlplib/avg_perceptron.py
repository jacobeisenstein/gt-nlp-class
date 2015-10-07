from collections import defaultdict
from gtnlplib.tagger_base import classifierTagger
from gtnlplib.tagger_base import evalTagger 
from gtnlplib import scorer

def oneItAvgPerceptron(inst_generator,featfunc,weights,wsum,tagset,Tinit=0):
    """
    :param inst_generator: iterator over instances
    :param featfunc: feature function on (words, tag_m, tag_m_1, m)
    :param weights: default dict
    :param wsum: weight sum, for averaging
    :param tagset: set of permissible tags
    :param Tinit: initial value of t, the counter over instances
    """
    tr_err = 0.0
    for i,(words,y_true) in enumerate(inst_generator):
        pass # your code here
    # note that i'm computing tr_acc for you, as long as you properly update tr_err
    return weights, wsum, 1.-tr_err / float(sum([len(s) for s,t in inst_generator])), i


def trainAvgPerceptron(N_its,inst_generator,featfunc,tagset):
    """
    :param N_its: number of iterations
    :param inst_generator: generate words,tags pairs
    :param featfunc: feature function
    :param tagset: set of all possible tags
    :returns average weights, training accuracy, dev accuracy
    """
    tr_acc = [None]*N_its
    dv_acc = [None]*N_its
    T = 0
    weights = defaultdict(float)
    wsum = defaultdict(float)
    for i in xrange(N_its):
        # your code here
        confusion = evalTagger(lambda words, alltags: classifierTagger(words,featfunc,avg_weights,tagset),'perc')
        dv_acc[i] = scorer.accuracy(confusion)
        tr_acc[i] = tr_acc_i
        print i,'dev:',dv_acc[i],'train:',tr_acc[i]
    return avg_weights, tr_acc, dv_acc