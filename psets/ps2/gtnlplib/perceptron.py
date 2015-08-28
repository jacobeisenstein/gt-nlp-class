import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer

def oneItPerceptron(data_generator,weights,labels):
    errors = 0.
    num_insts = 0.
    # your code here
    return weights, errors, num_insts

# this code trains the perceptron for N iterations on the supplied training data
def trainPerceptron(N_its,inst_generator,labels, outfile, devkey):
    tr_acc = [None]*N_its #holder for training accuracy
    dv_acc = [None]*N_its #holder for dev accuracy
    weights = defaultdict(float) 
    for i in xrange(N_its):
        weights,tr_err,tr_tot = oneItPerceptron(inst_generator,weights,labels) #call your function for a single iteration
        confusion = evalClassifier(weights,outfile, devkey) #evaluate on dev data
        dv_acc[i] = scorer.accuracy(confusion) #compute accuracy
        tr_acc[i] = 1. - tr_err/float(tr_tot) #compute training accuracy from output
        print i,'dev: ',dv_acc[i],'train: ',tr_acc[i] 
    return weights, tr_acc, dv_acc
