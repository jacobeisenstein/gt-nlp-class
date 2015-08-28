import operator
from  constants import *
from collections import defaultdict, Counter
from clf_base import predict, evalClassifier
import scorer

def trainAvgPerceptron(N_its,inst_generator,labels, outfile, devkey):
    return avg_weights, tr_acc, dv_acc


def oneItAvgPerceptron(inst_generator,weights,wsum,labels,Tinit=0):
    return weights, wsum, tr_err, i
