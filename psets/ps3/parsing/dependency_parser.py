'''
Original code by Joao Graca and Andre Martins (2011-2012)
Modified by Jacob Eisenstein (2013) for Georgia Tech CS 4650/7650 NLP
'''
import sys
import numpy as np
from dependency_reader import * 
from dependency_writer import * 
from dependency_features import * 
from dependency_decoder import * 
from util.my_math_utils import *


class DependencyParser():
    '''
    Dependency parser class

    '''
    def __init__(self, feature_function=DependencyFeatures()):
        self.trained = False
        self.language = ""
        self.weights = []
        self.decoder = DependencyDecoder()
        self.reader = DependencyReader()
        self.writer = DependencyWriter()
        self.features = feature_function # DependencyFeatures()

    def read_data(self, language):
        self.language = language
        self.reader.load(language)
        self.features.create_dictionary(self.reader.train_instances)

    def evalInstances(self,instances,weight_update = None):
        n_mistakes = 0
        n_tokens = 0
        n_instances = 0
        for instance in instances:
            true_feats = []
            pred_feats = []
            feats = self.features.create_features(instance)
            scores = self.features.compute_scores(feats, self.weights)
            heads_pred = self.decoder.parse_nonproj(scores)
            for m in range(np.size(heads_pred)): #find each modifier's head, add the score
                if heads_pred[m] != instance.heads[m]:
                    n_mistakes += 1
                    if weight_update:
                        true_feats.extend(feats[instance.heads[m]][m])
                        pred_feats.extend(feats[heads_pred[m]][m])
                n_tokens += 1
                if weight_update:
                    weight_update(true_feats,pred_feats,scores,heads_pred,instance)
            n_instances += 1
        return (n_tokens - n_mistakes)/float(n_tokens)

    def perceptron_update(self,true_feats,pred_feats,scores,heads_pred,instance):
        for f in true_feats:
            self.weights[f] += 1.0
        for f in pred_feats:
            self.weights[f] -= 1.0

    def train_perceptron(self, n_epochs):
        '''Trains the parser by running the averaged perceptron algorithm for n_epochs.''' 
        self.weights = np.zeros(self.features.n_feats)
        total = np.zeros(self.features.n_feats)
        for epoch in range(n_epochs):
            print "Epoch {0}".format(epoch+1),
            print "Train:",
            print "%.3f" % self.evalInstances(self.reader.train_instances,self.perceptron_update),
            total += self.weights   
            print "Dev:",

            #weight averaging
            old_weights = self.weights.copy()
            self.weights = total.copy() / (epoch + 1.0)
            print "%.3f" % self.evalInstances(self.reader.test_instances)
            
            #return the weights
            self.weights = old_weights

    def test(self):
        arr_heads_pred = []
        for instance in self.reader.loadInstances("data/deppars/english_test.conll"):
            feats = self.features.create_features(instance)
            scores = self.features.compute_scores(feats, self.weights)
            heads_pred = self.decoder.parse_nonproj(scores)
            #print [heads_pred, instance.heads]
            arr_heads_pred.append(heads_pred)
        self.writer.save(self.language, arr_heads_pred)
