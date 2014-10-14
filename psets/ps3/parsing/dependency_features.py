'''
Original code by Joao Graca and Andre Martins (2011-2012)
Modified by Jacob Eisenstein (2013) for Georgia Tech CS 4650/7650 NLP
'''
import sys
import numpy as np
from dependency_reader import *
from itertools import chain, combinations

class DependencyFeatures(object):
    '''
    Dependency features class
    '''
    def __init__(self):
        self.feat_dict = {}
        self.n_feats = 0

    def create_dictionary(self, instances):
        '''Creates dictionary of features (note: only uses supported features)'''
        self.feat_dict = {}
        self.n_feats = 0
        for instance in instances:
            nw = np.size(instance.words)-1
            heads = instance.heads
            for m in range(1, nw+1):
                h = heads[m]
                self.create_arc_features(instance, h, m, True)

        print "Number of features: {0}".format(self.n_feats)


    def create_features(self, instance):
        '''Creates arc features from an instance.'''
        nw = np.size(instance.words)-1
        feats = np.empty((nw+1, nw+1), dtype=object)
        for h in range(0,nw+1): 
            for m in range(1,nw+1):
                if h == m:
                    feats[h][m] = []
                    continue
                feats[h][m] = self.create_arc_features(instance, h, m)

        return feats

    def create_arc_features(self,instance,h,m,add=False):
        '''
        Create features for arc h-->m
        This is the function you should modify to do the project
        '''
        ff = []

        ## Example: 
        ## h - head pos, m - modifier pos
        f = self.getF((0,instance.pos[h], instance.pos[m]), add)
        ff.append(f)
        return(ff)

    def getF(self, feats, add=True):
        return self.lookup_fid(feats,add)

    def lookup_fid(self, fname, add=False):
        '''Looks up dictionary for feature ID.'''
        if not fname in self.feat_dict:
            if add:
                fid = self.n_feats
                self.n_feats += 1
                self.feat_dict[fname] = fid
                return fid
            else:
                return -1
        else:
            return self.feat_dict[fname]

    def compute_scores(self, feats, weights):
        '''
        Compute scores by taking the dot product between the feature and weight vector.
        Return numpy array of heads by modifiers
        ''' 
        nw = np.size(feats, 0) - 1
        scores = np.zeros((nw+1, nw+1))
        for h in range(nw+1):
            for m in range(nw+1):
                if feats[h][m] == None:
                    continue
                scores[h][m] = sum([weights[f] for f in feats[h][m] if f>=0])
        return scores


