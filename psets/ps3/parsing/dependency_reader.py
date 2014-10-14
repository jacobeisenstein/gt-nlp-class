'''
Original code by Joao Graca and Andre Martins (2012)
Modified by Jacob Eisenstein (2013) for Georgia Tech CS 4650/7650 NLP
'''
import sys
import numpy as np
import os
from os import path


class Instance():
    '''
    Instance class
    '''
    def __init__(self):
        self.words = []
        self.pos = []
        self.heads = []


class DependencyReader():
    '''
    Dependency reader class
    '''
    def __init__(self):
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []

    def loadInstances(self,filename):
        instances = []
        inst = Instance()
        inst.words.append(self.word_dict["__START__"])
        inst.pos.append(self.pos_dict["__START__"])
        inst.heads.append(-1)
        with open(filename,'r') as conll_file:
            for line in conll_file:
                line = line.rstrip()
                if len(line) == 0:
                    instances.append(inst)
                    inst = Instance()
                    inst.words.append(self.word_dict["__START__"])
                    inst.pos.append(self.pos_dict["__START__"])
                    inst.heads.append(-1)
                    continue
                fields = line.split("\t")

                word = fields[1]
                pos = fields[3]
                head = int(fields[6])

                if word not in self.word_dict:
                    word_id = -1
                else:
                    word_id = self.word_dict[word]
                if pos not in self.pos_dict:
                    pos_id = -1
                else:
                    pos_id = self.pos_dict[pos]

                inst.words.append(word_id)
                inst.pos.append(pos_id)
                inst.heads.append(head)
        return(instances)

    def load(self, language):
        '''Loads training and test data for dependency parsing.''' 
        self.word_dict = {}
        self.pos_dict = {}
        self.train_instances = []
        self.test_instances = []
        base_deppars_dir = path.join(path.dirname(__file__),"..","data","deppars")
        print base_deppars_dir
        languages = ["danish","dutch","portuguese","english"]
        i = 0
        word_dict = {}
        pos_dict = {}
        feat_counts = {}
        if(language not in languages):
            print "Language does not exist: \"%s\": Available are: %s"%(language,languages)
            return

        ### Create alphabet from training data
        n_sents = 0
        n_toks = 0
        word_id = 0
        pos_id = 0
        conll_file = open(path.join(base_deppars_dir, language+"_train.conll"))

        self.word_dict["__START__"] = word_id # Start symbol
        word_id+=1
        self.word_dict["__STOP__"] = word_id # Stop symbol
        word_id+=1
        self.pos_dict["__START__"] = pos_id # Start symbol
        pos_id+=1
        self.pos_dict["__STOP__"] = pos_id # Stop symbol
        pos_id+=1

        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_sents+=1
                continue
            fields = line.split("\t")
            n_toks+=1
            word = fields[1]
            pos = fields[3]
            if word not in self.word_dict:
                self.word_dict[word] = word_id
                word_id+=1
            if pos not in self.pos_dict:
                self.pos_dict[pos] = pos_id
                pos_id+=1
        conll_file.close()

        print "Number of sentences: {0}".format(n_sents)
        print "Number of tokens: {0}".format(n_toks)
        print "Number of words: {0}".format(word_id)
        print "Number of pos: {0}".format(pos_id)
    

        ### Load training data
        self.train_instances = self.loadInstances(path.join(base_deppars_dir, language+"_train.conll"))
        self.test_instances = self.loadInstances(path.join(base_deppars_dir, language+"_dev.conll"))
