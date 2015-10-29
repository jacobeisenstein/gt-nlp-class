'''
Original code by Joao Graca and Andre Martins (2012)
Modified by Jacob Eisenstein (2013) for Georgia Tech CS 4650/7650 NLP
'''
import sys
import numpy as np
import os
from os import path

class DependencyWriter():
    '''
    Dependency writer class
    '''
    def __init__(self):
        pass

    def saveToFile (self, inputFile, outputFile, heads_pred):
        ''' Saves predicted dependency trees in output file '''
        i = 0
        word_dict = {}
        pos_dict = {}
        feat_counts = {}

        n_toks = 0
        n_sents = 0

        conll_file = open(inputFile)
        conll_file_out = open(outputFile, 'w')
        for line in conll_file:
            line = line.rstrip()
            if len(line) == 0:
                n_toks = 0
                n_sents+=1
                conll_file_out.write("\n")
                continue
            fields = line.split("\t")

            fields[6] = "{0}".format(heads_pred[n_sents][n_toks + 1])
            line_out = "\t".join(fields)
            n_toks+=1

            conll_file_out.write(line_out)
            conll_file_out.write("\n")

        conll_file_out.close()
        conll_file.close()
