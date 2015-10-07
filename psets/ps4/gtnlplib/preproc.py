from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import os.path
from itertools import chain
from gtnlplib.constants import OFFSET, UNKNOWN
"""
    Data processing code
"""

def getAllTags(input_file):
    """Return unique set of tags in the conll file"""
    alltags = set([])
    for _, tags in conllSeqGenerator(input_file):
        for tag in tags:
            alltags.add(tag)
    return alltags

def getAllData(input_file):
    tr_all = []
    for i,(words,tags) in enumerate(conllSeqGenerator(input_file)):
        tr_all.append((words,tags))
    return tr_all

def conllSeqGenerator(input_file,max_insts=1000000):
    """ return an instance generator for a filename
        
        The generator yields lists of words and tags.
        """
    cur_words = []
    cur_tags = []
    num_insts = 0
    with open(input_file) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
                    num_insts += 1
                    yield cur_words,cur_tags
                    cur_words = []
                    cur_tags = []
            else:
                parts = line.rstrip().split()
                cur_words.append(parts[0])
                if len(parts)>1:
                    cur_tags.append(parts[1])
                else: cur_tags.append(unk)
        if len(cur_words)>0:
            num_insts += 1
            yield cur_words,cur_tags
