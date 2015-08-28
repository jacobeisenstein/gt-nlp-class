from nltk.tokenize import word_tokenize, sent_tokenize
from collections import defaultdict, Counter
import os.path
from itertools import chain
from constants import OFFSET

def docsToBOWs(keyfile):
    if os.path.exists (keyfile):
        dirname = os.path.dirname (keyfile)
    with open(keyfile,'r') as keys:
        with open(keyfile.replace('.key','.bow'),'w') as outfile:
            for keyline in keys:
                dataloc = keyline.rstrip().split(' ')[0]
                dataloc = os.path.join (dirname, dataloc)
                fcounts = defaultdict(int) 
                with open(dataloc,'r') as infile:
                    for line in infile: 
                        decoded = line.decode('ascii','ignore')
                        # YOUR CODE HERE
                for word,count in fcounts.items():
                    print >>outfile,"{}:{}".format(word,count), #write the word and its count to a line
                print >>outfile,""

def dataIterator(keyfile,test_mode=False):
    """
    The dataIterator above incrementally re-reads the keyfile and BOW file every time you call it. 
    This is a good idea if you have huge data that won't fit in memory, but the file I/O involves some overhead.
    If you want, you can write a second dataIterator that iterates across data stored in memory, which
    will be faster.
    """
    with open(keyfile.replace('key','bow'),'r') as bows:
        with open(keyfile,'r') as keys:
            for keyline in keys:
                if test_mode:
                    label = 'UNK'
                else:
                    textloc,label = keyline.rstrip().split(' ')
                fcounts = {word:int(count) for word,count in\
                           [x.split(':') for x in bows.readline().rstrip().split(' ')]}
                fcounts[OFFSET] = 1
                yield fcounts,label

def getAllCounts(datait):
    allcounts = Counter()
    for fcounts, _ in datait:
        allcounts += Counter(fcounts)
    return allcounts

def loadInstances (trainkey, devkey):
    all_tr_insts = []
    for inst, label in dataIterator (trainkey):
        all_tr_insts.append ((inst, label))
    all_dev_insts = []
    for inst, label in dataIterator (devkey):
        all_dev_insts.append ((inst, label))
    return all_tr_insts, all_dev_insts

def getCountsAndKeys (trainkey):
    counts = defaultdict(lambda : Counter()) 
    class_counts = defaultdict(int) 
    for words,label in dataIterator(trainkey):
        counts[label] += Counter(words)
        class_counts[label] += 1
    allkeys = set(chain.from_iterable(count.keys() for count in counts.values()))
    return counts, class_counts, allkeys
