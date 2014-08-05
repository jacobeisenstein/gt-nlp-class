import csv
from scipy.sparse import csr_matrix

def csv2csr(filename):
    word = []
    context = []
    count = []
    with open(filename,'rb') as infile:
        reader = csv.reader(infile)
        for row in reader:
            word.append(int(row[0]))
            context.append(int(row[1]))
            count.append(int(row[2]))
    return csr_matrix((count,(word,context)))

def readVocab(filename):
    vocab = []
    with open(filename,'rb') as vocabfile:
        for line in vocabfile:
            vocab.append(line.split()[0])
    index = dict(zip(range(0,len(vocab)),vocab)) #from numbers to words
    # inv_index = {i[1]:i[0] for i in index.items()} #from words to numbers
    inv_index = {}
    for i in index.iteritems():
        inv_index[i[1]] = i[0]
    return index,inv_index

