import nltk
import random

#read training file, read grammar, try to parse each sentence, return number of sentences analyzed and number of analyses per sentence

def doNothing(words,tags):
    return words,tags

def conllSeqGenerator(input_file):
    """ return an instance generator for a filename

    The generator yields lists of words and tags.  For test data, the tags
    may be unknown.  For usage, see trainClassifier and applyClassifier below.

    """
    cur_words = []
    cur_tags = []
    with open(input_file) as instances:
        for line in instances:
            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
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
            yield cur_words,cur_tags



def parseTags(tags,parser):
    trees = []
    try:
        trees = parser.nbest_parse(tags)
    except:
        pass
    return(trees)

def getShuffledTags(tags):
    out = list(tags) 
    random.shuffle(out)
    return out


def evalParser(cfg,input_file="oct27.train",debug=False,max_len=10,preprocessor=doNothing,seed=0):
    random.seed(seed)
    grammar = nltk.data.load(cfg,cache=False)
    parser = nltk.ChartParser(grammar)
    tp = 0.0 # True positive
    fp = 0.0 # False positive
    fn = 0.0 # False negative
    num_parses = 0.0
    unparsed = []
    for words,tags in conllSeqGenerator(input_file):
        words,tags = preprocessor(words,tags)
        if len(words) < max_len:
            trees = parseTags(tags,parser)
            if len(trees) == 0: fn += 1
            else: 
                tp += 1
                num_parses += len(trees)
            # 
            shuftags = getShuffledTags(tags)
            trees = parseTags(shuftags,parser)
            if len(trees) > 0: fp += 1
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    fmeasure = 2 * recall * precision / (recall + precision)
    quality = {'f-measure': fmeasure, 'recall': recall, 'precision' : precision, 
               'parses-per-sent': num_parses / tp}
    return quality
        
        
            
