import nltk
import random

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
        if float(nltk.__version__[0]) >= 3:
            tree_gen = parser.parse(tags)
            trees = [tree for tree in tree_gen]
        else:
            trees = parser.nbest_parse(tags)
    except:
        pass
    return trees

def evalParser(cfg,
               input_file="oct27.clean.train",
               debug=False,
               show_fns=False,
               show_fps=False,
               max_len=10,
               preprocessor=lambda words,tags : (words,tags),
               seed=0,
               num_neg = 5):
    random.seed(seed)
    grammar = nltk.data.load(cfg,cache=False)
    parser = nltk.ChartParser(grammar)
    tp = 0.0 # True positive
    fp = 0.0 # False positive
    fn = 0.0 # False negative
    num_parses = 0.0
    unparsed = []
    for words,tags in conllSeqGenerator(input_file):
        words_post,tags_post = preprocessor(words,tags)
        if len(words) < max_len:
            trees = parseTags(tags_post,parser)
            if len(trees) == 0: 
                fn += 1
                if show_fns:
                    print "No parse:",words_post,tags_post
            else: 
                tp += 1
                num_parses += len(trees)
            # 
            for _ in xrange(num_neg):
                pairs = zip(words,tags)
                random.shuffle(pairs)
                words_post,tags_post = preprocessor([pair[0] for pair in pairs],[pair[1] for pair in pairs])
                trees = parseTags(tags_post,parser)
                if len(trees) > 0: 
                    fp += 1
                    if show_fps:
                        print "False parse:",words_post,tags_post

    recall = tp / (tp + fn)
    precision = tp / (tp + fp + 1e-6)
    fmeasure = 2 * recall * precision / (recall + precision + 1e-6)
    quality = {'f-measure': fmeasure, 'recall': recall, 'precision' : precision, 
               'parses-per-sent': num_parses / (1e-5+tp)}
    return quality
        
