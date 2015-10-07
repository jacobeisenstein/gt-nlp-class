import operator
from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE

argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]
def classifierTagger(words,featfunc,weights,all_tags):
    """
    :param words: list of words
    :param features: function from lists of words and tags to list of features
    :param weights: defaultdict of weights
    :param all_tags: list of permissible tags
    :returns list of tags
    """
    out = []
    # your code
    return out



def evalTagger(tagger,outfilename,testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    alltags = set()
    for i,(words, tags) in enumerate(preproc.conllSeqGenerator(TRAIN_FILE)):
        for tag in tags:
            alltags.add(tag)
    with open(outfilename,'w') as outfile:
        for words,_ in preproc.conllSeqGenerator(testfile):
            pred_tags = tagger(words,alltags)
            for tag in pred_tags:
                print >>outfile, tag
            print >>outfile, ""
    return scorer.getConfusion(testfile,outfilename) #run the scorer on the prediction file
