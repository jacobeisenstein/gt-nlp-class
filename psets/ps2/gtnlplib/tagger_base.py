from gtnlplib import scorer
from gtnlplib import preproc
from gtnlplib import clf_base # call clf_base.predict
from gtnlplib.constants import DEV_FILE, OFFSET, TRAIN_FILE
import operator

def make_classifier_tagger(weights):
    """

    :param weights: a defaultdict of classifier weights
    :returns: a function that takes a list of words, and a list of candidate tags, and returns tags for all words
    :rtype: function

    """
    # inner function, which is what you return    
    def classify(words, all_tags):
        """This nested function should return a list of tags, computed using a classifier with the weights passed as arguments to make_classifier_tagger

        :param words: list of words
        :param all_tags: all possible tags
        :returns: list of tags
        :rtype: list

        """
        return None
    return classify

def apply_tagger(tagger,outfilename,all_tags=None,trainfile=TRAIN_FILE,testfile=DEV_FILE):
    if all_tags is None:
       all_tags = set()

       # this is slow
       for i,(words, tags) in enumerate(preproc.conll_seq_generator(trainfile)):
           for tag in tags:
               all_tags.add(tag)
            
    with open(outfilename,'w') as outfile:
        for words,_ in preproc.conll_seq_generator(testfile):
            pred_tags = tagger(words,all_tags)
            for i,tag in enumerate(pred_tags):
                print >>outfile, tag
            print >>outfile, ""

def eval_tagger(tagger,outfilename,all_tags=None,trainfile=TRAIN_FILE,testfile=DEV_FILE):
    """Calculate confusion_matrix for a given tagger

    Parameters:
    tagger -- Function mapping (words, possible_tags) to an optimal
              sequence of tags for the words
    outfilename -- Filename to write tagger predictions to
    testfile -- (optional) Filename containing true labels

    Returns:
    confusion_matrix -- dict of occurences of (true_label, pred_label)
    """
    apply_tagger(tagger,outfilename,all_tags,trainfile,testfile)
    return scorer.get_confusion(testfile,outfilename) #run the scorer on the prediction file
