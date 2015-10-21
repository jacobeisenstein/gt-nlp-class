import csv
import gtnlplib.scorer
import operator
from gtnlplib.constants import DEV_FILE
from gtnlplib.constants import TRAIN_FILE
from gtnlplib.constants import TEST_FILE
from gtnlplib import preproc
from gtnlplib import scorer
# use this to find the highest-scoring label
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]


def evalTagger(tagger,outfilename,testfile=DEV_FILE):
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


def generateKaggleSubmission(tagger,outfilename):
    with open(outfilename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Prediction'])
        writer.writeheader()

        alltags = set()
        for i,(words, tags) in enumerate(preproc.conllSeqGenerator(TEST_FILE)):
            for tag in tags:
                alltags.add(tag)

        i=0
        for words,_ in preproc.conllSeqGenerator(TEST_FILE):
            pred_tags = tagger(words,alltags)
            if isinstance(pred_tags, tuple):
                pred_tags = pred_tags[0] 
            for tag in pred_tags:
                writer.writerow({
                    'Id': 'test-{}'.format(i),
                    'Prediction':tag})
                i+=1
        i=0 
        for words,_ in preproc.conllSeqGenerator(DEV_FILE):
            pred_tags = tagger(words,alltags)
            if isinstance(pred_tags, tuple):
                pred_tags = pred_tags[0]
            for tag in pred_tags:
                # print >>outfile, tag
                writer.writerow({
                    'Id': 'dev-{}'.format(i),
                    'Prediction':tag})
                i+=1