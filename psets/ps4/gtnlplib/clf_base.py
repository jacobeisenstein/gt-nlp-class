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


def generateKaggleSubmission(weights,outfilename):
    with open(outfilename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Prediction'])
        writer.writeheader()

        # Test data is used for private leaderboard
        testData = dataIterator(TEST_FILE,test_mode=True)
        for i,(counts,_) in enumerate(testData):
            predictedLabel,_ = predict(counts,weights,ALL_LABELS)
            predictedIndex = ALL_LABELS.index(predictedLabel)
            writer.writerow({
                'Id': 'test-{}'.format(i),
                'Prediction': predictedIndex})

        # Dev data is used for public leaderboard
        devData = dataIterator(DEVKEY,test_mode=False)
        devCorrect = 0
        devTotal = 0
        for i,(counts,label) in enumerate(devData):
            devTotal += 1
            predictedLabel,_ = predict(counts,weights,ALL_LABELS)
            devCorrect += (predictedLabel == label)
            predictedIndex = ALL_LABELS.index(predictedLabel)
            writer.writerow({
                'Id': 'dev-{}'.format(i),
                'Prediction': predictedIndex})
    
    devAccuracy = float(devCorrect) / devTotal
    print 'Dev accuracy is ', devAccuracy, '({} correct of {})'.format(devCorrect, devTotal)
    print 'Kaggle submission saved to', outfilename, ('. Sanity check: '
        'public leaderboard accuracy should be '), devAccuracy, 'on submission.'

