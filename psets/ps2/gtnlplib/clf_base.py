import csv
import gtnlplib.scorer
import operator
from gtnlplib.preproc import dataIterator
from gtnlplib.constants import ALL_LABELS
from gtnlplib.constants import TESTKEY
from gtnlplib.constants import DEVKEY

# use this to find the highest-scoring label
argmax = lambda x : max(x.iteritems(),key=operator.itemgetter(1))[0]

# hide inner code
# should return two outputs: the highest-scoring label, and the scores for all labels
def predict(instance,weights,labels):
    pass
    #YOUR CODE HERE
    #return argmax(scores),scores

def evalClassifier(weights,outfilename,testfile,test_mode=False):    
    with open(outfilename,'w') as outfile:
        for counts,label in dataIterator(testfile,test_mode): #iterate through eval set
            print >>outfile, predict(counts,weights,ALL_LABELS)[0] #print prediction to file
    if test_mode:
        return
    else:
        return gtnlplib.scorer.getConfusion(testfile,outfilename) #run the scorer on the prediction file

def generateKaggleSubmission(weights,outfilename):
    with open(outfilename, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'Prediction'])
        writer.writeheader()

        # Test data is used for private leaderboard
        testData = dataIterator(TESTKEY,test_mode=True)
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

