import sys
from collections import defaultdict

#scoring script for text classification. 
#first argument should be a key file, containing space-separated filename and label
#second argument should be a response file, containing just the predicted label

def main():
    key = sys.argv[1]
    response = sys.argv[2]
    counts = getConfusion(key,response)
    printScoreMessage(counts)

def getConfusion(keyfilename,responsefilename):
    counts = defaultdict(int)
    with open(keyfilename,'r') as keyfile:
        with open(responsefilename,'r') as resfile:
            for keyline in keyfile:
                keyline = keyline.split(' ')[-1].rstrip()
                resline = resfile.readline().rstrip()
                counts[tuple((keyline,resline))] += 1
    return(counts)

def accuracy(counts):
    return sum([y for x,y in counts.items() if x[0] == x[1]]) / float(sum(counts.values()))

def printScoreMessage(counts):
    true_pos = 0
    total = 0

    keyclasses = set([x[0] for x in counts.keys()])
    resclasses = set([x[1] for x in counts.keys()])
    print "%d classes in key: %s" % (len(keyclasses),keyclasses)
    print "%d classes in response: %s" % (len(resclasses),resclasses)
    print "confusion matrix"
    print "key\\response:\t"+"\t".join(resclasses)
    for i,keyclass in enumerate(keyclasses):
        print keyclass+"\t\t",
        for j,resclass in enumerate(resclasses):
            c = counts[tuple((keyclass,resclass))]
            #countarr[i,j] = c
            print "{}\t".format(c),
            total += float(c)
            if resclass==keyclass:
                true_pos += float(c)
        print ""
    print "----------------"
    print "accuracy: %.4f = %d/%d\n" % (true_pos / total, true_pos,total)


if __name__ == "__main__":
    main()

