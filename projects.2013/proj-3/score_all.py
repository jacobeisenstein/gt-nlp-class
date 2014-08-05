from glob import glob
import fnmatch
import os

# import argparse


# #use argparse
# args = argparse.ArgumentParser()
# args.add_argument('preds',default='
# for predfile in 

keyfile = 'english_test.conll'

keylines = []
with open(keyfile,'r') as fin:
    for line in fin:
        parts = line.split()
        if len(parts) > 1:
            keylines.append(parts[6].rstrip())

#print keylines

for root, dirnames, filenames in os.walk('.'):
    for filename in fnmatch.filter(filenames, '*.pred'):
        fullfilename = os.path.join(root, filename)
        with open(fullfilename,'r') as fin:
            num_correct = 0
            i = 0
            for line in fin:
                parts = line.split()
#                print parts
                if len(parts) > 1:
                    if parts[6].rstrip() == keylines[i]:
                        num_correct += 1
                    i += 1
        print num_correct/float(i),filename
          
