import nltk
import string
import pandas as pd
from collections import Counter
from nltk import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')

def tokenize_and_downcase(string,vocab=None):
    """for a given string, corresponding to a document:
    - tokenize first by sentences and then by word
    - downcase each token
    - return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    # use nltk built-in function to break it into sentences

    bow = Counter()
    sents = nltk.sent_tokenize(string)
    for sent in sents:
        tempToken = nltk.word_tokenize(sent)
        bow += Counter([w.lower() for w in tempToken])

    return bow


### Helper code

def read_data(csvfile,labelname,preprocessor=lambda x : x):
    # note that use of utf-8 encoding to read the file
    df = pd.read_csv(csvfile,encoding='utf-8')
    return df[labelname].values,[preprocessor(string) for string in df['text'].values]

def get_corpus_counts(list_of_bags_of_words):
    counts = Counter()
    for bow in list_of_bags_of_words:
        for key,val in bow.iteritems():
            counts[key] += val
    return counts

### Secret bakeoff code
def custom_preproc(string1):
    """for a given string, corresponding to a document, tokenize first by sentences and then by word; downcase each token; return a Counter of tokens and frequencies.

    :param string: input document
    :returns: counter of tokens and frequencies
    :rtype: Counter

    """
    #sw = stopwords.words('english')
    bow = Counter()
    sents = nltk.sent_tokenize(string1)
    translate_table = dict((ord(char), None) for char in string.punctuation)   
    for sent in sents:
        tempToken = nltk.word_tokenize(sent.translate(translate_table))
        bow += Counter([w.lower() for w in tempToken])

    return bow
