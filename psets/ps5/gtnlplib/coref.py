import os
import xmltodict
import re
import subprocess
from glob import glob
import numpy as np

# For students

## deliverable 1.1
def get_markables_for_entity(markables,entity):
    """Return a list of markables for a given entity string

    :param markables: list of all markables in the document
    :param entity: entity string
    :returns: list of markables corresponding to a given entity
    :rtype: list

    """
    raise NotImplementedError
    
    
## deliverable 1.2
def get_distances(markables, string):
    """Return a list of distances to antecedents```

    :param markables: list of markables in the document
    :param term: mention string
    :returns: list of integer distances
    :rtype: 

    """
    ants = get_true_antecedents(markables) #hint
    ## hide
    raise NotImplementedError
    
## Deliverable 2.1
def get_tp(pred_ant,markables):
    """Return a list of booleans, indicating whether an instance is a true positive or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans

    """
    raise NotImplementedError
    
## Deliverable 2.1
def get_fp(pred_ant,markables):
    """Return a list of booleans, indicating whether an instance is a false positive or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans

    """
    raise NotImplementedError

## Deliverable 2.1
def get_fn(pred_ant,markables):
    """Return a list of booleans, indicating whether an instance is a false negative or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans

    """
    raise NotImplementedError
    
def recall(pred_ant,markables):
    """Compute the recall, tp/(tp+fn)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float

    """
    tp = float(sum(get_tp(pred_ant,markables)))
    return tp/(tp + sum(get_fn(pred_ant,markables)))

def precision(pred_ant,markables):
    """Compute the recall, tp/(tp+fp)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float

    """
    tp = float(sum(get_tp(pred_ant,markables)))
    return tp/(tp + sum(get_fp(pred_ant,markables)))

def f1(pred_ant,markables):
    """Compute the f1 = 2*r*p/(r+p)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float

    """

    r = recall(pred_ant,markables)
    p = precision(pred_ant,markables)
    return 2 * r * p / (r + p)

def evaluate(resolver,markables):
    """Perform a complete evaluation for a coreference resolver

    :param resolver: function from markable list to antecedent list
    :param markables: list of markables
    :returns: f1, recall, precision
    :rtype: triple

    """
    sys_ant = resolver(markables)
    r = recall(sys_ant,markables)
    p = precision(sys_ant,markables)
    f = f1(sys_ant,markables)
    return f,r,p

def eval_on_dataset(resolver,markables_list):
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    for markables in markables_list:
        sys_ant = resolver(markables)
        tot_tp += sum(get_tp(sys_ant,markables))
        tot_fp += sum(get_fp(sys_ant,markables))
        tot_fn += sum(get_fn(sys_ant,markables))
    r = tot_tp / float(tot_tp + tot_fn)
    p = tot_tp / float(tot_tp + tot_fp)
    f = 2 * r * p / ( r + p )
    print eval_string(f,r,p)
    return f,r,p

eval_string = lambda f,r,p : "F: %.4f\tR: %.4f\tP:%.4f"%(f,r,p)

def write_predictions(resolver,markables_list,outfile):
    with open(outfile,'w') as fout:
        for i,markables in enumerate(markables_list):
            sys_ant = resolver(markables)
            for sys_ant_i in sys_ant:
                print >>fout, i, sys_ant_i

def eval_predictions(pred_file,markables):
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    with open(pred_file) as fin:
        for i,markables_i in enumerate(markables):
            sys_ant = []
            for j in range(len(markables_i)):
                line = fin.readline()
                parts = line.split()
                assert int(parts[0]) == i
                sys_ant.append(int(parts[1]))
            tot_tp += sum(get_tp(sys_ant,markables_i))
            tot_fp += sum(get_fp(sys_ant,markables_i))
            tot_fn += sum(get_fn(sys_ant,markables_i))
    r = tot_tp / float(tot_tp + tot_fn)
    p = tot_tp / float(tot_tp + tot_fp)
    f = 2 * r * p / ( r + p )
    print eval_string(f,r,p)
    return f,r,p

### reading data

def read_dataset(basedir,tagger=None,max_markables=300):
    markables_list = []
    words_list = []
    for filename in glob(os.path.join(basedir,"*")):
        markables,words = read_data(os.path.basename(filename).replace('_',' '),
                                    basedir=basedir,
                                    tagger=tagger,
                                    max_markables=max_markables)
        markables_list.append(markables)
        words_list.append(words)
    return markables_list, words_list

def read_data(page_name,
              basedir=os.path.join('data','tr'),
              max_markables=300,
              tagger = None):
    """For a given page name, parse it and return a list of markables and words

    Note! Markable word spans seem to be 1-indexed rather than 0-indexed

    :param page_name: name of the page to parse
    :param basedir: base directory containing wiki coref files
    :returns: list of markables, list of words
    :rtype: list, list

    """
    filedir = os.path.join(basedir,page_name.replace(' ','_'))
    #print filedir
    with open(os.path.join(filedir,page_name+'.txt')) as fin:
        words = [line.rstrip() for line in fin.readlines()]
        words = [word for word in words if len(word) > 0]
    suffix = '_coref_level_OntoNotesScheme.xml'

    def get_words_for_markable(markable,words):
        p = re.compile('word_([0-9]*)\.\.word_([0-9]*)')
        span_str = markable['@span']
        m = p.match(span_str)
        return words[int(m.group(1))-1:int(m.group(2))],m.group(1),m.group(2)

    with open(os.path.join(filedir,'Markables',page_name+suffix)) as fin:
        filedict = xmltodict.parse(fin.read())
        markables = []
        for markable in filedict['markables']['markable']:
            string,tok_start,tok_end = get_words_for_markable(markable,words)
            markables.append({'string':string,
                              'start_token':int(tok_start)-1,
                              'end_token':int(tok_end),
                              'entity':markable['@coref_class']})

    # this is apparently necessary
    if True:
        markables.sort(key =lambda m : (m['start_token'],m['end_token']))

        gaps = np.diff(np.array([markable['start_token'] for markable in markables]))
        assert gaps.min() >= 0
            
    markables = markables[:max_markables]

    
    # maybe make students add this part?
    if tagger is not None:
        tags = tagger(words)
        for markable in markables:
            markable['tags'] = [tag for word,tag in tags[markable['start_token']:markable['end_token']]]
    
    return markables,words

def get_entities(markables):
    """Given list of markables, return list of lists of mention indices (one list per entity)

    :param markables: list of markables, probably from getMarkablesAndWords
    :returns: list of list of mention indices 
    :rtype: list

    """

    entities = set([markable['entity'] for markable in markables])
    C_dict = {entity:[i for i,markable in enumerate(markables) if markable['entity']==entity] for entity in entities}
    return C_dict.values()

def get_true_antecedents(markables):
    """Given a list of markables (from getMarkablesAndWords), return list of antecedent indices.
    The antecedent is always the most recent mention in the cluster.

    :param markables: list of markables
    :returns: list of antecdent indices
    :rtype: list

    """
    last_mentions = dict()
    antecedents = []
    for idx,markable in enumerate(markables):
        if markable['entity'] in last_mentions:
            antecedents.append(last_mentions[markable['entity']])
        else:
            antecedents.append(idx)
        last_mentions[markable['entity']] = idx
    return antecedents


