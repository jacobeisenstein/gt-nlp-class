import os
import xmltodict
import re
import subprocess
from . import bcm_evaluate
from collections import namedtuple
from glob import glob
import numpy as np

Markable = namedtuple('Markable', ['string', 'entity', 'start_token', 'end_token', 'tags'])
Document = namedtuple('Document', ['clusters', 'gold', 'mention_to_gold', 'mention_to_cluster'])

## deliverable 1.1
def get_markables_for_entity(markables, entity):
    '''
    Return a list of markable strings for a given entity string

    :param markables: list of all markables in the document
    :param entity: entity string
    :returns: all markables corresponding to a given entity
    :rtype: list
    '''
    raise NotImplementedError


## deliverable 1.2
def get_distances(markables, string):
    '''
    Return a list of distances to antecedents

    :param markables: list of markables in the document
    :param term: mention string
    :returns: integer distances
    :rtype: list
    '''
    raise NotImplementedError

## Deliverable 2.1
def get_tp(pred_ant, markables):
    '''
    Return a list of booleans, indicating whether an instance is a true positive or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans
    '''
    raise NotImplementedError

## Deliverable 2.1
def get_fp(pred_ant, markables):
    '''
    Return a list of booleans, indicating whether an instance is a false positive or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans
    '''
    raise NotImplementedError

## Deliverable 2.1
def get_fn(pred_ant, markables):
    '''
    Return a list of booleans, indicating whether an instance is a false negative or not

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :returns: list of booleans
    '''
    raise NotImplementedError

def recall(pred_ant, markables):
    '''
    Compute the recall, tp/(tp+fn)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float
    '''
    tp = float(sum(get_tp(pred_ant, markables)))
    return tp/(tp + sum(get_fn(pred_ant, markables)))

def precision(pred_ant, markables):
    '''
    Compute the recall, tp/(tp+fp)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float
    '''
    tp = float(sum(get_tp(pred_ant, markables)))
    return tp/(tp + sum(get_fp(pred_ant, markables)))

def f1(pred_ant, markables):
    '''
    Compute F1 = 2*r*p/(r+p)

    :param pred_ant: predicted antecedent sequence
    :param markables: list of markables
    :rtype: float
    '''

    r = recall(pred_ant, markables)
    p = precision(pred_ant, markables)
    return 2 * r * p / (r + p)

def evaluate_f(resolver, markables):
    '''
    Perform a complete F-score evaluation for a coreference resolver

    :param resolver: function from markable list to antecedent list
    :param markables: list of markables
    :returns: f1, recall, precision
    :rtype: triple
    '''
    sys_ant = resolver(markables)
    r = recall(sys_ant, markables)
    p = precision(sys_ant, markables)
    f = f1(sys_ant, markables)
    return f, r, p

def eval_on_dataset(resolver, markables_list):
    tot_tp = 0
    tot_fp = 0
    tot_fn = 0
    for markables in markables_list:
        sys_ant = resolver(markables)
        tot_tp += sum(get_tp(sys_ant, markables))
        tot_fp += sum(get_fp(sys_ant, markables))
        tot_fn += sum(get_fn(sys_ant, markables))
    r = 0 if tot_tp == 0 else tot_tp / (tot_tp + tot_fn)
    p = 0 if tot_tp == 0 else tot_tp / (tot_tp + tot_fp)
    f = 0 if r == 0 or p == 0 else 2 * r * p / ( r + p )
    print(eval_string(f, r, p))
    return f, r, p

eval_string = lambda f, r, p : f'F: {f:.4f}\tR: {r:.4f}\tP:{p:.4f}'

def write_predictions(resolver,markables_list,outfile):
    with open(outfile,'w') as fout:
        for i,markables in enumerate(markables_list):
            sys_ant = resolver(markables)
            for sys_ant_i in sys_ant:
                print(i, sys_ant_i, file=fout)

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
    print(eval_string(f,r,p))
    return f,r,p

### reading data

def read_dataset(basedir, tagger=None, max_markables=300):
    '''
    Read entire directory and return as list of objects as returned from read_data()
    See read_data() for parameter descriptions
    '''
    markables_list = []
    words_list = []
    for filename in sorted(glob(os.path.join(basedir,"*"))):
        markables,words = read_data(os.path.basename(filename),
                                    basedir=basedir,
                                    tagger=tagger,
                                    max_markables=max_markables)
        markables_list.append(markables)
        words_list.append(words)
    return markables_list, words_list

def read_data(page_name,
              basedir=os.path.join('data','wsj','tr'),
              max_markables=300,
              tagger = None):
    '''
    For a given page name, parse it and return a list of markables and words

    :param page_name: name of the page to parse
    :param basedir: base directory containing wiki coref files
    :param max_markables: limit for number of markables to take from each document
                          (not necessary in our dataset)
    :returns: list of markables, list of words
    :rtype: list, list
    '''

    filename = os.path.join(basedir, page_name)
    with open(filename, 'rb') as fin:
        parsed_data = xmltodict.parse(fin)
    
    tokens = parsed_data['story']['rep'][2]['desc']
    xmarkables = parsed_data['story']['rep'][3]['desc']
    annotations = parsed_data['story']['rep'][4]['desc']
    if type(annotations) != list: # workaround for hidden test files
        annotations = [annotations]
    
    token_to_id = {t['@id']:i for i,t in enumerate(tokens)}
    token_dict = {t['@id']:t['#text'] for t in tokens}
    words = [t['#text'] for t in tokens]
    
    if tagger is not None:
        tags = [x[1] for x in tagger(words)]
    else:
        tags = ['NULL'] * len(words)

    ann_dict = {}
    for a in annotations:
        marks = a['#text'].split('|')[-1].split(',')
        ann_dict.update({i:a['@id'] for i in marks})
    
    def markify(xmlmark):
        tok_list = []
        for strech in xmlmark['#text'].split(','):
            tok_list.extend(strech.split('~'))
        start_tok = token_to_id[tok_list[0]]
        end_tok = token_to_id[tok_list[-1]] + 1
        return Markable([token_dict[i] for i in tok_list],\
               'set_{}'.format(ann_dict[xmlmark['@id']]),
               start_tok,
               end_tok,
               tags[start_tok:end_tok])
    
    markables = sorted([markify(c) for c in xmarkables], key = lambda x: x.start_token)

    markables = markables[:max_markables]

    return markables, words    

### markable/entity transformations

def get_entities(markables):
    '''
    Given list of markables, return list of lists of mention indices (one list per entity)

    :param markables: list of markables, probably from getMarkablesAndWords
    :returns: list of list of mention indices
    :rtype: list
    '''

    entities = set([markable.entity for markable in markables])
    C_dict = {entity:[i for i, markable in enumerate(markables) if markable.entity==entity] for entity in entities}
    return list(C_dict.values())

def markables_to_entities(markables, antecedents):
    m2e = dict()
    e2m = dict()
    for i,(m_i,ant_i) in enumerate(zip(markables, antecedents)):
        if i == ant_i:
            m2e[i] = len(list(e2m.keys()))
            e2m[len(list(e2m.keys()))] = [i]
        else:
            m2e[i] = m2e[ant_i]
            e2m[m2e[ant_i]].append(i)
    return m2e, e2m

def get_true_antecedents(markables):
    '''
    Given a list of markables, return list of antecedent indices.
    The antecedent is always the most recent mention in the cluster.

    :param markables: list of markables
    :returns: list of antecdent indices
    :rtype: list
    '''
    last_mentions = dict()
    antecedents = []
    for idx,markable in enumerate(markables):
        if markable.entity in last_mentions:
            antecedents.append(last_mentions[markable.entity])
        else:
            antecedents.append(idx)
        last_mentions[markable.entity] = idx
    return antecedents

### Standard metrics
    
def docify(mbls, pred_ants):
    '''
    Prepare a single document for evaluate using B-Cubed, CEAF, MUC
    :param mbls: list of markables
    :param pred_ants: predicted antecedent for each markable
    '''
    gold_ants = get_true_antecedents(mbls)
    gold_m2e, gold_e2m = markables_to_entities(mbls, gold_ants)
    pred_m2e, pred_e2m = markables_to_entities(mbls, pred_ants)
    return Document(pred_e2m.values(), gold_e2m.values(), gold_m2e, pred_m2e)
    
def evaluate_bcm(all_markables, all_predictions):
    '''
    Evaluate document corpus in terms of B-Cubed, CEAF, MUC F-1 scores
    '''
    b3_eval = bcm_evaluate.Evaluator(bcm_evaluate.b_cubed)
    ceaf_eval = bcm_evaluate.Evaluator(bcm_evaluate.ceafe)
    muc_eval = bcm_evaluate.Evaluator(bcm_evaluate.muc)
    
    all_docs = [docify(x,y) for x,y in zip(all_markables, all_predictions)]
    for d in all_docs:
        b3_eval.update(d)
        ceaf_eval.update(d)
        muc_eval.update(d)

    return b3_eval.get_f1(), ceaf_eval.get_f1(), muc_eval.get_f1()