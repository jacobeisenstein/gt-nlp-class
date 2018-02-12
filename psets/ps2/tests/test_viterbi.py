from nose.tools import with_setup, ok_, eq_, assert_almost_equal, nottest, assert_not_equal

from gtnlplib.constants import START_TAG, END_TAG, UNK
from gtnlplib import hmm, viterbi
import torch
import numpy as np

def setup():
    global nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix, ix_to_tag, all_tags
    nb_weights={('NOUN','they'):-1,\
            ('NOUN','can'):-3,\
            ('NOUN','fish'):-3,\
            ('VERB','they'):-11,\
            ('VERB','can'):-2,\
            ('VERB','fish'):-4,}
    hmm_trans_weights={('NOUN','NOUN'):-5,\
                       ('VERB','NOUN'):-2,\
                       (END_TAG,'NOUN'):-2,\
                       ('NOUN','VERB'):-1,\
                       ('VERB','VERB'):-3,\
                       (END_TAG,'VERB'):-3,\
                       ('NOUN',START_TAG):-1,\
                       ('VERB',START_TAG):-2}
    vocab = ['they','can','fish',UNK]
    word_to_ix={'they':0, 'can':1, 'fish':2, UNK:3}
    tag_to_ix = {START_TAG:0, 'NOUN':1, 'VERB':2, END_TAG:3}
    ix_to_tag = {0:START_TAG, 1:'NOUN', 2:'VERB', 3:END_TAG}
    all_tags = [START_TAG, 'NOUN', 'VERB',END_TAG]


# 3.2
def test_compute_hmm_weights_variables():
    global nb_weights, hmm_trans_weights, vocab, word_to_ix, tag_to_ix
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights, \
                                                                         vocab, word_to_ix, tag_to_ix)
    
    eq_(emission_probs[0][0].data.numpy(),-np.inf)
    eq_(emission_probs[0][1].data.numpy(),-1)
    eq_(emission_probs[0][2].data.numpy(),-11)
    eq_(emission_probs[2][1].data.numpy(),-3)
    eq_(emission_probs[2][2].data.numpy(),-4)
    
    eq_(tag_transition_probs[0][0].data.numpy(),-np.inf)
    eq_(tag_transition_probs[1][1].data.numpy(),-5)
    eq_(tag_transition_probs[2][2].data.numpy(),-3)
    eq_(tag_transition_probs[2][3].data.numpy(),-np.inf)
    eq_(tag_transition_probs[3][3].data.numpy(),-np.inf)
    
    

# 3.3
def test_viterbi_step_init():
    global nb_weights, hmm_trans_weights, tag_to_ix, all_tags, vocab, word_to_ix
    
    sentence = "they can can fish".split()
    
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0 #setting all the score to START_TAG
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32)))
    
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights, \
                                                                         vocab, word_to_ix, tag_to_ix)
    
    viterbivars, bptrs = viterbi.viterbi_step(all_tags, tag_to_ix, 
                                          emission_probs[0], 
                                          tag_transition_probs,
                                          prev_scores)
    
    eq_(viterbivars[1].data.numpy(),-2)
    eq_(viterbivars[2].data.numpy(),-13)
    eq_(bptrs[1],0)
    eq_(bptrs[2],0)
    eq_(bptrs[3],0)
    
    
    prev_scores = torch.autograd.Variable(torch.from_numpy(np.array([-np.inf, -2, -13, -np.inf]).astype(np.float32))) 
    viterbivars, bptrs = viterbi.viterbi_step(all_tags, tag_to_ix,
                                              emission_probs[1],
                                              tag_transition_probs,
                                              prev_scores)
    
    eq_(viterbivars[1].data.numpy(),-10)
    eq_(viterbivars[2].data.numpy(),-6)
    eq_(bptrs[1],1)
    eq_(bptrs[2],1)
    eq_(bptrs[3],0)

#3.4a
def test_trellis_score():
    global nb_weights, hmm_trans_weights, tag_to_ix, all_tags, vocab, word_to_ix
    
    sentence = "they can can fish".split()
    
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0 #setting all the score to START_TAG
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32)))
    
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights,\
                                                                         vocab, word_to_ix, tag_to_ix)
    
    path_score, best_path = viterbi.build_trellis(all_tags, 
                                                  tag_to_ix, 
                                                  [ emission_probs[word_to_ix[w]] for w in sentence], 
                                                  tag_transition_probs)
    
    eq_(path_score.data.numpy(),-17)
    
    
    sentence = "they can can can can fish".split()
    path_score, best_path = viterbi.build_trellis(all_tags, 
                                                  tag_to_ix, 
                                                  [ emission_probs[word_to_ix[w]] for w in sentence], 
                                                  tag_transition_probs)
    
    eq_(path_score.data.numpy(),-25.)
    
    

# 3.4b
def test_build_trellis():
    global nb_weights, hmm_trans_weights, tag_to_ix, all_tags, vocab, word_to_ix
    
    sentence = "they can can fish".split()
    
    initial_vec = np.full((1,len(all_tags)),-np.inf)
    initial_vec[0][tag_to_ix[START_TAG]] = 0 #setting all the score to START_TAG
    prev_scores = torch.autograd.Variable(torch.from_numpy(initial_vec.astype(np.float32)))
    
    emission_probs, tag_transition_probs = hmm.compute_weights_variables(nb_weights, hmm_trans_weights,\
                                                                         vocab, word_to_ix, tag_to_ix)
    
    path_score, best_path = viterbi.build_trellis(all_tags, 
                                                  tag_to_ix, 
                                                  [ emission_probs[word_to_ix[w]] for w in sentence], 
                                                  tag_transition_probs)
    
    eq_(best_path,['NOUN', 'VERB', 'VERB', 'NOUN'])
    
    
    sentence = "they can can can can fish".split()
    path_score, best_path = viterbi.build_trellis(all_tags, 
                                                  tag_to_ix, 
                                                  [ emission_probs[word_to_ix[w]] for w in sentence], 
                                                  tag_transition_probs)
    
    eq_(best_path,['NOUN', 'VERB', 'VERB', 'NOUN', 'VERB', 'NOUN'])

