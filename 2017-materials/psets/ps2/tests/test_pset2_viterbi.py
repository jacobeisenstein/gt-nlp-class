from nose.tools import with_setup, ok_, eq_, assert_almost_equals, nottest, assert_not_equal

from gtnlplib.constants import START_TAG, END_TAG, TRANS, EMIT
from gtnlplib import hmm, viterbi

def setup():
    global hand_weights
    hand_weights = {('NOUN','they',EMIT):-1,\
                    ('NOUN','can',EMIT):-3,\
                    ('NOUN','fish',EMIT):-3,\
                    ('VERB','they',EMIT):-11,\
                    ('VERB','can',EMIT):-2,\
                    ('VERB','fish',EMIT):-4,\
                    ('NOUN','NOUN',TRANS):-5,\
                    ('VERB','NOUN',TRANS):-2,\
                    (END_TAG,'NOUN',TRANS):-2,\
                    ('NOUN','VERB',TRANS):-1,\
                    ('VERB','VERB',TRANS):-3,\
                    (END_TAG,'VERB',TRANS):-3,\
                    ('NOUN',START_TAG,TRANS):-1,\
                    ('VERB',START_TAG,TRANS):-2}

# 3.2
def test_hmm_features():
    sentence = "They can can fish".split()
    f1 = hmm.hmm_features(sentence,'VERB','PRON',1)
    assert(('VERB','PRON',TRANS) in f1)
    assert(('PRON','VERB',TRANS) not in f1)
    assert(('VERB','can',EMIT) in f1)
    eq_(len(f1),2)
    
    f0 = hmm.hmm_features(sentence,'NOUN',START_TAG,0)
    assert(('NOUN',START_TAG,TRANS) in f0)
    assert(('NOUN','They',EMIT) in f0)
    assert((START_TAG,'They',EMIT) not in f0)
    eq_(len(f0),2)

    f4 = hmm.hmm_features(sentence,END_TAG,'VERB',4)
    assert((END_TAG,'VERB',TRANS) in f4)
    eq_(len(f4),1)


# 3.3
def test_viterbi_step_init():
    global hand_weights
    sentence = "they can can fish".split()

    v_0_noun = viterbi.viterbi_step('NOUN',0,sentence,hmm.hmm_features,hand_weights,{START_TAG:0})
    eq_(v_0_noun,(-2,START_TAG))
    
    v_0_verb = viterbi.viterbi_step('VERB',0,sentence,hmm.hmm_features,hand_weights,{START_TAG:0})
    eq_(v_0_verb,(-13,START_TAG))

    v_1_noun = viterbi.viterbi_step('NOUN',1,sentence,
                                   hmm.hmm_features,
                                   hand_weights,
                                   {'NOUN':-2,'VERB':-13})
    eq_(v_1_noun,(-10,'NOUN'))

    v_1_verb = viterbi.viterbi_step('VERB',1,sentence,
                                   hmm.hmm_features,
                                   hand_weights,
                                   {'NOUN':-2,'VERB':-13})
    eq_(v_1_verb,(-6,'NOUN'))

# 3.4
def test_build_trellis():
    global hand_weights

    sentence = "they can can fish".split()
    all_tags = ['NOUN','VERB']
    
    # modify the hand weights so you can't read off the answer to 3.1 :)
    hand_weights['NOUN','they',EMIT] = -2
    hand_weights['VERB','fish',EMIT] = -5
    hand_weights['VERB','VERB',TRANS] = -2

    trellis = viterbi.build_trellis(sentence,hmm.hmm_features,hand_weights,all_tags)

    eq_(len(trellis),4)
    eq_(len(trellis[-1]),2)
    eq_(trellis[-1]['VERB'],(-18,'VERB'))
    eq_(trellis[-2]['NOUN'],(-11,'VERB'))
    eq_(trellis[1]['VERB'],(-7,'NOUN'))
    eq_(trellis[1]['NOUN'],(-11,'NOUN'))

# 3.5
def test_viterbi():
    global hand_weights
    hand_weights['NOUN','they',EMIT] = -2
    hand_weights['VERB','fish',EMIT] = -5
    hand_weights['VERB','VERB',TRANS] = -2

    all_tags = ['NOUN','VERB']
    
    sentence1 = "they can can fish".split()
    tags,score = viterbi.viterbi_tagger(sentence1,hmm.hmm_features,hand_weights,all_tags)
    eq_(score,-17)
    eq_(tags,['NOUN','VERB','VERB','NOUN'])

    sentence2 = "they can can can can can fish".split()
    tags,score = viterbi.viterbi_tagger(sentence2,hmm.hmm_features,hand_weights,all_tags)
    eq_(score,-29)
    eq_(tags,['NOUN','VERB','VERB','VERB','VERB','VERB','NOUN'])
    
    
