from nose.tools import eq_, assert_almost_equals
from gtnlplib.preproc import *
from gtnlplib.preproc_metrics import *

def setup_module():
    global tr_tok, corpus_counts
    _,tr_tok = read_data('reddit-train.csv',
                         'subreddit',
                         preprocessor=tokenize_and_downcase)
    corpus_counts = get_corpus_counts(tr_tok)

    global dv_tok, corpus_counts_dv
    _,dv_tok = read_data('reddit-dev.csv',
                         'subreddit',
                         preprocessor=tokenize_and_downcase)
    corpus_counts_dv = get_corpus_counts(dv_tok)
    
    
    global te_tok, corpus_counts_te
    _,te_tok = read_data('reddit-test.csv',
                         'subreddit',
                         preprocessor=tokenize_and_downcase)
    corpus_counts_te = get_corpus_counts(te_tok)

# 0.5 points    
def test_preproc_d1_1():
    global tr_tok

    # public
    eq_(len(tr_tok[268]),116)
    eq_(len(tr_tok[13]),75)

# 0.1 points
def test_preproc_d1_2a():
    global corpus_counts

    # public
    ttr_tr = get_token_type_ratio(corpus_counts)
    assert_almost_equals(ttr_tr,19.6711,places=2)
    
# 0.1 points
def test_preproc_d1_2b():
    global corpus_counts, corpus_counts_te

    # public
    eq_(type_frequency(corpus_counts,1),14134)
    eq_(type_frequency(corpus_counts,10),263)

# 0.1 points
def test_preproc_d1_2c():
    global corpus_counts, corpus_counts_dv, corpus_counts_te

    # public
    eq_(unseen_types(corpus_counts, corpus_counts_dv),1737)


