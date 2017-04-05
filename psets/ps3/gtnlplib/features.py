from gtnlplib import constants

# Deliverable 1.1
def word_feats(words,y,y_prev,m):
    """This function should return at most two features:
    - (y,constants.CURR_WORD_FEAT,words[m])
    - (y,constants.OFFSET)

    Note! You need to handle the case where $m >= len(words)$. In this case, you should only output the offset feature. 

    :param words: list of word tokens
    :param m: index of current word
    :returns: dict of features, containing a single feature and a count of 1
    :rtype: dict

    """
    fv = dict()
    if (m < len(words)):
        fv[y, constants.CURR_WORD_FEAT,words[m]] = 1
    fv[y,constants.OFFSET] = 1

    return fv

# Deliverable 2.1
def word_suff_feats(words,y,y_prev,m):
    """This function should return all the features returned by word_feats,
    plus an additional feature for each token, indicating the final two characters.

    You may call word_feats in this function.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """
    new_feat = word_feats(words, y, y_prev, m)
    if m < len(words):
        word = words[m][-2:]
        new_feat[y,constants.SUFFIX_FEAT,word] = 1
    return new_feat

    
def word_neighbor_feats(words,y,y_prev,m):
    """compute features for the current word being tagged, its predecessor, and its successor.

    :param words: list of word tokens
    :param y: proposed tag for word m
    :param y_prev: proposed tag for word m-1 (ignored)
    :param m: index m
    :returns: dict of features
    :rtype: dict

    """

    # hint: use constants.PREV_WORD_FEAT and constants.NEXT_WORD_FEAT
    fv = dict()
    fv[y,constants.OFFSET] = 1
    if (m < len(words)):
        if m == 0:
            fv[y, constants.PREV_WORD_FEAT,constants.PRE_START_TOKEN] = 1
        else:
            fv[y, constants.PREV_WORD_FEAT, words[m-1]] = 1

        fv[y, constants.CURR_WORD_FEAT,words[m]] = 1

        if m == len(words)-1:
            fv[y,constants.NEXT_WORD_FEAT, constants.POST_END_TOKEN] = 1
        else:
            fv[y, constants.NEXT_WORD_FEAT, words[m+1]] = 1

    if m == len(words):
        fv[y, constants.PREV_WORD_FEAT, words[m-1]] = 1

    return fv


def word_feats_competitive_en(words,y,y_prev,m):
    wn_fv = word_neighbor_feats(words,y,y_prev,m)

    if m < len(words):
        word1 = words[m][-3:]
        word2 = words[m][:2]
        wn_fv[y,constants.SUFFIX_FEAT,word1] = 1
        wn_fv[y,constants.PREFIX_FEAT,word2] = 1
    return wn_fv

def word_feats_competitive_ja(words,y,y_prev,m):
    ws_fv = word_suff_feats(words,y,y_prev,m)
    
    if m < len(words):
        word2 = words[m][:2]
        wordlen = len(words[m])
        ws_fv[y,constants.LEN_FEAT,wordlen] = 1
        ws_fv[y,constants.PREFIX_FEAT,word2] = 1
    return ws_fv

def hmm_feats(words,y,y_prev,m):
    fv = dict()

    fv[y,constants.PREV_TAG_FEAT,y_prev] = 1
    if m < len(words):
        fv[(y,constants.CURR_WORD_FEAT,words[m])] = 1
    return fv

def hmm_feats_competitive_en(words,y,y_prev,m):
    fv = hmm_feats(words,y,y_prev,m)
    wn_fv = word_neighbor_feats(words,y,y_prev,m)
    if m < len(words):
        word1 = words[m][-3:]
        word2 = words[m][:2]
        wn_fv[y,constants.SUFFIX_FEAT,word1] = 1
        wn_fv[y,constants.PREFIX_FEAT,word2] = 1
    fv.update(wn_fv)
    return fv


def hmm_feats_competitive_ja(words,y,y_prev,m):
    fv = hmm_feats(words,y,y_prev,m)
    ws_fv = word_suff_feats(words,y,y_prev,m)
    if m < len(words):
        word2 = words[m][:2]
        wordlen = len(words[m])
        ws_fv[y,constants.PREFIX_FEAT,word2] = 1
        ws_fv[y,constants.LEN_FEAT,wordlen] = 0.5
    fv.update(ws_fv)
    return fv

