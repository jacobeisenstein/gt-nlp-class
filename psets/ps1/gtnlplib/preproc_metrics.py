def get_token_type_ratio(vocabulary):
    """compute the ratio of tokens to types

    :param vocabulary: a Counter of words and their frequencies
    :returns: ratio of tokens to types
    :rtype: float

    """
    return sum(vocabulary.values())/ (1.0*len(vocabulary))


def type_frequency(vocabulary, k):
    """compute the number of words that occur exactly k times

    :param vocabulary: a Counter of words and their frequencies
    :param k: desired frequency
    :returns: number of words appearing k times
    :rtype: int

    """
    type_freq = 0
    for i in vocabulary.values():
        if i == k:
            type_freq += 1
    return type_freq

def unseen_types(first_vocab, second_vocab):
    """compute the number of words that appear in the second vocab but not in the first vocab

    :param first_vocab: a Counter of words and their frequencies in one dataset
    :param second_vocab: a Counter of words and their frequencies in another dataset
    :returns: number of words that appear in the second dataset but not in the first dataset
    :rtype: int

    """
    unseen = 0
    for i in set(second_vocab.elements()):
        if first_vocab[i] == 0:
            unseen += 1
    return unseen