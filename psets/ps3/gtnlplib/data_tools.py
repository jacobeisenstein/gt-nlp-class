from collections import namedtuple
from gtnlplib.constants import END_OF_INPUT_TOK, NULL_STACK_TOK

Instance = namedtuple("Instance", ["sentence", "action_sequence"])


def parse_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        instances = []
        vocab = set()
        for inst in f:
            sentence, actions = inst.split(" ||| ")

            # Make sure there is no leading/trailing whitespace
            sentence = sentence.strip().split()
            actions = actions.strip().split()

            for word in sentence:
                vocab.add(word)

            instances.append(Instance(sentence, actions))
    return instances, vocab


def read_test_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        sentences = []
        vocab = set()
        for sent in f:
            sentence = sent.strip().split()
            for word in sentence:
                vocab.add(word)

            sentences.append(sentence)
    return sentences, vocab


class Dataset:
    """
    Class for holding onto the train, dev and test data
    and returning iterators over it.
    Also stores the full data's vocab in a set
    """


    def __init__(self, train_filename, dev_filename, test_filename):
        self._vocab = set()
        if train_filename is not None:
            self._training_data, tmp = parse_file(train_filename)
            self._vocab.update(tmp)
        if dev_filename is not None:
            self._dev_data, tmp = parse_file(dev_filename)
            self._vocab.update(tmp)
        if test_filename is not None:
            self._test_data, tmp = read_test_file(test_filename)
            self._vocab.update(tmp)    

        self._vocab.add(END_OF_INPUT_TOK)
        self._vocab.add(NULL_STACK_TOK)

    @property
    def training_data(self):
        return self._training_data

    @property
    def dev_data(self):
        return self._dev_data

    @property
    def test_data(self):
        return self._test_data

    @property
    def vocab(self):
        return self._vocab


def make_file_key(output_filename, sentences, gold_parses):
    with open(output_filename, "w") as outfile:
        for sentence, parse in zip(sentences, gold_parses):
            for i, word in enumerate(sentence):
                for edge in parse:
                    if edge.modifier[1] == i:
                        outfile.write("{}\t{}\t{}\t{}\n".format(i, word, edge.head[0], edge.head[1]))
            outfile.write("\n")


def make_kaggle_key(output_filename, sentences, gold_parses):
    wc = 1
    with open(output_filename, "w") as outfile:
        outfile.write("Id,Prediction\n")
        for sentence, parse in zip(sentences, gold_parses):
            for i, word in enumerate(sentence):
                for edge in parse:
                    if edge.modifier[1] == i:
                        outfile.write("{},{}\n".format(wc, edge.head[1]))
                        wc += 1
