import codecs
from gtnlplib.constants import UNK
#from gtnlplib.bilstm import prepare_sequence

def load_data(input_file):
    """
    loads the entire data from the file into a list of words and their respective tags
    """
    X = []
    Y = []
    for i,(words,tags) in enumerate(conll_seq_generator(input_file)):
        X.append(words)
        Y.append(tags)
    
    return X,Y

def get_all_tags(input_file):
    """
    Return unique set of tags in the conll file
    
    Parameters:
    input_file -- the name of the input file
    returns -- a set of all the unique tags occuring in the file
    """
    all_tags = set([])
    for _, tags in conll_seq_generator(input_file):
        for tag in tags:
            all_tags.add(tag)
    return all_tags


def conll_seq_generator(input_file,max_insts=1000000):
    """
    Create a generator of (words, tags) pairs over the conll input file
    
    Parameters:
    input_file -- The name of the input file
    max_insts -- (optional) The maximum number of instances (words, tags)
                 instances to load
                 default value: 1000000 : is sufficient for our dataset
    returns -- generator of (words, tags) pairs
    """
    
    cur_words = []
    cur_tags = []
    num_insts = 0
    with codecs.open(input_file, encoding='utf-8') as instances:
        for line in instances:
            if num_insts >= max_insts:
                return

            if len(line.rstrip()) == 0:
                if len(cur_words) > 0:
                    num_insts += 1
                    yield cur_words,cur_tags
                    cur_words = []
                    cur_tags = []
            elif not line.startswith("# "):
                parts = line.rstrip().split()
                cur_words.append(parts[1])
                if len(parts)>3:
                    cur_tags.append(parts[3])
                else: 
                    cur_tags.append(UNK)
        
        #checking at the end of file
        if num_insts >= max_insts:
            return

        if len(cur_words)>0:
            num_insts += 1
            yield cur_words,cur_tags
    

