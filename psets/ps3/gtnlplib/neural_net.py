import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F

from gtnlplib.constants import Actions, HAVE_CUDA
import gtnlplib.utils as utils

if HAVE_CUDA:
    import torch.cuda as cuda

# ===-----------------------------------------------------------------------------===
# WORD EMBEDDING COMPONENTS
# ===-----------------------------------------------------------------------------===
# These components are responsible for initializing the input buffer with embeddings.
# An embedding must be supplied for each word in the sentence.
# 
# This class of components has the interface
# inputs: the input sentence as a list of strings
# outputs: a list of autograd Variables, where the ith element of the list is the
#          embedding for the ith word.
# 
# The output of forward() for these components is what is used to initialize the
# input buffer, and what will be shifted onto the stack, and used in combination
# when doing arc operations.


class VanillaWordEmbedding(nn.Module):
    """
    A component that simply returns a list of the word embeddings as
    autograd Variables.
    """

    def __init__(self, word_to_ix, embedding_dim):
        """
        Construct an embedding lookup table for use in the forward()
        function
        :param word_to_ix Dict mapping words to unique indices
        :param embedding_dim The dimensionality of the embeddings
        """
        super(VanillaWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.embedding_dim = embedding_dim
        self.use_cuda = False

        # This is just to let the parser know the size of embeddings it is getting
        self.output_dim = embedding_dim

        # STUDENT
        # name your embedding member "word_embeddings"
        raise NotImplementedError
        # END STUDENT


    def forward(self, sentence):
        """
        :param sentence A list of strings, the text of the sentence
        :return A list of autograd.Variables, where list[i] is the
            embedding of word i in the sentence.
            NOTE: the Variables returned should be row vectors, that
                is, of shape (1, embedding_dim)
        """
        embeds = [] # store each Variable in here
        # STUDENT
        # END STUDENT
        return embeds


class BiLSTMWordEmbedding(nn.Module):
    """
    In this component, you will use a Bi-Directional
    LSTM to get the initial embeddings.  The embedding
    for word i to initailize the input buffer is the ith hidden state of the LSTM
    after passing the sentence through the LSTM.
    """

    def __init__(self, word_to_ix, word_embedding_dim, hidden_dim, num_layers, dropout):
        """
        :param word_to_ix Dict mapping words to unique indices
        :param word_embedding_dim The dimensionality of the input word embeddings
        :param hidden_dim The dimensionality of the output embeddings that go
            on the stack
        :param num_layers The number of LSTM layers to have
        :param dropout Amount of dropout to have in LSTM
        """
        super(BiLSTMWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.use_cuda = False

        self.output_dim = hidden_dim

        # STUDENT
        # Construct the needed components in this order:
        # 1. An embedding lookup table
        # 2. The LSTM
        # Note we want the output dim to be hidden_dim, but since our LSTM
        # is bidirectional, we need to make the output of each direction hidden_dim/2
        # name your embedding member "word_embeddings"
        raise NotImplementedError
        # END STUDENT

        self.hidden = self.init_hidden()

    def forward(self, sentence):
        """
        This function has two parts
        1. Look up the embeddings for the words in the sentence.
           These will be the inputs to the LSTM sequence model.
           NOTE: At this step, rather than a list of embeddings, it should be a single tensor.
        2. Now that you have your tensor of embeddings, You can pass it through your LSTM.
        3. Convert the outputs into the correct return type, which is a list of
           embeddings of shape (1, embedding_dim)
        NOTE: Make sure you are reassigning self.hidden_state to the new hidden state!

        :param sentence A list of strs, the words of the sentence
        :return A list of autograd.Variables, where list[i] is the embedding of word i in the sentence.
        NOTE: the Variables returned should be row vectors, that is, of shape (1, embedding_dim)
        """
        assert self.word_to_ix is not None, "ERROR: Make sure to set word_to_ix on \
                the embedding lookup components"
        # STUDENT
        raise NotImplementedError
        # END STUDENT

    def init_hidden(self):
        """
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        """
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers * 2, 1, self.hidden_dim//2).zero_()))
        else:
            return (ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)),
                    ag.Variable(torch.zeros(self.num_layers * 2, 1, self.hidden_dim//2)))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()

class SuffixAndWordEmbedding(nn.Module):
    """
    A component that embeds words and their suffixes, and concatenates the embeddings
    """

    def __init__(self, word_to_ix, suff_to_ix, embedding_dim):
        """
        Construct an embedding lookup table for the words and suffixes
        :param word_to_ix Dict mapping words to unique indices
        :param suff_to_ix Dict mapping suffixes to unique indices
        :param embedding_dim The dimensionality of the output word embeddings
        """
        super(SuffixAndWordEmbedding, self).__init__()
        self.word_to_ix = word_to_ix
        self.suff_to_ix = suff_to_ix
        self.embedding_dim = embedding_dim
        self.use_cuda = False

        self.output_dim = embedding_dim

        # STUDENT create your embeddings here. 
        # Note that embedding_dim should be the final (i.e. concatenated) word embedding size
        # suffix and word embeddings should be the same size
        raise NotImplementedError
        # END STUDENT


    def forward(self, sentence):
        """
        Compute word embeddings by concatenating the word and suffix embeddings together

        :param sentence A list of strings, the text of the sentence
        :return A list of autograd.Variables, where list[i] is the embedding of word i in the sentence.
        NOTE: the Variables returned should be row vectors, that is, of shape (1, embedding_dim)
        """
        embeds = [] # store each Variable in here
        # STUDENT
        # END STUDENT
        return embeds


# ===-----------------------------------------------------------------------------===
# COMBINER NETWORK COMPONENTS
# ===-----------------------------------------------------------------------------===
# These components have the interface:
# inputs: head_embed, modifier_embed from the stack during an arc operation
# outputs: A new embedding to place back on the stack, representing the combination
#       of head and modifier

class FFCombiner(nn.Module):
    """
    This network piece takes the top two elements of the stack's embeddings
    and combines them to create a new embedding after an arc operation.

    The network architecture is:
    Inputs: 2 word embeddings (the head and the modifier embeddings)
    Output: Run through a linear layer -> tanh -> linear layer
    """

    def __init__(self, embedding_dim):
        """
        Construct the linear components you will need in forward()
        NOTE: Think carefully about what the input and output
            dimensions of your linear layers should be
        :param embedding_dim The dimensionality of the embeddings
        """
        super(FFCombiner, self).__init__()

        # STUDENT
        # Construct the components in this order
        # 1. The first linear layer
        # 2. The second linear layer
        # The output of the first linear layer should be embedding_dim
        # (the rest of the input/output dims are thus determined)
        raise NotImplementedError
        # END STUDENT

    def forward(self, head_embed, modifier_embed):
        """
        First, concatenate head_embed and modifier_embed into a single tensor.
        Then, apply linear -> tanh -> linear to the concatenated tensor to get a new representation.

        :param head_embed The embedding of the head in the arc operation
        :param modifier_embed The embedding of the modifier in the arc operation
        :return The embedding of the combination as a row vector of shape (1, embedding_dim)
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT


class LSTMCombiner(nn.Module):
    """
    A combiner network that does a sequence model over states, rather
    than just a simple encoder like above.

    Input: 2 embeddings, the head embedding and modifier embedding
    Output: Concatenate the 2 embeddings together and do one timestep
        of the LSTM, returning the hidden state, which will be placed
        on the stack.
    """

    def __init__(self, embedding_dim, num_layers, dropout):
        """
        Construct your LSTM component for use in forward().
        Think about what size the input and output of your LSTM
        should be

        :param embedding_dim Dimensionality of stack embeddings
        :param num_layers How many LSTM layers to use
        :param dropout The amount of dropout to use in LSTM
        """
        super(LSTMCombiner, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.use_cuda = False

        # STUDENT
        raise NotImplementedError
        # END STUDENT

        self.hidden = self.init_hidden()


    def forward(self, head_embed, modifier_embed):
        """
        Do the next LSTM step, and return the hidden state as the new
        embedding for the arc operation

        Here, note that PyTorch's LSTM wants the input to be a tensor with axis semantics
        (seq_len, batch_size, input_dimensionality), but we are not minibatching (so batch_size=1)
        and seq_len=1 since we are only doing 1 timestep at a time

        NOTE: Make sure the tensor you hand to your LSTM is the size it wants:
            (seq_len, batch_size, embedding_dim), which in this case, is (1, 1, embedding_dim)
        NOTE: If you add more layers to the LSTM (more than 1), your code may break.
            To fix it, look at the value of self.hidden whenever you have more layers.

        :param head_embed Embedding of the head word
        :param modifier_embed Embedding of the modifier of shape (1, embedding_dim)
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT

    def init_hidden(self):
        """
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        """
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()))
        else:
            return (ag.Variable(torch.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()),
                    ag.Variable(torch.FloatTensor(self.num_layers, 1, self.embedding_dim).zero_()))


    def clear_hidden_state(self):
        self.hidden = self.init_hidden()


# ===-----------------------------------------------------------------------------===
# ACTION CHOOSING COMPONENTS
# ===-----------------------------------------------------------------------------===
class FFActionChooser(nn.Module):
    """
    This network piece takes features from the current
    state of the parser and runs them through a feedforward network,
    returning log probabilities over actions

    The network should be
    inputs -> linear layer -> relu -> linear layer -> log softmax
    """

    def __init__(self, input_dim):
        """
        Construct the linear components that you need in forward() here.
        The output of the first linear layer should have the same size as its input
        What should be the dimensionality of your log softmax at the end?

        :param input_dim The dimensionality of your input: that is, when all your
            feature embeddings are concatenated together
        """
        super(FFActionChooser, self).__init__()
        self.use_cuda = False
        # STUDENT
        # Construct in this order:
        # 1. The first linear layer (the one that is called first in the forward pass)
        # 2. The second linear layer
        raise NotImplementedError
        # END STUDENT

    def forward(self, inputs):
        """
        combine all the features into one big row vector, then compute log probabilities

        :param inputs A list of autograd.Variables, which are all of the features we will use
        :return a Variable which is the log probabilities of the actions, of shape (1, 3)
            (it is a row vector, with an entry for each action)
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT

class LSTMActionChooser(nn.Module):
    """
    This network piece takes features from the current
    state of the parser and runs them through an LSTM,
    returning log probabilities over actions

    The network should be
    inputs -> LSTM -> relu -> linear layer -> log softmax
    """

    def __init__(self, input_dim, num_layers, dropout):
        """
        Construct the linear components that you need in forward() here.
        Think carefully about the input and output dimensionality of your linear layers
        HINT: What should be the dimensionality of your log softmax at the end?

        :param input_dim The dimensionality of your input: that is, when all your
            feature embeddings are concatenated together
        """
        super(LSTMActionChooser, self).__init__()
        self.num_layers = num_layers
        self.use_cuda = False
        self.input_dim = input_dim
        # STUDENT
        # Construct in this order:
        # 1. The LSTM layer 
        # 2. The linear layer to predict actions
        raise NotImplementedError
        # END STUDENT
        self.hidden = self.init_hidden()
    
    def forward(self, inputs):
        """
        combine all the features into one big row vector, then compute log probabilities

        :param inputs A list of autograd.Variables, which are all of the features we will use
        :return a Variable which is the log probabilities of the actions, of shape (1, 3)
            (it is a row vector, with an entry for each action)
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT

    def init_hidden(self):
        """
        PyTorch wants you to supply the last hidden state at each timestep
        to the LSTM.  You shouldn't need to call this function explicitly
        """
        if self.use_cuda:
            return (ag.Variable(cuda.FloatTensor(self.num_layers, 1, self.input_dim).zero_()),
                    ag.Variable(cuda.FloatTensor(self.num_layers, 1, self.input_dim).zero_()))
        else:
            return (ag.Variable(torch.FloatTensor(self.num_layers, 1, self.input_dim).zero_()),
                    ag.Variable(torch.FloatTensor(self.num_layers, 1, self.input_dim).zero_()))

    def clear_hidden_state(self):
        self.hidden = self.init_hidden()



