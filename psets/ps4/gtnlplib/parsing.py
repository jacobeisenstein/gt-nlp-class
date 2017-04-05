from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.autograd as ag

from gtnlplib.constants import Actions, NULL_STACK_TOK, END_OF_INPUT_TOK, HAVE_CUDA, ROOT_TOK
import gtnlplib.utils as utils
import gtnlplib.neural_net as neural_net

if HAVE_CUDA:
    import torch.cuda as cuda


# Named tuples are basically just like C structs, where instead of accessing
# indices where the indices have no semantics, you can access the tuple with a name.
# check python docs
DepGraphEdge = namedtuple("DepGraphEdge", ["head", "modifier"])

# These are what initialize the input buffer, and are used on the stack.
# headword: The head word, stored as a string
# headword_pos: The position of the headword in the sentence as an int
# embedding: The embedding of the phrase as an autograd.Variable
StackEntry = namedtuple("StackEntry", ["headword", "headword_pos", "embedding"])


class ParserState:
    """
    Manages the state of a parse by keeping track of
    the input buffer and stack, and offering a public interface for
    doing actions (i.e, SHIFT, REDUCE_L, REDUCE_R)
    """

    def __init__(self, sentence, sentence_embs, combiner, null_stack_tok_embed=None):
        """
        :param sentence A list of strings, the words in the sentence
        :param sentence_embs A list of ag.Variable objects, where the ith element is the embedding
            of the ith word in the sentence
        :param combiner A network component that gives an output embedding given two input embeddings
            when doing a reduction
        :param null_stack_tok_embed ag.Variable The embedding of NULL_STACK_TOK
        """
        self.combiner = combiner

        # Input buffer is a list, along with an index into the list.
        # curr_input_buff_idx points to the *next element to pop off the input buffer*
        self.curr_input_buff_idx = 0
        self.input_buffer = [ StackEntry(we[0], pos, we[1]) for pos, we in enumerate(zip(sentence, sentence_embs)) ]

        self.stack = []
        self.null_stack_tok_embed = null_stack_tok_embed

    def shift(self):
        next_item = self.input_buffer[self.curr_input_buff_idx]
        self.stack.append(next_item)
        self.curr_input_buff_idx += 1

    def reduce_left(self):
        return self._reduce(Actions.REDUCE_L)

    def reduce_right(self):
        return self._reduce(Actions.REDUCE_R)

    def done_parsing(self):
        """
        Returns True if we are done parsing.
        Otherwise, returns False
        Remember that we are padding the input with an <END-OF-INPUT> token.
        self.curr_input_buff_idx is the just an index to the next token in the input buffer
        (i.e, each SHIFT increments this index by 1).
        <END-OF-INPUT> should not be shifted onto the stack ever.
        """
        # STUDENT
        pass
        # END STUDENT

    def stack_len(self):
        return len(self.stack)

    def input_buffer_len(self):
        return len(self.input_buffer) - self.curr_input_buff_idx

    def stack_peek_n(self, n):
        """
        Look at the top n items on the stack.
        If you ask for more than are on the stack, copies of the null_stack_tok_embed
        are returned
        :param n How many items to look at
        """
        if len(self.stack) - n < 0:
            return [ StackEntry(NULL_STACK_TOK, -1, self.null_stack_tok_embed) ] * (n - len(self.stack)) \
                   + self.stack[:]
        return self.stack[-n:]

    def input_buffer_peek_n(self, n):
        """
        Look at the next n words in the input buffer
        :param n How many words ahead to look
        """
        assert self.curr_input_buff_idx + n - 1 <= len(self.input_buffer)
        return self.input_buffer[self.curr_input_buff_idx:self.curr_input_buff_idx+n]

    def _reduce(self, action):
        """
        Reduce the top two items on the stack, combine them,
        and place the combination back on the stack.
        Combination means running the embeddings of the two stack items
        through your combiner network and getting a dense output.
        (The ParserState stores the combiner in the instance variable self.combiner).

        Important things to note:
            - Make sure you get which word is the head and which word
              is the modifer correct
            - Make sure that the order of the embeddings you pass into
              the combiner is correct.  The head word should always go first,
              and the modifier second.  Mixing it up will make it more difficult
              to learn the relationships it needs to learn
            - Make sure when creating the new dependency graph edge, that you store it in the DepGraphEdge object
              like this ( (head word, position of head word), (modifier word, position of modifier) ).
              Keeping track of the positions in the sentence is necessary to be able to uniquely
              identify edges when a sentence contains the same word multiple times.
              Technically the position is all that is needed, but it will be easier to debug
              if you carry along the word too.

        :param action Whether we reduce left or reduce right
        :return DepGraphEdge The edge that was formed in the dependency graph.
        """
        assert len(self.stack) >= 2, "ERROR: Cannot reduce with stack length less than 2"
        
        # STUDENT
        # hint: use list.pop()
        # END STUDENT

    def __str__(self):
        """
        Print the state for debugging
        """
        # only print the words, dont want to print the embeddings too
        return "Stack: {}\nInput Buffer: {}\n".format([ entry.headword for entry in self.stack ], 
                [ entry.headword for entry in self.input_buffer[self.curr_input_buff_idx:] ])



class TransitionParser(nn.Module):

    def __init__(self, feature_extractor, word_embedding_component, action_chooser_component, combiner_component):
        """
        :param feature_extractor A FeatureExtractor object to get features
            from the parse state for making decisions
        :param word_embedding_component Network component to get embeddings for each word in the sentence
            TODO implement this
        :param action_chooser_component Network component that gives probabilities over actions (makes decisions)
        :param combiner_component Network component to combine embeddings during reductions
        """
        super(TransitionParser, self).__init__()

        self.word_embedding_component = word_embedding_component
        self.feature_extractor = feature_extractor
        self.combiner = combiner_component
        self.action_chooser = action_chooser_component
        self.use_cuda = False

        # This embedding is what is returned to indicate that part of the stack is empty
        self.null_stack_tok_embed = nn.Parameter(torch.randn(1, word_embedding_component.output_dim))


    def forward(self, sentence, actions=None):
        """
        Does the core parsing logic.
        If you are supplied actions, you should do those.
        Make sure to return everything that needs to be returned
            * The log probabilities from every choice made
            * The dependency graph
            * The actions you did as a list

        The boiler plate at the beginning is just gross padding and stuff
        You can basically ignore it.  Just know that it initializes a valid
        ParserState object, and now you may do actions on that state by calling
        shift(), reduce_right(), reduce_left(), or get features from it in your
        feature extractor.

        Make sure that if you are not supplied actions, and you do actions your
        network predicts, that you only do valid actions.

        Also, note that symbolic constants have been defined for the different Actions in constants.py
        E.g Actions.SHIFT is 0, Actions.REDUCE_L is 1, so that the 0th element of
        the output of your action chooser is the log probability of shift, the 1st is the log probability
        of REDUCE_L, etc.
        """
        self.refresh() # clear up hidden states from last run, if need be

        padded_sent = sentence + [END_OF_INPUT_TOK]

        # Initialize the parser state
        sentence_embs = self.word_embedding_component(padded_sent)

        parser_state = ParserState(padded_sent, sentence_embs, self.combiner, null_stack_tok_embed=self.null_stack_tok_embed)
        outputs = [] # Holds the output of each action decision
        actions_done = [] # Holds all actions we have done

        dep_graph = set() # Build this up as you go

        # Make the action queue if we have it
        if actions is not None:
            action_queue = deque()
            action_queue.extend([ Actions.action_to_ix[a] for a in actions ])
            have_gold_actions = True
        else:
            have_gold_actions = False

        # STUDENT
        # END STUDENT

        dep_graph.add(DepGraphEdge((ROOT_TOK, -1), (parser_state.stack[-1].headword, parser_state.stack[-1].headword_pos)))
        return outputs, dep_graph, actions_done


    def refresh(self):
        if isinstance(self.combiner, neural_net.LSTMCombinerNetwork):
            self.combiner.clear_hidden_state()
        if isinstance(self.word_embedding_component, neural_net.BiLSTMWordEmbeddingLookup):
            self.word_embedding_component.clear_hidden_state()


    def predict(self, sentence):
        _, dep_graph, _ = self.forward(sentence)
        return dep_graph


    def predict_actions(self, sentence):
        _, _, actions_done = self.forward(sentence)
        return actions_done
    

    def to_cuda(self):
        self.use_cuda = True
        self.word_embedding_component.use_cuda = True
        self.combiner.use_cuda = True
        self.cuda()


    def to_cpu(self):
        self.use_cuda = False
        self.word_embedding_component.use_cuda = False
        self.combiner.use_cuda = False
        self.cpu()


def train(data, model, optimizer, verbose=True):
    criterion = nn.NLLLoss()

    if model.use_cuda:
        criterion.cuda()

    correct_actions = 0
    total_actions = 0
    tot_loss = 0.
    instance_count = 0

    for sentence, actions in data:

        if len(sentence) <= 2:
            continue

        optimizer.zero_grad()
        model.refresh()

        outputs, _, actions_done = model(sentence, actions)

        if model.use_cuda:
            loss = ag.Variable(cuda.FloatTensor([0]))
            action_idxs = [ ag.Variable(cuda.LongTensor([ a ])) for a in actions_done ]
        else:
            loss = ag.Variable(torch.FloatTensor([0]))
            action_idxs = [ ag.Variable(torch.LongTensor([ a ])) for a in actions_done ]

        for output, act in zip(outputs, action_idxs):
            loss += criterion(output.view(-1, 3), act)

        tot_loss += utils.to_scalar(loss.data)
        instance_count += 1

        for gold, output in zip(actions_done, outputs):
            pred_act = utils.argmax(output.data)
            if pred_act == gold:
                correct_actions += 1
        total_actions += len(outputs)

        loss.backward()
        optimizer.step()

    acc = float(correct_actions) / total_actions
    loss = float(tot_loss) / instance_count
    if verbose:
        print "Number of instances: {}    Number of network actions: {}".format(instance_count, total_actions)
        print "Acc: {}  Loss: {}".format(float(correct_actions) / total_actions, tot_loss / instance_count)


def evaluate(data, model, verbose=False):

    correct_actions = 0
    total_actions = 0
    tot_loss = 0.
    instance_count = 0
    criterion = nn.NLLLoss()

    for sentence, actions in data:

        if len(sentence) > 1:
            outputs, _, actions_done = model(sentence, actions)

            loss = ag.Variable(torch.FloatTensor([0]))
            action_idxs = [ ag.Variable(torch.LongTensor([ a ])) for a in actions_done ]
            for output, act in zip(outputs, action_idxs):
                loss += criterion(output.view((-1, 3)), act)

            tot_loss += utils.to_scalar(loss.data)
            instance_count += 1

            for gold, output in zip(actions_done, outputs):
                pred_act = utils.argmax(output.data)
                if pred_act == gold:
                    correct_actions += 1

            total_actions += len(outputs)

    acc = float(correct_actions) / total_actions
    loss = float(tot_loss) / instance_count
    if verbose:
        print "Number of instances: {}    Number of network actions: {}".format(instance_count, total_actions)
        print "Acc: {}  Loss: {}".format(float(correct_actions) / total_actions, tot_loss / instance_count)
    return acc, loss
