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
    doing actions (i.e, SHIFT, ARC_L, ARC_R)
    """

    def __init__(self, sentence, sentence_embs, combiner, null_stack_tok_embed=None, root_tok_embed=None):
        """
        :param sentence A list of strings, the words in the sentence
        :param sentence_embs A list of ag.Variable objects, where the ith element is the embedding
            of the ith word in the sentence
        :param combiner A network component that gives an output embedding given two input embeddings
            when doing an arc operation
        :param null_stack_tok_embed ag.Variable The embedding of NULL_STACK_TOK
        :param root_tok_embed ag.Variable The embedding of ROOT_TOK
        """
        self.combiner = combiner

        self.input_buffer = [ StackEntry(we[0], pos, we[1]) for pos, we in enumerate(zip(sentence, sentence_embs)) ]

        self.stack = [StackEntry(ROOT_TOK, -1, root_tok_embed)]
        self.null_stack_tok_embed = null_stack_tok_embed

    def shift(self):
        next_item = self.input_buffer.pop(0)
        self.stack.append(next_item)

    def arc_left(self):
        return self._arc(Actions.ARC_L)

    def arc_right(self):
        return self._arc(Actions.ARC_R)

    def done_parsing(self):
        """
        Returns True if we are done parsing, else returns False
        Remember that we are padding the input with an <END-OF-INPUT> token.
        <END-OF-INPUT> should not be shifted onto the stack ever.
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT

    def stack_len(self):
        return len(self.stack)

    def input_buffer_len(self):
        return len(self.input_buffer)

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
        assert n <= len(self.input_buffer)
        return self.input_buffer[:n]

    def _arc(self, action):
        assert len(self.stack) >= 1, "ERROR: Cannot arc with stack length less than 2"
        
        head, modifier = self._get_arc_components(action)
        new_dep_edge = self._create_arc(head, modifier)
        
        return new_dep_edge

    def _create_arc(self, head, modifier):
        """
        Pass the head and modifier through the combiner to create a new embedding
        Create a new StackEntry and insert it back on the input buffer
        Create a new DepGraphEdge object for the arc and return it

        Note:
            - Make sure that the order of the embeddings you pass into
              the combiner is correct. The head word should always go first,
              and the modifier second. 
            - Make sure when creating the new dependency graph edge, that you store it in the DepGraphEdge object
              like this ( (head word, position of head word), (modifier word, position of modifier) ).
              Keeping track of the positions in the sentence is necessary to be able to uniquely
              identify edges when a sentence contains the same word multiple times.

        :param head: the StackEntry for the head of the arc
        :param modifier: the StackEntry for the modifier of the arc
        :return a new DepGraphEdge object for the arc from modifier to head
        """
        #STUDENT
        raise NotImplementedError
        #END STUDENT

    def _validate_action(self, action_to_do):
        """
        Ensure that the given action is legal with the current parser state
        There are three main cases to consider:
        - Don't shift when less than two items are on the input buffer
            - Do arc-right instead
        - Don't do an arc- operation when the stack is empty
            - Do shift instead
        - Don't do an arc-left when the root token is at the top of the stack
            - Do shift or arc-right, depending on the input buffer, instead
        :param action_to_do: the action chosen by the parser
        :return action_to_do: either the chosen action or the default legal one
        """
        # STUDENT
        raise NotImplementedError
        # END STUDENT

    def _get_arc_components(self, action):
        """
        Pop the first items from the stack and input buffer.
        Set one as head and one as modifier, according to the action 

        :param action: either arc left (ARC_L) or arc right (ARC_R)
        :return head: the StackEntry for the head according to the action
        :return modifier: the StackEntry for the modifier according to the action
        """
        # STUDENT
        # hint: use list.pop() to get and remove the left and right items
        raise NotImplementedError
        # END STUDENT

    def __str__(self):
        """
        Print the state for debugging
        """
        # only print the words, dont want to print the embeddings too
        return "Stack: {}\nInput Buffer: {}\n".format([ entry.headword for entry in self.stack ], 
                [ entry.headword for entry in self.input_buffer ])



class TransitionParser(nn.Module):

    def __init__(self, feature_extractor, word_embedding, action_chooser, combiner):
        """
        :param feature_extractor A FeatureExtractor object to get features
            from the parse state for making decisions
        :param word_embedding Network component to get embeddings for each word in the sentence
        :param action_chooser Network component that gives probabilities over actions (makes decisions)
        :param combiner Network component to combine embeddings during arc operations
        """
        super(TransitionParser, self).__init__()

        self.word_embedding = word_embedding
        self.feature_extractor = feature_extractor
        self.combiner = combiner
        self.action_chooser = action_chooser
        self.use_cuda = False

        # Embeddings for special tokens
        self.null_stack_tok_embed = nn.Parameter(torch.randn(1, word_embedding.output_dim))
        self.root_tok_embed = nn.Parameter(torch.randn(1, word_embedding.output_dim))


    def forward(self, sentence, actions=None):
        """
        Does the core parsing logic.
        Make sure to return everything that needs to be returned
            1. The log probabilities from every choice made
            2. The dependency graph
            3. The actions you did, as a list

        The boiler plate at the beginning initializes a valid
        ParserState object, and now you may do actions on that state by calling
        shift(), arc_right(), arc_left(), or get features from it in your
        feature extractor.

        If you are supplied gold actions, you should do those.
        Make sure that you only do valid actions if you are not supplied gold actions (use _validate_action).

        Also, note that symbolic constants have been defined for the different Actions in constants.py
        E.g Actions.SHIFT is 0, Actions.ARC_L is 1, so that the 0th element of
        the output of your action chooser is the log probability of shift, the 1st is the log probability
        of ARC_L, etc.
        """
        self.refresh() # clear up hidden states from last run, if need be

        padded_sent = sentence + [END_OF_INPUT_TOK]

        # Initialize the parser state
        sentence_embs = self.word_embedding(padded_sent)
        parser_state = ParserState(padded_sent, sentence_embs, self.combiner,
                                   null_stack_tok_embed=self.null_stack_tok_embed,
                                   root_tok_embed=self.root_tok_embed)

        outputs = [] # Holds the output of each action decision
        actions_done = [] # Holds all actions we have done
        dep_graph = set() # Build this up as you go

        # Make the gold action queue if we have it
        if actions is not None:
            action_queue = deque()
            action_queue.extend([ Actions.action_to_ix[a] for a in actions ])
            have_gold_actions = True
        else:
            have_gold_actions = False

        while not parser_state.done_parsing():
            # STUDENT
            pass
            # END STUDENT
        return outputs, dep_graph, actions_done

    
    def refresh(self):
        if isinstance(self.combiner, neural_net.LSTMCombiner):
            self.combiner.clear_hidden_state()
        if isinstance(self.action_chooser, neural_net.LSTMActionChooser):
            self.action_chooser.clear_hidden_state()
        if isinstance(self.word_embedding, neural_net.BiLSTMWordEmbedding):
            self.word_embedding.clear_hidden_state()


    def predict(self, sentence):
        _, dep_graph, _ = self.forward(sentence)
        return dep_graph


    def predict_actions(self, sentence):
        _, _, actions_done = self.forward(sentence)
        return actions_done
    

    def to_cuda(self):
        self.use_cuda = True
        self.word_embedding.use_cuda = True
        self.action_chooser.use_cuda = True
        self.combiner.use_cuda = True
        self.cuda()


    def to_cpu(self):
        self.use_cuda = False
        self.word_embedding.use_cuda = False
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
        print("Number of instances: {}    Number of network actions: {}".format(instance_count, total_actions))
        print("Acc: {}  Loss: {}".format(float(correct_actions) / total_actions, tot_loss / instance_count))


def evaluate(data, model, verbose=False):

    correct_actions = 0
    total_actions = 0
    tot_loss = 0.
    instance_count = 0
    criterion = nn.NLLLoss()

    if model.use_cuda:
        criterion.cuda()

    for sentence, actions in data:

        if len(sentence) > 1:
            outputs, _, actions_done = model(sentence, actions)

            if model.use_cuda:
                loss = ag.Variable(cuda.FloatTensor([0]))
                action_idxs = [ ag.Variable(cuda.LongTensor([ a ])) for a in actions_done ]
            else:
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
        print("Number of instances: {}    Number of network actions: {}".format(instance_count, total_actions))
        print("Acc: {}  Loss: {}".format(float(correct_actions) / total_actions, tot_loss / instance_count))
    return acc, loss
