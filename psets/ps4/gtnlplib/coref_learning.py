import torch
from torch import nn
from torch import autograd as ag
from torch.nn import functional as F

from collections import defaultdict

from . import utils, coref

class FFCoref(nn.Module):
    '''
    A component that scores coreference relations based on a one-hot feature vector
    Architecture: input features -> Linear layer -> tanh -> Linear layer -> score
    '''
    
    ## deliverable 3.2
    def __init__(self, feat_names, hidden_dim):
        '''
        :param feat_names: list of keys to possible pairwise matching features
        :param hidden_dim: dimension of intermediate layer
        '''
        super(FFCoref, self).__init__()
        
        # STUDENT
        
        # END STUDENT
        
        
    ## deliverable 3.2
    def forward(self, features):
        '''
        :param features: defaultdict of pairwise matching features and their values for some pair
        :returns: model score
        :rtype: 1x1 torch Variable
        '''
        raise NotImplementedError
        
    ## deliverable 3.3
    def score_instance(self, markables, feats, i):
        '''
        A function scoring all coref candidates for a given markable
        Don't forget the new-entity option!
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param feats: feature extraction function
        :returns: list of scores for all candidates
        :rtype: torch.FloatTensor of dimensions 1x(i+1)
        '''
        raise NotImplementedError

    ## deliverable 3.4
    def instance_top_scores(self, markables, feats, i, true_antecedent):
        '''
        Find the top-scoring true and false candidates for i in the markable.
        If no false candidates exist, return (None, None).
        :param markables: list of all markables in the document
        :param i: index of current markable
        :param true_antecedent: gold label for markable
        :param feats: feature extraction function
        :returns trues_max: best-scoring true antecedent
        :returns false_max: best-scoring false antecedent
        '''
        scores = self.score_instance(markables, feats, i)
        
        raise NotImplementedError


def train(model, optimizer, markable_set, feats, margin=1.0, epochs=2):
    _zero = ag.Variable(torch.Tensor([0])) # this var is reusable
    model.train()
    for i in range(epochs):
        tot_loss = 0.0
        instances = 0
        for doc in markable_set:
            true_ants = coref.get_true_antecedents(doc)
            for i in range(len(doc)):
                max_t, max_f = model.instance_top_scores(doc, feats, i, true_ants[i])
                if max_t is None: continue
                marg_tensor = ag.Variable(torch.Tensor([margin])) # this var is not reusable
                unhinged_loss = marg_tensor - max_t + max_f
                loss = torch.max(torch.cat((_zero, unhinged_loss)))
                tot_loss += utils.to_scalar(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                instances += 1
        print(f'Loss = {tot_loss / instances}')
        
def evaluate(model, markable_set, feats):
    model.eval()
    coref.eval_on_dataset(make_resolver(feats, model), markable_set)
    
# helper
def make_resolver(features, model):
    return lambda markables : [utils.argmax(model.score_instance(markables, features, i)) for i in range(len(markables))]