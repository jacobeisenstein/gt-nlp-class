from gtnlplib import evaluation

import torch
import numpy as np
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
from torch import optim

# deliverable 5.2
def build_linear(X_tr, Y_tr):
    '''
    Build a linear model in torch

    :param X_tr: the set of training documents
    :param Y_tr: the set of training labels
    :returns: PyTorch linear model
    :rtype: PyTorch model
    '''
    #change these
    size1 = 100
    size2 = 2

    model = torch.nn.Sequential()
    model.add_module("Linear",torch.nn.Linear(size1, size2, bias = True))

    return model

# deliverable 5.3
def log_softmax(scores):
    '''
    Compute the (log of the) softmax given the scores

    Hint: Use logsumexp

    :param scores: output of linear model as a numpy array
    :returns: the softmax result
    :rtype: numpy array
    '''

    raise NotImplementedError

# deliverable 5.4
def nll_loss(logP, Y_tr):
    '''
    Compute the neg-log likelihood loss from log softmax probabilities, averaged across documents

    return the loss in a number
    :param logP: output of log softmax
    :param Y_tr: the set of training labels
    :returns: the NLL loss
    :rtype: float
    '''

    raise NotImplementedError


######################### helper code
def train_model(loss, model, X_tr_var, Y_tr_var,
                num_its = 200,
                X_dv_var = None,
                Y_dv_var = None,
                status_frequency=10,
                optim_args = {'lr':0.002,'momentum':0},
                param_file = 'best.params'):

    # initialize optimizer
    optimizer = optim.SGD(model.parameters(), **optim_args)

    losses = []
    accuracies = []

    for epoch in range(num_its):
        # set gradient to zero
        optimizer.zero_grad()
        # run model forward to produce loss
        output = loss.forward(model.forward(X_tr_var),Y_tr_var)
        # backpropagate and train
        output.backward()
        optimizer.step()

        losses.append(output.data[0])

        # write parameters if this is the best epoch yet
        if X_dv_var is not None:
            # run forward on dev data
            _, Y_hat = model.forward(X_dv_var).max(dim=1)
            # compute dev accuracy
            acc = evaluation.acc(Y_hat.data.numpy(),Y_dv_var.data.numpy())
            # save
            if len(accuracies) == 0 or acc > max(accuracies):
                state = {'state_dict':model.state_dict(),
                         'epoch':len(accuracies)+1,
                         'accuracy':acc}
                torch.save(state,param_file)
            accuracies.append(acc)

        # print status message if desired
        if status_frequency > 0 and epoch % status_frequency == 0:
            print("Epoch "+str(epoch+1)+": Dev Accuracy: "+str(acc))

    # load parameters of best model
    checkpoint = torch.load(param_file)
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, losses, accuracies



def plot_results(losses, accuracies):
    fig,ax = plt.subplots(1,2,figsize=[12,2])
    ax[0].plot(losses)
    ax[0].set_ylabel('loss')
    ax[0].set_xlabel('iteration');
    ax[1].plot(accuracies);
    ax[1].set_ylabel('dev set accuracy')
    ax[1].set_xlabel('iteration');
