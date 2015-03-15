import numpy as np
from scipy.special import xlogy

class Criterion(object):

    def __init__(self):
        self.output = None
        self.gradInput = None
        self.one_hot = None #Make np.eye vector of N_classes size
    
    def forward(self, inp, target):
        """ Computes criterion and updates self.output"""

    def backward(self, inp, target):
        """ Computer gradient w.r.t. input and updates gradInput"""


class CrossEntropyCriterion(Criterion):
    """
        Cross Entropy Criterion. Good for classification, target is usually
        one-hot vector.
    """

    def softmax(self, z):
        maxsub = z - np.max(z, axis=0)
        sm = np.exp(maxsub)
        return sm/sm.sum(axis=0)

    def crossEntropy(self, p, t):
        return np.sum(xlogy(t, t/p))

    def forward(self, inp, targets):
        if self.one_hot is None:
            self.one_hot = np.eye(inp.shape[0])
        self.output = self.crossEntropy(self.softmax(inp),
                                        self.one_hot[targets].T)
        return self.output

    def backward(self, inp, targets):
        self.gradInput = self.softmax(inp) - self.one_hot[targets].T
        return self.gradInput



