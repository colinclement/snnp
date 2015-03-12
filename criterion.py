import numpy as np
from itertools import izip

class Criterion(object):
    """ A training criterion """

    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, inp, target):
        """ Compute the forward pass of the input against
        the target
        
        Return the output"""

    def backward(self, inp, target):
        """ Return gradInput """

class NLLCriterion(Criterion):
    def __init__(self, weights=None):
        self.weights = weights

    def forward(self, inp, target):
        self.output = -np.array([ a[b] for a,b in zip(inp.T, target) ])
        return self.output

    def backward(self, inp, target):
        grad_input = np.zeros_like(inp)
        for i,t in enumerate(target):
            grad_input[t,i] = -1
            if self.weights is not None:
                grad_input[t,i] *= self.weights[t]
        self.grad_input = grad_input
        return grad_input

class MSECriterion(Criterion):
    def forward(self, inp, target):
        self.output = 0.5*(inp - target)**2
        return self.output

    def backward(self, inp, target):
        self.grad_input = (inp - target)
        return self.grad_input
