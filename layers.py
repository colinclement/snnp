import numpy as np
from scipy.linalg import inv, sqrtm
import activations as act
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt

def orthogonalize(m):
    return np.real(m.dot(inv(sqrtm(m.T.dot(m)))))

class Layer(object):
    """ A single layer of a neural network 
        Contains the following state variables:
        self.output - computed with last call of self.forward()
        self.gradInput - Gradients w.r.t. inputs computed with
            last call of self.updateGradInput
        self.gradOutput - Gradients w.r.t. output of layer
        self.train - Bool if network is being trained
    """


    def forward(self, inp):
        """ Given an input, compute the forward pass """
        return self.updateOutput(inp)

    def backward(self, inp, gradOutput):
        """ Given an gradient above, do backprob back
        through the layer """
        self.gradInput = self.updateGradInput(inp, gradOutput)
        return self.accGradParameters(inp, gradOutput, self.scale)

    def updateOutput(self, inp):
        """ Override to set and return self.output"""
        self.output = inp
        return self.output

    def updateGradInput(self, inp, gradOutput):
        """Compute gradient of layer w.r.t. its own paramm"""
        return self.gradInput

    def accGradParameters(self, inp, gradOutput, scale):
        """Accumulates gradients w.r.t. parameters"""

    def zeroGradParameters(self):
        """Used to set grad parameter accumulation to zero"""
        
    def updateParameters(self, learningrate):
        """Update parameters accumulated
        parameters = parameters - learningRate * gradients_wrt_parameters
        """

    def parameters(self):
        """Returns learnable parameters and gradients of those parameters """
    
    def getParameters(self):
        """Returns flat arrays of params and gradParams"""
        p, gp = self.parameters()
        return p.ravel(), gp.ravel()
    
    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False


class Linear(Layer):
    """Linear layer with weights and biases  Wx+b"""
    def __init__(self, fan_in, fan_out):
        self.W = np.random.randn(fan_out, fan_in)
        self.b = np.random.randn(fan_out)

    def updateOutput(self, inp):
        self.output = self.W.dot(inp) + self.b
        return self.output

    def updateGradInput(self, inp, gradOutput):

    
class LossLayer(Layer):
    """ Represents a loss layer at the top of the network """
    def __init__(self, N):
        """
        N : number of nodes
        """
        self.W = np.eye(N) #No weights
    
    def forward(self, inp, tl):
        self.newinp = inp.copy()
        self.newtl = tl
        self.newact = act.crossEntropy_of_softmax(inp, tl)
        return self.newact 

    def backward(self, out = None, tl=None):
        return act.d_crossEntropy_of_softmax(out or self.newinp,
                                             tl or self.newtl) 


class InputLayer(Layer):
    """ Represents the input """
    def __init__(self, data):
        self.W = 1 #No weights

    def forward(self, inp, tl):
        self.newact = inp.copy()
        self.newtl = tl
        return self.newact

    def backward(self, inp=None, tl = None):
        return inp or self.newact


