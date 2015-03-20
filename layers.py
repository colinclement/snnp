"""
SNNP - Simple Neural Networks with Python

author: Colin Clement
date: 2015-03-13

Neural network layer classes based on torch nn module
"""

import numpy as np
from scipy.linalg import inv, sqrtm
from scipy.special import expit
from scipy.misc import logsumexp

def orthogonalize(m):
    """Make columns orthogonal, to use Ganguli's initial conditions"""
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
    def __init__(self):
        """Default initializatin"""
        self.output = None #Computed with last call of forward()
        self.gradInput = None # Gradients w.r.t. inputs

        self.W = None #learnable weights
        self.b = None #learnable biases
        self.a = None #Activation parameters

        self.gradW = None #Gradient of W
        self.gradb = None #Gradient of b
        self.grada = None #Gradient of a

        self.train = False
        self.scale = 1.

    def forward(self, inp):
        """ Given an input, compute the forward pass """
        return self.updateOutput(inp)

    def backward(self, inp, gradOutput, scale=1.):
        """ Given an gradient of output, do backprob back
        through the layer and return gradient of input"""
        self.updateGradInput(inp, gradOutput)
        self.accGradParameters(inp, gradOutput, scale)
        return self.gradInput

    def updateOutput(self, inp):
        """ Override to set and return self.output"""
        self.output = inp
        return self.output

    def updateGradInput(self, inp, gradOutput):
        """Compute gradient of layer w.r.t. its own paramm"""

    def accGradParameters(self, inp, gradOutput, scale):
        """Accumulates gradients w.r.t. parameters"""
        return self.gradInput

    def zeroGradParameters(self):
        """Used to set grad parameter accumulation to zero"""
        p, gp = self.parameters()
        for g in gp:
            g.fill(0)
    
    def updateParameters(self, learningrate):
        """Update parameters accumulated """
        ps, gps = self.parameters()
        for p, gp in zip(ps, gps):
            p -= learningrate*gp
    
    def parameters(self):
        """Returns learnable parameters and gradients of those parameters """
        if self.W is not None and self.b is not None:
            return [self.W, self.b], [self.gradW, self.gradb]
        elif self.W is not None:
            return [self.W], [self.gradW]
        elif self.b is not None:
            return [self.b], [self.gradb]
        elif self.a is not None: #Activations like PReLU
            return [self.a], [self.grada]
        else:
            return [], []
    
    def getParameters(self):
        """Returns flat arrays of params and gradParams"""
        p, gp = self.parameters()
        if p and gp:
            return map(np.ravel, p), map(np.ravel, gp)
        else:
            return p, gp
    
    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False


class Linear(Layer):
    """Linear layer with weights and biases  Wx+b"""
    def __init__(self, fan_in, fan_out, weight_init='PReLU'):
        if weight_init == 'PReLU':
            self.W = np.random.normal(0.0, np.sqrt(2./fan_out), (fan_out, fan_in))
            self.b = np.random.normal(0.0, np.sqrt(2./fan_out), fan_out)
        elif weight_init == 'Ganguli': #Saxe & Ganguli initial conditions
            self.W = np.random.normal(0.0, 1./np.sqrt(fan_out), (fan_out, fan_in))
            self.W = orthogonalize(self.W)
            self.b = np.random.normal(0.0, 1./np.sqrt(fan_out), fan_out)
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def updateOutput(self, inp):
        self.output = self.W.dot(inp) + self.b[:,None]
        return self.output

    def updateGradInput(self, inp, gradOutput):
        self.gradInput = (gradOutput.T.dot(self.W)).T

    def accGradParameters(self, inp, gradOutput, scale=1.0):
        """ Update gradient accumulations, mean used so learningrate is
            independent of number of samples"""    
        self.gradW += scale * (gradOutput[:,None,:] * inp[None,:,:]).mean(-1)
        self.gradb += scale * gradOutput.mean(-1)
   

class Softmax(Layer):
    """ Softmax layer"""
   
    def softmax(self, inp):
        maxsub = inp - inp.max(0)
        sm = np.exp(maxsub)
        return  sm/sm.sum(axis=0)

    def updateOutput(self, inp):
        self.output = self.softmax(inp)
        return self.output 

    def updateGradInput(self, inp, gradOutput):
        """ Assumes updateOutput was called previously, uses self.output as
            softmax of last inp on foreward call"""
        if self.output is not None:
            sm = self.output
        else:
            sm = self.softmax(inp) 
        smeye = (sm[:,None].T*np.eye(len(sm))).T
        self.gradInput = gradOutput *(smeye - np.einsum('np,mp->nmp',sm,sm))


class ReLU(Layer):
    """ Rectified Linear activation layer """

    def updateOutput(self, inp):
        """inp if inp>0 else 0"""
        self.mask = int(inp>0.)
        self.output = self.mask*inp
        return self.output

    def updateGradInput(self, inp, gradOutput):
        self.gradInput = gradOutput*self.mask


class PReLU(Layer):
    """ Parametrized Rectified Linear activation layer"""
    def __init__(self, n_hidden, a = None):
        self.n_hidden = n_hidden
        self.a = a or np.random.rand(self.n_hidden)
        self.grada = np.zeros_like(self.a)
        super(PReLU, self).__init__()

    def updateOutput(self, inp):
        positive = inp > 0
        self.posmask = int(positive)
        self.negmask = int(not positive)*self.a[:,None]
        self.output = (self.posmask + self.negmask)*inp
        return self.output

    def updateGradInput(self, inp, gradOutput):
        self.gradInput = gradOutput * (self.posmask + self.negmask) 

    def accGradInput(self, inp, gradOutput, scale=1.0):
        self.grada += scale * gradOutput * (self.posmask + self.negmask)


class Sigmoid(Layer):
    """ Sigmoid activation """

    def updateOutput(self, inp):
        self.output = expit(inp)
        return self.output

    def updateGradInput(self, inp, gradOutput):
        """ Assumes forward was called before backward """
        self.gradInput = gradOutput * self.output*(1.-self.output)


class Tanh(Layer):
    """ Hyperbolic tangent activation layer """

    def updateOutput(self, inp):
        self.output = np.tanh(inp)
        return self.output

    def updateGradInput(self, inp, gradOutput):
        self.gradInput = gradOutput * (1.-self.output*self.output)


class Dropout(Layer):
    """ Removes connections while training to promote
        sparsity and regularize learning process."""
    def __init__(self, p=0.5):
        self.p = p
        super(Dropout, self).__init__()

    def updateOutput(self, inp):
        if self.train:
            self.mask = int(np.random.rand(*inp.shape) > self.p)/(1.-self.p)
            self.output = self.mask*inp
        else:
            self.output = inp
        return self.output

    def updateGradInput(self, inp, gradOutput):
        if self.train:
            self.gradInput = gradOutput * self.mask
        else:
            self.gradInput = gradOutput





