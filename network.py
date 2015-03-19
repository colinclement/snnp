"""
SNNP - Simple Neural Networks with Python

author: Colin Clement
date: 2015-03-13

Neural network Network class for use with Layers from layers.py 
"""

import numpy as np

class Network(object):

    def __init__(self):
        self.layers = []
        self.output = None
        self.criterion = None
        self.inputlist = []
        self.train = True

    def forward(self, inp, targets):
        if np.prod(inp.shape) == 0 and len(targets)==0:
            print 'Warning, no input or targets'
            return None
        if len(inp.shape)==1:
            inp = inp[:,None]
        self.last_targets = targets
        self.inputlist = [inp.copy()]
        for L in self.layers:
            inp = L.forward(inp)
            self.inputlist += [inp.copy()]
        inp = self.criterion.forward(inp, targets)
        self.output = inp
        return self.output 

    def backward(self, inp = None, targets = None):
        if inp is None:
            inp = self.inputlist
        if targets is None:
            targets = self.last_targets
        gradOutput = self.criterion.backward(inp[-1], targets)
        for L, Linp in zip(self.layers[::-1], inp[-2::-1]):
            gradOutput = L.backward(Linp, gradOutput)
        return gradOutput 

    def zeroGradParameters(self):
        for L in self.layers:
            L.zeroGradParameters()

    def updateParameters(self, learningrate=0.1):
        for L in self.layers:
            L.updateParameters(learningrate)
    
    def training(self):
        self.train = True
        for L in self.layers:
            L.training()

    def evaluate(self):
        self.train = False
        for L in self.layers:
            L.evaluate()

    def getParameters(self):
        """ Returns flattened array for learable parameters and their
        gradients"""
        plist, gplist = np.array([]), np.array([])
        for L in self.layers:
            ps, gps = L.getParameters()
            for p, gp in zip(ps, gps):
                plist = np.r_[plist, p]
                gplist = np.r_[gplist, gp]
        return plist, gplist

    def _setParameters(self, x):
        """ Set learnable parameters to x, for use in an external gradient
            calculation for checking analytic gradients
            NOTE: Numeric gradient will be off by a factor of the number of
            data points used (backwards uses means instead of sums)"""
        run = 0
        for L in self.layers:
            if L.W is not None and L.b is not None:
                lws, lb = np.prod(L.W.shape), len(L.b)
                L.W = x[run:run+lws].reshape(L.W.shape)
                run += lws
                L.b = x[run:run+lb]
                run += lb

    def _getCost_dCost(self, x, inptargets):
        """ Computes cost and gradient for external optimization
            routine.
            input:
                x : learnable parameters in array of dimension 1
                inptargets : Two-element tuple containing
                        (data input of shape (dimension, N_samples)
                        integer array of shape (N_samples))
            returns criterion cost, gradient of parameters
        """
        inp, targets = inptargets
        self._setParameters(x)
        cost = self.forward(inp, targets)
        self.zeroGradParameters()
        gradOutput = self.backward()
        ps, gps = self.getParameters()
        return cost, gps

    def trainSGD(self, inp, targets, learningrate=1E-3):
        """
            Stochastic gradient descent step.
            input:
                inp : training data array shape (dimension, N_sample)
                targets : integer array of length (N_sample)
                learningrate : gradient stepsize
        """
        output = self.forward(inp, targets)
        if output is None:
            print 'Training halted, no data provided'
            return None
        self.zeroGradParameters()
        gradOutput = self.backward()
        self.updateParameters(learningrate)
        return output.sum() #Final cost

    




