"""
Optimize.py - Train SNNP Networks classes

author: Colin Clement
date: 2015-03-19

"""

from __future__ import division
import numpy as np
import scipy as sp
try:
    from sfo import SFO
    flagSFO = True
except ImportError:
    print "Please install Sum-of-Functions-Optimizer from\n \
           https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer"
    flagSFO = False

def getBatch(inp, target, size):
    """
        Creates an iterator which returns chunks of inp and 
        target of length size
    """
    n = 0
    while True:
        low = n*size
        high = (n+1)*size
        if high >= len(target): #>= important
            n = 0
        else:
            n += 1
        yield inp[:,low:high], target[low:high]


class SGD(object):
    def __init__(self, network, data, targets, batchsize=100,
                 lr=1E-3, **kwargs):
        """
        Stochastic Gradient Descent for SNNP.
        input:
            network : Instance of Network class
            data : input data of shape (dimension, N_samples)
            targets : training targets of len N_samples
            batchsize : Number of samples per SGD step.
            lr : learning rate or gradient stepsize
        kwargs:
            shuffle : Boolean for shuffling data after each Epoch
        """
        self.Net = network
        self.inputs = data
        self.targets = targets
        self.batchsize = batchsize
        self.lr = lr
        if 'shuffle' in kwargs:
            self.shuffle = kwargs['shuffle']
        else:
            self.shuffle = True
        self.batchiter = self.getBatchIter(self.batchsize, self.shuffle)
        self.costs = []

    def getBatchIter(self, batchsize, shuffle=True):
        sh = range(len(self.targets))
        if shuffle:
            np.random.shuffle(sh)
        return getBatch(self.inputs[:,list(sh)], self.targets[list(sh)],
                        self.batchsize)

    def batchStep(self):
        inp, targ = self.batchiter.next()
        self.costs.append(self.Net.trainSGD(inp, targ, self.lr))

    def changeBatchsize(self, batchsize):
        self.batchsize = batchsize
        self.batchiter = self.getBatchIter(self.batchsize, self.shuffle)
        self.costs = []

    def train(self, N_epochs):
        self.Net.training()
        N_samples = len(self.targets)
        batchPerEpoch = int(np.ceil(N_samples/self.batchsize))
        for epoch in xrange(N_epochs):
            for batch in xrange(batchPerEpoch):
                self.batchStep()
            if self.shuffle:
                self.batchiter = self.getBatchIter(self.batchsize, self.shuffle)
        self.Net.evaluate()
        

class SFOmin(object):
    def __init__(self, network, data, targets, conv = 0.01,
                **kwargs):
        """
            Train network on data and targets with the
            Sum-of-Functions-Optimizer (arXiv:1311.2115)
            input:
                network : Instance of Network
                data : shape (dimension, N_samples)
                targets : array of targets (N_samples)
                conv : Stops if cost change is less than this 
            **kwargs:
                maxiters : maximum number of iterations to limit
                           self.optimizeToConv
                iprint : (0,1,2) verbosity level of SFO
        """
        self.Net = network
        self.inputs = data
        self.targets = targets
        self.conv = conv
        self.initial_p, _ = self.Net.getParameters()
        if 'maxiters' in kwargs:
            self.maxiters = kwargs['maxiters']
        else:
            self.maxiters = 30
        if 'iprint' in kwargs:
            self.iprint = kwargs['iprint']
        else:
            self.iprint = 0
        
        self.optimizer = self.getOptimizer()

    def getSFOBatches(self):
        Nsamp = len(self.targets)
        N_batch = max(25, int(np.sqrt(Nsamp)/10.))
        bsize = int(np.ceil(Nsamp//N_batch))
        slices = [[i*bsize, (i+1)*bsize] for i in range(N_batch)]
        return [(self.inputs[:,l:h], self.targets[l:h]) for l, h in slices]
        
    def getOptimizer(self):
        self.batches = self.getSFOBatches()
        return SFO(self.Net._getCost_dCost, self.initial_p, self.batches,
                  display = self.iprint)

    def optimizeToConv(self):
        deltaCost = 1E6 #arbitrary large number
        niters = 0
        self.Net.training()
        while deltaCost > self.conv:
            self.optimizer.optimize(1) #One pass over the data
            hist = self.optimizer.hist_f_flat
            deltaCost = np.max(np.abs(hist[-1] - hist[-2]))
            niters += 1
            if niters > self.maxiters:
                print '{} iterations passed, deltaCost = {}'.format(self.maxiters, 
                                                                    deltaCost)
                break
        self.Net.evaluate()
        return niters

