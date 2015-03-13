"""
SNNP - Simple Neural Networks with Python

Author: Alex Alemi
Date: 2015-03-11

A simple framework for exploring and checking 
out neural networks, based on the torch nn library
"""

import numpy as np
from scipy.misc import logsumexp
from scipy.special import expit

class Layer(object):
    """ A basic object for the neural network, denotes
    a single layer of the network with its own activation
    function.
    """

    def __init__(self):
        """ Initialization """

        # The cached output
        self.output = None

        # The gradient with respect to the input, cached
        self.grad_input = None 

        # By default, we will assume we have weights and biases
        self.weights = None
        self.bias = None

        self.grad_weights = None
        self.grad_bias = None

        self.train = False

    def forward(self, inp): 
        """ Maps the input layer forward through the network,
        returns the result of the activation """
        return self.update_output(inp)

    def backward(self, inp, grad_output, scale=1.0):
        """ Do a backward pass through the network, given
        the input and the gradient at the output of the network

        Returns: grad_input
        """
        self.update_grad_input(inp, grad_output)
        self.acc_grad_parameters(inp, grad_output, scale)
        return self.grad_input

    def update_output(self, inp):
        """ Change the output of the network based on the input """
        self.output = inp
        return self.output

    def update_grad_input(self, inp, grad_output):
        """ Update the grad_input variable """
        return self.grad_input

    def acc_grad_parameters(self, inp, grad_output, scale=1.0):
        """ Accumulate gradients for the paramters """

    def zero_grad_parameters(self):
        """ Reset the grad of the parameters """

        ps, gps = self.parameters()
        for gp in gps:
            gp.fill(0)

    def update_parameters(self, learning_rate):
        """ Called after update_grad_parameters, actually
        adjusts the parameters of the layer """
        ps, gps = self.parameters()
        for p,gp in zip(ps,gps):
            p -= learning_rate*gp

    def acc_update_grad_parameters(self, inp, grad_output, learning_rate):
        """ Convience function that does both an update
        on the grad parameters and an update """

    def parameters(self):
        """ Returns the (weights, grad_weights) for the layer """
        if self.weights is not None and self.bias is not None:
            return [self.weights, self.bias], [self.grad_weights, self.grad_bias]
        elif self.weights is not None:
            return [self.weights], [self.grad_weights]
        elif self.bias is not None:
            return [self.bias], [self.grad_bias]
        return [], []

    def training(self):
        self.train = True

    def evaluate(self):
        self.train = False

class Linear(Layer):
    """ A simple Rectified Linear Layer """
    def __init__(self, fan_in, fan_out):
        self.fan_in = fan_in
        self.fan_out = fan_out

        # Do the PreLU paper initialization
        std = np.sqrt(2.0/fan_in)
        self.weights = std*np.random.randn(fan_out, fan_in)
        self.bias = np.zeros(fan_out)

        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

        self.output = self.bias.copy()

    def update_output(self, inp):
        self.output = self.weights.dot(inp) + self.bias[:,None]
        return self.output

    def update_grad_input(self, inp, grad_output):
        self.grad_input = (grad_output).T.dot(self.weights).T

    def acc_grad_parameters(self, inp, grad_output, scale=1.0):
        self.grad_weights += (grad_output[:,None,:] * inp[None,:,:]).mean(-1)
        self.grad_bias += grad_output.mean(-1)

class ReLu(Layer):
    """ The parameterless rectified linear activation function """

    def update_output(self, inp):
        z = inp.copy()
        z[z<0] = 0
        self.output = z
        return self.output

    def update_grad_input(self, inp, grad_output):
        self.grad_input = grad_output * (inp > 0)

class Tanh(Layer):
    """ The parameterless rectified linear activation function """

    def update_output(self, inp):
        self.output = np.tanh(inp)
        return self.output

    def update_grad_input(self, inp, grad_output):
        self.grad_input = grad_output * ( 1 - self.output**2 )

class Sigmoid(Layer):

    def update_output(self, inp):
        self.output = expit(inp)
        return self.output

    def update_grad_input(self, inp, grad_output):
        self.grad_input = grad_output * ( self.output * ( 1- self.output) ) 

class LogSoftMax(Layer):

    def update_output(self, inp):
        self.output = inp - logsumexp(inp, axis=0)
        return self.output

    def update_grad_input(self, inp, grad_output):
        a = np.exp(inp)
        self.grad_input = grad_output - grad_output.sum(0)[None,:] * a / a.sum(0)[None,:]

class Dropout(Layer):

    def __init__(self, p = 0.5):
        self.p = p
        super(Dropout, self).__init__()

    def update_output(self, inp):
        # If we are training, apply a mask
        if self.train:
            self.mask = (np.random.rand(*inp.shape) < self.p) / (1-self.p)
            self.output = self.mask * inp 
            return self.output
        else:
            self.output = inp
            return self.output

    def update_grad_input(self, inp, grad_output):
        if self.train:
            self.grad_input = self.mask * grad_output
        else:
            self.grad_input = grad_output

