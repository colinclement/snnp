
class Layer(object):
    """ A single layer of a neural network """

    def forward(inp):
        """ Given an input, compute the forward pass """

    def backward(inp):
        """ Given an gradient above, do backprob back
        through the layer """

class LossLayer(Layer):
    """ Represents a loss layer at the top of the network """

class InputLayer(Layer):
    """ Represents the input """

class Network(object):
    """ A collection of Layers """


