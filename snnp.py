import numpy as np
import activations as act
import cPickle as pickle
import gzip
import matplotlib.pyplot as plt

class Layer(object):
    """ A single layer of a neural network """
    def __init__(self, W, b, activ, d_activ, **kwargs):
        self.activ = activ
        self.d_activ = d_activ
        self.W = W 
        self.b = b 
        self.N_neurons = W.shape[1] #no. of columns in W

    def forward(self, inp, tl):
        """ Given an input, compute the forward pass """
        self.newinp = inp.copy()
        self.newtl = tl
        self.newact = (np.dot(self.W, inp).T+self.b).T
        return self.activ(self.newact)

    def backward(self, act = None, tl=None):
        """ Given an gradient above, do backprob back
        through the layer """
        return self.d_activ(act or self.newact)


class LossLayer(Layer):
    """ Represents a loss layer at the top of the network """
    def __init__(self, N):
        """
        N : number of nodes
        """
        self.W = np.eye(N) #No weights
    
    def forward(self, inp, tl):
        self.newinp = inp
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
        self.newact = inp
        self.newtl = tl
        return self.newact

    def backward(self, inp=None, tl = None):
        return inp or self.newact


class Network(object):
    """ A collection of Layers """
    def __init__(self, data, labels, widthlist, lr = 0.1, activ=act.ReLU,
                 d_activ=act.d_ReLU, conv = 1E-2):
        """
        data : Training data of shape (N_samples, N_dimensions)
        labels : Integer labels for training data of shape (N_samples)
        widthlist : list of layer widths
        lr : learning rate, gradient step size
        activ : vectorized function which takes inputs and maps them
                e.g. sigmoid, ReLU, maxout
        d_activ: The derivative of activ
        conv : Convergence criterion, norm of [dW, db]
        """
        self.data = data.T
        if data.shape[0] < data.shape[1]:
            print 'Check data shape'
        self.labels = labels
        self.N_classes = len(set(labels))
        self.train_labels = np.eye(self.N_classes)
        self.widthlist = list(widthlist)
        self.lr = lr
        self.activ = activ
        self.d_activ = d_activ
        self.rng = np.random.RandomState()
        self.rng.seed(92089)
        connect = [self.data.shape[0]]+self.widthlist
        
        #Create layers
        self.layers = [InputLayer(data)]
        for n in range(1,len(widthlist)+1):
            W = self.rng.randn(connect[n], connect[n-1])
            b = self.rng.randn(connect[n])
            self.layers += [Layer(W, b, self.activ, self.d_activ)]
        self.layers += [LossLayer(widthlist[-1])]
        self.N_l = len(self.layers)

    def feedforward(self, inp, label):
        """
            inp : sample training data
        returns cost, cross entropy of softmax
        """
        trainlabel = self.train_labels[label]
        inptemp = inp.copy()
        for L in self.layers:
            inptemp = L.forward(inptemp, label)
        return inptemp

    def get_parameters(self):
        """
        Grab and arrange weights and biases into one vector
        """
        self.paramlist = np.array([])
        for L in self.layers[1:-1]:
            self.paramlist = np.r_[self.paramlist, L.W.flat, L.b]
        return self.paramlist

    def backpropogate(self, inp, label):
        """
        outlist : list of layer outputs, output from
                  self.feedforward(inp)
        label : label corresponding to inp
        """
        trainlab = self.train_labels[label]
        cost = self.feedforward(inp, label)
        self.d_params = np.array([])
        error = self.layers[-1].backward()
        for nl in range(self.N_l-2, 0, -1):
            L = self.layers[nl]
            error = np.dot(error, self.layers[nl+1].W*L.backward())
            dW = -self.lr*np.outer(error, L.newinp)
            db = -self.lr*error.copy()
            print error.shape, dW.shape, db.shape
            self.d_params = np.r_[dW.flat, db, self.d_params]
            L.W += dW
            L.b += db

    def train(self, N_epochs):
        pass

if __name__=='__main__':
    try:
        mnist = pickle.load(gzip.open('mnist/mnist.pkl.gz','r'))
        train = mnist[0]
        test1 = mnist[1]
        test2 = mnist[2]
        nn = Network(test1[0], test1[1], widthlist=[10,10])
    except IOError:
        print 'mnist/mnist.pkl.gz not found\n'
        pass

