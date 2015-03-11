import numpy as np

def softmax(z):
    sm = np.exp(z)
    return (sm.T/sm.sum(axis=-1)).T

#Need to vectorize these functions
def d_softmax(z):
    """
    Gradient of softmax function evaluated at z
    """
    sm = softmax(z)
    eye = np.eye(len(z))
    return sm*eye-np.outer(sm, sm)

def crossEntropy(p, t):
    return np.sum(t*np.nan_to_num(np.log(t/p)))

def crossEntropy_of_softmax(p, t):
    return crossEntropy(softmax(p), t)

def d_crossEntropy_of_softmax(p, t):
    return p-t

## Activations:

def ReLU(x):
    return np.maximum(0.,x)

def d_ReLU(x):
    return (x>=0).astype('float')

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s*(1-s)





