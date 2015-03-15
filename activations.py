import numpy as np

def softmax(z):
    """
    z : array of shape (N_dim, N_samples)
    broadcasts over N_samples
    """
    maxsub = z - np.max(z, axis=0)
    sm = np.exp(maxsub)
    return sm/sm.sum(axis=0)

#Need to vectorize these functions
def d_softmax(z):
    """
    Gradient of softmax
    z : array of shape (N_dim, N_samples)
    broadcasts over N_samples
    """
    if len(z.shape)==1:
        z = z[:,None]
    sm = softmax(z)
    smeye = (sm[:,None].T*np.eye(len(z))).T
    return smeye-np.einsum('np,mp->nmp',sm, sm)

def crossEntropy(p, t):
    return np.sum(t*np.nan_to_num(np.log(t)-np.log(p)))

def crossEntropy_of_softmax(p, t):
    return crossEntropy(softmax(p), t)

def d_crossEntropy_of_softmax(p, t):
    return softmax(p)-t

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





