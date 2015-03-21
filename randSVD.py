import numpy as np
from scipy.linalg import qr, svd

def streamRandomSVD(stream, k, q = 1, row_slice = xrange(100)):
    """
    Randomize Subspace Iteration SVD due to Halko, 
    Martinsson, and Tropp arXiv:0909.4061.
    input:
        stream : function with takes integers and returns one
                 row of the matrix to be decomposed
        k : int, the desired rank of the decomposition
        q : Number of passes over data will be 2q+1
        row_slice: iterator over rows of stream
        
    output: U, S, Vt - Singular Value Decomposition
    """
    row0 = stream(row_slice[0])
    omega = np.random.rand(len(row0),k)
    Y = np.array([row0.dot(omega)])
    for n in row_slice[1:]:
        Y = np.r_[Y,[stream(n).dot(omega)]]
    Q,R = qr(Y, mode='economic')
    
    for j in range(1,q+1):
        Ytilde = row0[:,None,None]*Q[0]
        for n in row_slice[1:]:
            Ytilde += stream(n)[:,None,None]*Q[n]
        Ytilde = Ytilde.sum(1)
        Qt, Rt = qr(Ytilde, mode='economic')
        
        Y = np.array([row0.dot(Qt)])
        for n in row_slice[1:]:
            Y = np.r_[Y, [stream(n).dot(Qt)]]
        Q,R = qr(Y, mode='economic')

    B = Q.T.dot(Y).dot(Qt.T)
    Utilde, S, Vt = svd(B)
    U = Q.dot(Utilde)
    return U, S, Vt

