import numpy as np
from scipy.linalg import qr, svdvals
from scipy.sparse.linalg import svds

def streamRandomSVD(stream, k, q = 1, row_slice = range(100)):
    """
    Randomize Subspace Iteration SVD due to Halko, 
    Martinsson, and Tropp arXiv:0909.4061.
    input:
        stream : function with takes integers and returns one
                 row of the matrix to be decomposed
        k : int, the desired rank of the decomposition
        q : Number of passes over data will be 2q+1
        row_slice: iterator over rows of stream
        
    output: S - Singular Values of the streamed matrix
    """
    row0 = stream(row_slice[0])
    omega = np.random.randn(len(row0),k)
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

    B = row0[:,None,None]*Q[0]
    for n in row_slice[1:]:
        B += stream(n)[:,None,None]*Q[n]
    B = B.sum(1)
    Svals = svdvals(B, overwrite_a = True)
    u, s, vt = svds(B, k=1) #top singular vectors
    return u, Svals, vt

