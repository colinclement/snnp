import numpy as np
import cPickle as pickle
import gzip
import sys, os
sys.path.append('../')
from network import Network
from layers import *
from criterion import *
from optimize import *
from randSVD import *
from datetime import datetime

networkfilename = 'trained_networks/2Layer/2Layer_ReLU_MNIST.pkl'
#networkfilename = 'trained_networks/2Layer/2Layer_Sigmoid_MNIST.pkl'

with open(networkfilename, 'r') as infile:
    NN = pickle.load(infile)

data = pickle.load(gzip.open('../mnist.pkl.gz', 'r'))
train = [data[0][0].T, data[0][1]]
nimages = len(train[1])
nparams = len(NN.getParameters()[1])

def getJacobianRow(n, net=NN, data=train):
    net.forward(data[0][:,n], data[1][n])
    net.zeroGradParameters()
    net.backward()
    return net.getParameters()[1]

networktype = networkfilename.split('_')[-2]
memfilename = 'trained_networks/2Layer/memmap2Layer_{}_jac.dat'.format(networktype)

if os.path.isfile(memfilename):
    print memfilename, " exists!"
else:
    fp = np.memmap(memfilename, dtype='float32',
            mode='w+', shape=(nimages, nparams))
    
    def assign(n):
        fp[n,:] = getJacobianRow(n)
    start = datetime.now()
    [assign(n) for n in range(nimages)]
    print 'Saving jacobian took', datetime.now()-start
    
    del fp #flush and save

fp = np.memmap(memfilename, dtype='float32',
        mode='r', shape=(nimages, nparams))

start = datetime.now()
u, Svals, vt = streamRandomSVD(lambda n: fp[n], 500, 1, range(10000))
print 'Singular values took ', datetime.now()-start

singfile='trained_networks/2Layer/2Layer_{}_jac_singular_values.npy'.format(networktype)

with open(singfile, 'w') as outfile:
    np.savez(outfile, u=u, Svals=Svals, vt=vt)

