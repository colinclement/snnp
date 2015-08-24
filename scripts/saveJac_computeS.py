import numpy as np
import cPickle as pickle
import gzip
import sys, os
sys.path.append('../src/')
from network import Network
from layers import *
from criterion import *
from optimize import *
from randSVD import *
from datetime import datetime

networkdirectory = '../trained_networks/2Layer/PReLU_initial_conds/'
datafile = '../mnist.pkl.gz'
NNfilename = 'Sigmoid.pkl'

networkfilename = os.path.join(networkdirectory, NNfilename)

networktype = NNfilename.split('.')[0]
memname = 'memmap_{}_jac.dat'.format(networktype)
memfilename = os.path.join(networkdirectory, memname)
singname = '{}_jac_singular_values.npy'.format(networktype)
singfile = os.path.join(networkdirectory, singname)

print 'Network type: ', networktype
print 'Memmap file name: ', memfilename
print 'Singular values file: ', singfile

with open(networkfilename, 'r') as infile:
    NN = pickle.load(infile)

data = pickle.load(gzip.open(datafile, 'r'))
train = [data[0][0].T, data[0][1]]
nimages = len(train[1])
nparams = len(NN.getParameters()[1])

def getJacobianRow(n, net=NN, data=train):
    net.forward(data[0][:,n], data[1][n])
    net.zeroGradParameters()
    net.backward()
    return net.getParameters()[1]

if os.path.isfile(memfilename):
    print memfilename, " exists!"
else:
    print 'Making memmap'
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


with open(singfile, 'w') as outfile:
    np.savez(outfile, u=u, Svals=Svals, vt=vt)

