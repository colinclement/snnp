import numpy as np
import h5py
import gzip
import cPickle as pickle
import sys
sys.path.append('../')
from network import *
from layers import *
from criterion import *

def getJacobianRow(row, net, data):
    net.forward(data[0][:,row], data[1][row])
    net.zeroGradParameters()
    net.backward()
    return net.getParameters()[1]

data = pickle.load(gzip.open('../mnist.pkl.gz','r'))
train = [data[0][0].T, data[0][1]]
nimages = len(train[1])

networkfilenames = ['trained_networks/2Layer_ReLU_MNIST.pkl',
                    'trained_networks/2Layer_Sigmoid_MNIST.pkl']

with h5py.File('2Layer_training_Jacobians.hdf5', 'a') as f:
    grp = f.create_group('jacobians')
    for nfname in networkfilenames:
        with open(nfname, 'r') as infile:
            NN = pickle.load(infile)
        print 'loaded', NN, 'with layers', NN.layers
        nparams = len(NN.getParameters()[1])
        jacrow = lambda row: getJacobianRow(row, NN, train)
        typename = nfname.split('_')[-2]
        print typename
        jac = grp.create_dataset(typename, shape=(nimages, nparams),
                                 dtype='f', chunks=True, compression='gzip')
        print jac, jac.shape
        for i in xrange(nimages):
            jac[i,:] = jacrow(i)
            if i % 500==0:
                jac.file.flush()
                print 'Saved row', i



