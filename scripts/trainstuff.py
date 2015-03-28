import numpy as np
import sys
sys.path.append('../src/')
from network import Network
from layers import *
from criterion import CrossEntropyCriterion
from optimize import SFOmin
import gzip
import cPickle as pickle

data = pickle.load(gzip.open('../mnist.pkl.gz', 'r'))
train = [data[0][0].T, data[0][1]]

NN = Network()
NN.criterion = CrossEntropyCriterion()
NN.training()

#layers = [Linear(784, 400, seed=92089), ReLU(), Linear(400, 10, seed=14850),
#          ReLU()]
#savefile = '../trained_networks/2Layer/PReLU_initial_conds/ReLU.pkl'
layers = [Linear(784, 400, seed=92089), Sigmoid(), Linear(400, 10, seed=14850),
          Sigmoid()]
savefile = '../trained_networks/2Layer/PReLU_initial_conds/Sigmoid.pkl'
NN.layers = layers
NN._initial_parameters = NN.getParameters()[0]

SFO = SFOmin(NN, train[0], train[1], iprint=2)

SFO.conv = 0.01
SFO.maxiters = 30

SFO.optimizeToConv()

NN.hist_f_flat = SFO.optimizer.hist_f_flat
NN.forward(data[1][0].T, data[1][1])
classes = np.argmax(NN.inputlist[-1], 0)
NN.test1_accuracy = np.mean(classes == data[1][1])
print '\nTrained accuracy = {}\n'.format(NN.test1_accuracy)
NN.forward(train[0][:,0], train[1][0]) #clear cached inputs
#Note to self: make pickling automatically delete this stuff

with open(savefile, 'wb') as outfile:
    pickle.dump(NN, outfile, -1)

