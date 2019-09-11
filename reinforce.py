from os.path import join
import os
import matplotlib.pyplot as plt
# import torch
# import torch.optim as optim
from Dynamic_system import integrator_model, specifications, DAE_system
from policies_cov import NeuralNetwork
from utilities_theta_single_rnn import run_episode, training, select_action
import numpy as np
import torch
from torch.distributions import Normal

from matplotlib import rc


import torch.optim as optim
from torch import Tensor
import datetime


eps = np.finfo(np.float32).eps.item()
torch.manual_seed(666)
#---------- Initialize Folder --------------------------------------------------------------#
now = datetime.datetime.now()

np.random.seed(seed=0)
path1 = 'rnn-backoff-different-obj-different-initial-obj'+str(now.date())+str(now.hour)+str(now.minute)
os.mkdir(path1)
#-------------------------------------------------------------------------------------------#
#----- Define size of each layer of  policy network and other learning parameters ----------#
hidden_layers_size = 20
policy1 = NeuralNetwork(hidden_layers_size)


epochs = 100
epoch_episodes = 500
optimizer1 = optim.Adam(policy1.parameters(), lr=0.01)
#-------------------------------------------------------------------------------------------#
#--------------------------Define the model for the physical system ------------------------#
nk, tf, x0, Lsolver, c_code = specifications()
xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states,\
algebraics, inputs, nd, na, nu, nmp, modparval = DAE_system()

dun = np.array(modparval)
F = integrator_model()
#--------------------------------------------------------------------------------------------#


# -----------------------Perform the main training ------------------------------------------#
epoch_rewards0, rewards_std_record0 = training(policy1, optimizer1, epochs, epoch_episodes, dun, F, nk,path1)
#--------------------------------------------------------------------------------------------#
print('Done')