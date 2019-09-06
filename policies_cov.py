from torch import sigmoid
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
STATE_DIM = 2*2+3*2+1
ACTION_DIM = 2


def mean_std(m, s):
    """Problem specific restrinctions on predicted mean and standard deviation."""
    mean = Tensor([400-120, 40]) * sigmoid(m) + Tensor([120, 0])
    std = Tensor([20, 10]) * sigmoid(s)
    return mean, std


class NeuralNetwork(nn.Module):

    def __init__(self, hidden_dim):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(STATE_DIM, hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear3 = nn.Linear(hidden_dim, ACTION_DIM, bias=True)
        self.linear3_ = nn.Linear(hidden_dim, ACTION_DIM, bias=True)
        self.linear4 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim, bias=True)

#        self.linear4 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, inputs):
        x = inputs
        x = F.leaky_relu(self.linear1(x), 0.1)#F.tanh(self.linear1(x))#
        x = F.leaky_relu(self.linear2(x), 0.1)#F.tanh(self.linear2(x))
        x = F.leaky_relu(self.linear4(x), 0.1)#F.tanh(self.linear2(x))
        x = F.leaky_relu(self.linear5(x), 0.1)#F.tanh(self.linear2(x))

#        x = F.tanh(self.linear4(x))#leaky_relu(self.linear4(x), 0.1)
        m, s = self.linear3(x), self.linear3_(x)
        mean, std = mean_std(m, s)
        return mean, std


class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(STATE_DIM, ACTION_DIM)
        self.linear_ = nn.Linear(STATE_DIM, ACTION_DIM)

    def forward(self, x):
        m, s = self.linear(x), self.linear_(x)
        mean, std = mean_std(m, s)
        return mean, std
