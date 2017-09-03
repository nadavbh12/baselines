import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
from operator import mul
import numpy as np


class Mlp(nn.Module):
    def __init__(self, input_dim, num_actions, hiddens, *args, **kwargs):
        super(Mlp, self).__init__()

        if not isinstance(input_dim, list):
            raise ValueError('Input size must be a list of sizes')
        if not isinstance(hiddens, list):
            raise ValueError
        if num_actions <= 0:
            raise ValueError('num_actions must be larger than 0')

        self.num_actions = num_actions

        layers = []
        self.input_flat_size = reduce(mul, input_dim, 1)
        sizes = [self.input_flat_size] + hiddens + [num_actions]
        for i in range(len(sizes)-1):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU(inplace=True)]

        self.net = nn.Sequential(*layers)

    def _init_layers(self):
    #     TODO: implement
        pass

    def forward(self, x):
        x = x.view(-1, self.input_flat_size)
        x = self.net(x)
        return x


def mlp(input_dim, num_actions, hiddens, *args, **kwargs):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    input_dim: [int]
        list of the dimensions of the input
    num_actions: int
        number of possible actions
    hiddens: [int]
        list of sizes of hidden layers

    Returns
    -------
    q_func: nn.Module
        q_function for DQN algorithm.
    """
    model = Mlp(input_dim, num_actions, hiddens, *args, **kwargs)
    return model


class CnnToMlp(nn.Module):
    def __init__(self, input_dim, num_actions, convs, hiddens, dueling=False, *args, **kwargs):
        super(CnnToMlp, self).__init__()

        if not isinstance(input_dim, list):
            raise ValueError('Input size must be a list of sizes')
        if not isinstance(hiddens, list):
            raise ValueError
        if not isinstance(convs, list):
            raise ValueError
        if num_actions <= 0:
            raise ValueError('num_actions must be larger than 0')

        self.num_actions = num_actions

        self.dueling = dueling
        layers = []
        in_channels = input_dim[0]
        for out_channels, kernel_size, stride in convs:
            layers += [nn.Conv2d(in_channels, out_channels, kernel_size, stride), nn.ReLU(inplace=True)]
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        layers = []
        conv_flat_size = CnnToMlp.get_flat_features(input_dim, self.features)
        sizes = [conv_flat_size] + hiddens + [num_actions]
        for i in range(len(hiddens)-1):
            layers += [nn.Linear(sizes[i], sizes[i+1]), nn.ReLU(inplace=True)]
        self.classifier = nn.Sequential(*layers)

        if dueling:
            layers = []
            sizes = [conv_flat_size] + hiddens + [1]
            for i in range(len(hiddens) - 1):
                layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU(inplace=True)]
            self.value_estimator = nn.Sequential(*layers)

    @staticmethod
    def get_flat_features(in_size, features):
        # print('in_size = {}, features_size = {}'.format(in_size, features.size()))
        f = features(Variable(torch.ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))

    def _init_layers(self):
    #     TODO: implement
        pass

    def forward(self, x):
        x = self.features(x)
        action_values = self.classifier(x)
        if self.dueling:
            value = self.value_estimator(x)
            action_value_mean = action_values.mean()
            action_value_centered = action_values - action_value_mean.expand_as(action_values)
            return value + action_value_centered
        else:
            return action_values


def cnn_to_mlp(input_dim, num_actions, convs, hiddens, dueling=False, *args, **kwargs):
    """This model takes as input an observation and returns values of all actions.

    Parameters
    ----------
    input_dim: [int, int, int]
        list of the dimensions of the input
        (num_channels, height, width)
    num_actions: int
        number of possible actions
    convs: [(int, int int)]
        list of convolutional layers in form of
        (out_channels, kernel_size, stride)
    hiddens: [int]
        list of sizes of hidden layers
    dueling: bool
        if true double the output MLP to compute a baseline
        for action scores

    Returns
    -------
    q_func: nn.Module
        q_function for DQN algorithm.
    """

    return CnnToMlp(input_dim, num_actions, convs, hiddens, dueling, *args, **kwargs)
