# The DQN Algorithm

import gymnasium as gym
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from collections import deque
from DQN_config import hidden_size, buffer_size, batch_size, gamma, alpha, epsilon, epsilon_decay, num_training_episodes

class DQN(torch.nn.Module):
    '''
    Feedforward neural network in PyTorch that can be used as a Q-function approximator for Deep Q-learning.
    '''
    def __init__(self, state_size=8, action_size=4, hidden_size=hidden_size):
        '''
        Define architecture of NN (two hidden layers).
        '''