# The Policy Gradient method - REINFORCE - Bart Kowal

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from collections import deque

from PG_config import alpha, gamma, hidden_size, num_training_episodes

class PolicyNetwork(nn.Module):
    """
    Feedforward neural network in PyTorch for REINFORCE policy gradient method.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        pass

    def forward(self):
        pass
