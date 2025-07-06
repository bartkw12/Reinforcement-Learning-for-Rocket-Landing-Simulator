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
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state):
        x = self.relu(self.layer1(state))
        logits = self.layer2(x)
        return logits

class ReinforceAgent:
    def  __init__(self, env, hidden_dim=hidden_size, gamma=gamma, lr=alpha):

        # initialize environment
        self.env = env

        self.lr = alpha
        self.gamma = gamma
        self.hidden_dim = hidden_size
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.policy = PolicyNetwork(self.state_dim, hidden_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

    def get_action(self):
        pass

    def compute_returns(self):
        pass

    def update_policy(self):
        pass