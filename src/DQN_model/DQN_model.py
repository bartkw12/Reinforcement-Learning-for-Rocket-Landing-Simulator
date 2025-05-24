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
        super(DQN, self).__init__()
        # Dense Layers
        self.layer1 = torch.nn.Linear(state_size, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        # Outputs Q-values, one for each possible action.
        self.layer_out = torch.nn.Linear(hidden_size, action_size)

    def forward(self, state):
        '''
        Defines how data flows through the NN. Using ReLU activation to introduce non-linearity.
        '''
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        actions = self.layer_out(x)
        return actions

class ReplayBuffer:
    '''
    This class stores experiences and sample mini-batches for training.
    '''
    def __init__(self, buffer_size=buffer_size):
        '''
        Initializes the replay buffer with a specified buffer size.
        '''
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # deque automatically discards the oldest experiences
        self.size = 0  # Tracks the number of experiences in the buffer

    def add(self, state, action, reward, next_state, done):
        '''
        Adds a new experience to the replay buffer.
        '''
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        self.size = min(self.size + 1, self.buffer_size)  # do not exceed buffer size

    def sample(self, batch_size):
        '''
        Randomly samples a batch of experiences from the replay buffer.
        '''
        states, actions, rewards, next_states, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(states), actions, rewards, np.stack(next_states), done

    def __len__(self):
        '''
        Returns the current number of experiences in the buffer.
        '''
        return len(self.buffer)

class DQNAgent:
    '''
    Deep Q-Learning agent that uses a Deep Q-Learning Network and a replay memory to solve Lunar Lander env.
    '''
    def __init__(self, state_size=8, action_size=4):

        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Select GPU/CPU for training
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_size = action_size
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.memory = ReplayBuffer(buffer_size)  # Initialize the replay buffer memory