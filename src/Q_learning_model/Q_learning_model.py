import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from state_discretizer import StateDiscretizer
from collections import deque
from Q_learning_config import alpha, gamma, epsilon, epsilon_decay, num_training_episodes


class LunarLanderAgent:
    def __init__(self):
        """
        Initialize your agent here.

        This method is called when you create a new instance of your agent.
        Use this method to Initializes the environment, the agent’s model (e.g., Q-table or neural network),
        and the optional state discretizer for Q-learning. Add any necessary initialization for model parameters here.
        """
        # TODO: Initialize your agent's parameters and variables

        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Get number of actions from environment
        # 4: do nothing, fire left orientation engine, fire main engine, fire right orientation engine
        self.num_actions = self.env.action_space.n

        # Initialize state discretizer if you are going to use Q-Learning
        self.state_discretizer = StateDiscretizer(self.env)

        # initialize Q-table or neural network weights
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Set learning parameters
        self.alpha = alpha / self.state_discretizer.num_tilings  # Learning rate per tiling
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate

        # Training performance tracking
        self.episode_rewards = []
        self.best_avg_reward = -float('inf')