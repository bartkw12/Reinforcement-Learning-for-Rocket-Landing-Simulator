# Hyperparameters setting for Q-learning

alpha = 0.5                  # learning rate - try 0.01 for more gradual learning
gamma = 0.99                 # discount factor - for more short term focused tasks try 0.8
epsilon = 1                  # exploration rate - full exploration rate at beginning = 1
epsilon_decay = 0.998        # exploration decay
num_training_episodes = 100  # define the number of training episodes