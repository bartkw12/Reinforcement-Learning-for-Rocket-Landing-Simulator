# Hyperparameters setting for DQN RL

# Deep Q-Learning
alpha = 1e-3                  # learning rate - try 0.01 for more gradual learning
gamma = 0.99                  # discount factor - for more short term focused tasks try 0.8
epsilon = 1                   # exploration rate - full exploration rate at beginning = 1
epsilon_decay = 0.995         # exploration decay
num_training_episodes = 1000  # Define the number of training episodes

# DQN NN
hidden_size = 64
buffer_size = 10000
batch_size = 64  # for learning from the replay memory.

