# Hyperparameters setting

# REINFORCE Policy Gradient Method
alpha = 1e-3                  # learning rate - try 0.01 for more gradual learning
gamma = 0.99                  # discount factor - for more short term focused tasks try 0.8
num_training_episodes = 1000  # Define the number of training episodes

# Hyperparameters specific to REINFORCE
# NN
hidden_size = 128