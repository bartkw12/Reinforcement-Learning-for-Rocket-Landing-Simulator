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

        # NN initialization
        # Initialize two NNs with the given state, action and hidden layer size
        # use networks on selected processor
        self.q_network = DQN(state_size, action_size, hidden_size).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_size).to(self.device)

        # Set identical weights for both NNs
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Initialize the optimizer for updating the DQNs parameters
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=alpha)

        # Set target network to evaluation mode
        # ensures that certain layers like dropout or batch normalization behave correctly during inference.
        self.target_network.eval()

        # Tracking best performance
        self.best_avg_reward = -float("inf")
        self.best_model_path = "best_model.pth"

    def select_action(self, state, testing=False):
        '''
        Given a state, select an action to take. The function should operate in training and testing modes:
            - Testing: Uses greedy policy.
            - Training: Uses epsilon-greedy policy.

            Args:
                state (array): The current state of the environment.
                testing (bool): If True, uses greedy policy; if False, uses epsilon-greedy policy.

            Returns:
                int: The action to take.
        '''
        # Convert the state to a PyTorch tensor and move it to the appropriate device (CPU or GPU)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # If testing (greedy policy)
        if testing:
            # Set the network to evaluation mode and get the Q-values for the given state
            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            # Choose the action with the highest Q-value (greedy)
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            # If training, use epsilon-greedy
            if random.random() < self.epsilon:
                # Choose a random action with probability epsilon
                action = random.choice(np.arange(self.action_size))
            else:
                # Choose the best action (greedy)
                self.q_network.eval()
                with torch.no_grad():
                    action_values = self.q_network(state)
                action = np.argmax(action_values.cpu().data.numpy())

        # Return the selected action (integer)
        return action

    def step(self, state, action, reward, next_state, done):
        '''
        Take a step in the environment, add the experience to the memory - update DQN
        '''
        # add the experience to memory
        self.memory.add(state, action, reward, next_state, done)

        # perform learning step when enough experiences are accumulated
        if len(self.memory) > self.batch_size:
            self.update_model()

    def update_model(self):
        '''
        Update the DQN based on experiences from the replay memory.
        '''
        # Sample random mini-batch of experiences from memory
        states, actions, rewards, next_states, done = self.memory.sample(self.batch_size)

        # transform NumPy arrays into PyTorch tensors - set GPU/CPU
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(np.array(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        done = torch.from_numpy(np.array(done).astype(np.uint8)).float().to(self.device)

        print(states, actions, rewards, next_states)

        # Extracting tensor of Q-values for actions taken
        q_values = self.q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Get max Q-value for the next states from target NN
        next_q_values = self.target_network(next_states).max(1)[0].detach()

        # Compute the TD target to get expected Q-values
        expected_q_values = rewards + self.gamma * next_q_values * (1 - done)

        # Compute the loss between the current and expected Q values
        # Huber loss - less sensitive to outliers
        loss_fn = torch.nn.SmoothL1Loss()
        loss = loss_fn(q_values, expected_q_values)

        # clear old gradients
        self.optimizer.zero_grad()

        # Backpropagate - computes gradients of the loss w.r.t. the Q-network’s parameters
        loss.backward()

    def sync_networks(self):
        '''
        Update the weights of the target NN to be identical to policy/Q NN (copy/paste).
        '''
        pass

    def load_agent(self, file_path):
        """
        Load Q-network's weights.
        """
        pass

    def train(self, num_episodes=num_training_episodes, target_update=10):
        """
        Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """
        pass

    def test(self, num_test_episodes=100):
        """
          Test your agent locally before submission to get a hint of the expected score.

          Args:
              num_episodes (int): Number of episodes to test for.
          """
        pass

    def plot_training_progress(self, window_size=100):
        pass

    def animate_episode(self):
        """
        Run a single episode with rendering to visualize the agent's performance.
        """
        pass


if __name__ == "__main__":
    # Initialization of agent and environment
    env_name = "LunarLander-v3"
    env = gym.make(env_name)
