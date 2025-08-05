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
    def __init__(self, env, hidden_dim=hidden_size, gamma=gamma, lr=alpha):

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

        # Training method tracking variables
        self.scores = []
        self.scores_window = deque(maxlen=100)
        self.best_avg_reward = -np.inf
        self.best_model_path = "reinforce_best_model.pth"

    def get_action(self, state, training=True):
        """
        Samples an action from the categorical distribution defined by the network's logits. Returns the sampled action
        and the log probability of the action. Log probability is used during the policy update (REINFORCE).

        During testing - select the action with the highest probability (greedy action).
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state)
        dist = distributions.Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action)

    def compute_returns(self, rewards, standardize=True):
        """
        Calculate the discounted returns from the rewards collected during an episode.
        """
        returns = []
        discounted_sum = 0

        # Method reverses the rewards
        for r in reversed(rewards):
            # calculate the cumulative discounted sum
            discounted_sum = r + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # normalize the returns
        returns = torch.tensor(returns)
        if standardize: # training stability
            returns = (returns - returns.mean()) / (returns.std() + self.eps)

        return returns

    def update_policy(self, log_probs, returns):
        """
        Use the log probabilities and the returns (discounted future rewards) to calculate the policy loss (the
        negative log-likelihood weighted by the returns).
        """
        policy_loss = []
        for log_prob, Gt in zip(log_probs, returns):
            policy_loss.append(-log_prob * Gt)

        # clear old gradients
        self.optimizer.zero_grad()

        # loss calc
        loss = torch.stack(policy_loss).sum()

        # Backpropagate - computes gradients of the loss w.r.t. the Q-network’s parameters
        loss.backward()

        # Step the optimizer to update the weights
        self.optimizer.step()

    def train(self, num_episodes=num_training_episodes):
        """
            Contains the main training loop where the agent learns over multiple episodes.
        Args:
            num_episodes (int): Number of episodes to train for.
        """
        self.scores = []
        self.scores_window = deque(maxlen=100)

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            log_probs = []
            rewards = []
            score = 0
            done = False

            while not done:
                action, log_prob = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
                score += reward

            # Compute returns and update policy
            returns = self.compute_returns(rewards)
            self.update_policy(log_probs, returns)

            # Save scores and update window
            self.scores_window.append(score)
            self.scores.append(score)
            avg_score = np.mean(self.scores_window)

            # Print progress
            print(f"\rEpisode {episode}\tAverage Score: {avg_score:.2f}", end="")
            if episode % 100 == 0:
                print(f"\rEpisode {episode}\tAverage Score: {avg_score:.2f}")
                self.plot_training_progress()
                print(f"Current Score: {score:.2f}")

                # Autosave best model
                if avg_score > self.best_avg_reward:
                    self.best_avg_reward = avg_scoreself.best_avg_reward = avg_score
                    torch.save(self.policy.state_dict(), self.best_model_path)
                    print(f"New best model saved with avg reward: {self.best_avg_reward:.2f}")

            # Early stopping if solved
            if episode >= 100 and avg_score >= 200:
                print(f"\nEnvironment solved in {episode} episodes! Average Score: {avg_score:.2f}")
                torch.save(self.policy.state_dict(), self.best_model_path)
                break





