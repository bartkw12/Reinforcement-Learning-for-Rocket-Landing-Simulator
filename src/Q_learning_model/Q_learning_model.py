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

    def select_action(self, state, testing=True):
        """
        Given a state, select an action to take. The function should operate in training and testing modes:
        - Testing: Uses greedy policy.
        - Training: Uses epsilon-greedy policy.

        Args:
            state (array): The current state of the environment.
            testing (bool): If True, uses greedy policy; if False, uses epsilon-greedy policy.

        Returns:
            int: The action to take.
        """
        # TODO: Implement your action selection policy here
        # For example, you might use an epsilon-greedy policy if you're using Q-learning
        # Ensure the action returned is an integer in the range [0, 3]

        # Discretize the state if you are going to use Q-Learning (before selecting action)
        state_features = self.state_discretizer.discretize(state)

        # Epsilon Greedy Policy Implementation
        if testing:
            # In testing, select the best action (purely greedy)
            action = np.argmax([np.sum(self.q_table[a][state_features]) for a in range(self.num_actions)])
        else:
            # In training, either explore or exploit
            if np.random.rand() < self.epsilon:
                # Exploration - choose a random action
                action = np.random.choice(self.num_actions)
            else:
                # Exploitation - choose the best action (based on max Q-value)
                action = np.argmax([np.sum(self.q_table[a][state_features]) for a in range(self.num_actions)])

        # action returns int in range 0-3
        return action

    def train(self, num_episodes):
        """
        Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """

        self.all_rewards = []  # Track rewards for ALL episodes
        episode_rewards = deque(maxlen=100)  # For recent 100-episode average

        # Initialize variables
        best_avg_reward = -float('inf')  # Store the best average reward

        # Loop through episodes
        for episode in range(num_episodes):
            state, _ = self.env.reset()  # Reset the environment for each new episode
            done = False
            total_reward = 0  # Track cumulative reward for the episode

            # Run the episode
            while not done:
                action = self.select_action(state)  # Select an action based on current policy
                next_state, reward, done, info, _ = self.env.step(action)  # Take the action and get feedback
                self.update(state, action, reward, next_state, done)  # Update the Q-table based on the experience
                state = next_state  # Move to the next state
                total_reward += reward  # Accumulate reward

            episode_rewards.append(total_reward)  # Add the total reward of the episode to the list
            self.all_rewards.append(total_reward)

            # Evaluate performance after every 100 episodes
            if len(episode_rewards) == 100:
                avg_reward = np.mean(episode_rewards)
                print(f"Episode {episode}: Average Reward over last 100 episodes = {avg_reward}")

                # Autosave the model if it performs better than before
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.save_agent("best_model.pkl")  # Save the best model
                    print(f"New best model saved with avg reward: {best_avg_reward}")

            # Plot training progress every 100 episodes
            if (episode + 1) % 100 == 0:
                self.plot_training_progress(window_size=100)

            # Decay epsilon (exploration rate) over time
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)  # Epsilon decay to 0.01

        return epsilon

    def update(self, state, action, reward, next_state, done):
        """
        Update your agent's knowledge based on the transition.

        Args:
            state (array): The previous state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (array): The new state after the action.
            done (bool): Whether the episode has ended.
        """
        # TODO: Implement your agent's update logic here
        # This method is where you would update your Q-table or neural network

        # Discretize the current and next states (convert cont. state to discrete features)
        state_features = self.state_discretizer.discretize(state)
        next_state_features = self.state_discretizer.discretize(next_state)

        # Get the current Q-value for the state-action pair
        current_q = np.sum([self.q_table[action][idx] for idx in state_features])

        # Find the maximum Q-value for the next state (over all possible actions)
        max_future_q = max(
            [np.sum([self.q_table[a][idx] for idx in next_state_features]) for a in range(self.num_actions)])

        # Compute the TD target - target value that the Q-value should move toward
        td_target = reward + self.gamma * max_future_q

        # Compute the TD error (difference between the target and the current Q-value)
        td_error = td_target - current_q