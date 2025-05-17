# Tabular Q-Learning

import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
from state_discretizer import StateDiscretizer
from collections import deque
from configuration import alpha, gamma, epsilon, epsilon_decay, num_training_episodes


class LunarLanderAgent:
    def __init__(self):

        # Initialize environment
        self.env = gym.make('LunarLander-v3')

        # Get number of actions from environment
        self.num_actions = self.env.action_space.n

        # Initialize state discretizer
        self.state_discretizer = StateDiscretizer(self.env)

        # initialize Q-table
        self.q_table = [np.zeros(self.state_discretizer.iht_size) for _ in range(self.num_actions)]

        # Set learning parameters
        self.alpha = alpha / self.state_discretizer.num_tilings  # Learning rate per tiling
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # Training performance tracking
        self.episode_rewards = []
        self.best_avg_reward = -float('inf')

    def select_action(self, state, testing=True):
        """
        Given a state, select an action to take. The function operates in training and testing modes:
        - Testing: Uses greedy policy.
        - Training: Uses epsilon-greedy policy.

        Args:
            state (array): The current state of the environment.
            testing (bool): If True, uses greedy policy; if False, uses epsilon-greedy policy.

        Returns:
            int: The action to take.
        """

        # Discretize the state for Q-Learning
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

        return action

    def train(self, num_episodes):
        """
        Contains the main training loop where the agent learns over multiple episodes.

        Args:
            num_episodes (int): Number of episodes to train for.
        """

        self.all_rewards = []                # Track rewards for ALL episodes
        episode_rewards = deque(maxlen=100)  # For recent 100-episode average

        # Initialize variables
        best_avg_reward = -float('inf')

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0

            # Run the episode
            while not done:
                action = self.select_action(state)  # Select an action based on current policy
                next_state, reward, done, info, _ = self.env.step(action)  # Take the action and get feedback
                self.update(state, action, reward, next_state, done)  # Update the Q-table based on the experience
                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
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

        # Compute the TD error
        td_error = td_target - current_q

        # Update the Q-values for the state-action pair using the TD error
        for idx in state_features:
            self.q_table[action][idx] += self.alpha * td_error

        return self.q_table[action]

    def test(self, num_episodes=100):
        """
        Test your agent locally before submission to get a hint of the expected score.

        Args:
            num_episodes (int): Number of episodes to test for.
        """

        cumulative_rewards = []
        reward_breakdowns = []
        success_count = 0  # track successful landings

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            episode_components = {
                "distance": 0, "velocity": 0, "angle": 0,
                "ground_contact": 0, "fuel": 0
            }

            while not done:
                action = self.select_action(state, testing=True)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Calculate reward components
                components = self._get_reward_breakdown(state, action, next_state)
                for key in episode_components:
                    episode_components[key] += components[key]

                total_reward += reward
                state = next_state

            cumulative_rewards.append(total_reward)
            reward_breakdowns.append(episode_components)

            # Check if the episode ended with a successful landing
            if terminated and total_reward >= 200:
                success_count += 1

        # Compute the average of the cumulative rewards from all test episodes
        average_reward = np.mean(cumulative_rewards)
        success_rate = success_count / num_episodes * 100

        print(f"Test Results (Over {num_episodes} Episodes):")
        print(f"  Average Reward: {average_reward}")
        print(f"  Success Rate: {success_rate:.2f}%")

        avg_breakdown = {key: np.mean([b[key] for b in reward_breakdowns])
                         for key in reward_breakdowns[0]}

        print("\n=== Reward Breakdown (Per Episode) ===")
        for key, value in avg_breakdown.items():
            print(f"{key:>15}: {value:7.2f}")

        return average_reward

    def save_agent(self, file_name):
        """
        Save your agent's model to a file.

        Args:
            file_name (str): The file name to save the model.
        """
        # TODO: Implement code to save your model (e.g., Q-table, neural network weights)
        # Example: for Q-learning:
        with open(file_name, 'wb') as f:
            # standard approach in ML to pickle data -
            # Serializes (converts to a byte stream) and deserializes (reconstructs) Python objects.
            pickle.dump({
                'q_table': self.q_table,
                'iht_dict': self.state_discretizer.iht.dictionary
            }, f)

    def load_agent(self, file_name):
        """
        Load your agent's model from a file.

        Args:
            file_name (str): The file name to load the model from.
        """
        # TODO: Implement code to load your model
        # Example: for Q-learning:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.state_discretizer.iht.dictionary = data['iht_dict']
        print(f"Model loaded from {file_name}.")

    def plot_training_progress(self, window_size=100):
        plt.figure(figsize=(10, 5))

        # Plot all rewards (optional)
        plt.plot(self.all_rewards, alpha=0.3, label='Episode Reward')

        # Calculate moving averages for the entire history
        moving_avg = [np.mean(self.all_rewards[max(0, i - window_size + 1):i + 1])
                      for i in range(len(self.all_rewards))]

        # Plot moving averages with markers
        plt.plot(moving_avg, 'g-', linewidth=2, marker='o', markersize=4,
                 label=f'Moving Average ({window_size} Episodes)')

        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('model_test_1.png')
        plt.close()  # Prevent memory leaks

    def _get_reward_breakdown(self, state, action, next_state):
        """
        Calculate and return individual reward components.
        Args:
            state (np.array): Current state (before action).
            action (int): Action taken.
            next_state (np.array): Next state (after action).
        Returns:
            dict: Reward components (distance, angle, velocity, ground contact, fuel).
        """
        # Landing pad coordinates (always at (0,0) in LunarLander)
        target_x, target_y = 0, 0

        # Extract state variables
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]
        left_leg, right_leg = state[6], state[7]

        # Reward components (approximated from Gymnasium's source)
        reward_components = {
            "distance": -np.sqrt((x - target_x) ** 2 + (y - target_y) ** 2),  # Penalize distance
            "velocity": -np.sqrt(vx ** 2 + vy ** 2),  # Penalize high speed
            "angle": -abs(angle),  # Penalize tilting
            "ground_contact": 10 * (left_leg + right_leg),  # Reward leg contact
            "fuel": -0.3 * (action in [1, 2, 3]),  # Penalize engine use
        }

        return reward_components

    def animate_episode(self):
        """
        Run a single episode with rendering to visualize the agent's performance.
        """
        # Create a new environment with rendering enabled
        render_env = gym.make('LunarLander-v3', render_mode='human')
        state, _ = render_env.reset()
        done = False
        total_reward = 0

        while not done:
            action = self.select_action(state, testing=True)
            next_state, reward, terminated, truncated, info = render_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = next_state

        render_env.close()

        # Success message
        print("\n")
        if terminated and total_reward >= 200:
            print("SUCCESS! Landed safely between the flags. 🚀")
        else:
            print("FAILED! Crashed or timed out. 💥")

        print(f"Animation Episode Reward: {total_reward}")


if __name__ == '__main__':

    agent = LunarLanderAgent()
    agent_model_file = 'model_test_1.pkl'  # Set the model file name

    # Example usage:
    # Uncomment the following lines to train your agent and save the model
    # agent.load_agent('model_test_1.pkl')  # Load the previously saved model

    # Load the previously saved best model (if available) to continue training from it
    try:
        agent.load_agent('best_model.pkl')  # Attempt to load the best model
        print("Loaded the best model for training...")
    except FileNotFoundError:
        print("No saved model found, starting training from scratch...")

    # -------------
    ''' 
    print("Training the agent...")
    agent.train(num_training_episodes)
    print("Training completed.")

    # Save the trained model
    #agent.save_agent(agent_model_file)
    #print("Model saved.")

    # Test the agent with greedy policy
    print("\nTesting the agent...")
    test_avg_reward = agent.test(num_episodes=100)
    #print(f"Average test reward: {test_avg_reward}")
    '''
    # Visualize the trained agent
    agent.animate_episode()