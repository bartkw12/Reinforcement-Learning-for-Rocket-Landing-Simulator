# Reinforcement-Learning-for-Rocket-Landing-Simulator

The goal of this project is to develop, train, and compare three reinforcement learning
agents to solve the Lunar Lander problem; a game where you must land a spacecraft safely
between two flags in a “windy” environment. The environment assigns rewards based on
the lander's actions and state, encouraging successful landings, and penalizing crashes, or
taking too long. The reinforcement learning algorithms used will be Q-learning, Deep Q-
learning (DQN), and a policy gradient method, each aimed at training an agent to maximize
reward.

Each agent will be trained to attain a satisfactory performance based on performance
metrics such as cumulative reward. The agents' learning curves with varying configurations
will be compared, and insights into their strengths and weaknesses will be analyzed to
further cement our reinforcement learning understanding.

### Important Links

Link to find the LunarLander-Gymnasium Python library:
https://gymnasium.farama.org/environments/box2d/lunar_lander/

*Note: *there is a very common problem with Box2D environments (such as LunarLander) in Gymnasium on Windows typically involving missing C++ build tools and dependencies. 
I did not run into any issues using Linux, but if you do use Windows here is a great video that solves the problem:*
https://www.youtube.com/watch?v=gMgj4pSHLww

## Model Summary

### Q-Learning

Q-learning is a reinforcement learning algorithm that trains an agent to assign values to its possible actions based on its current state, without requiring a model of the environment (model-free).

### Deep Q-Learning Network

Deep Q-Network (DQN) is defined as a model that combines Q-learning with a deep CNN to train a network to approximate the value of the Q function, which maps state-action pairs to their expected discounted return.

### Policy Gradient

The third reinforcement learning algorithm implemented and used in the Lunar Lander
environment was REINFORCE, a policy gradient method. It uses Monte Carlo methods to
estimate the necessary policy gradients and returns. Unlike Q-learning or DQN, the agent
directly samples all actions from the Lunar Lander environment. The previous two
methods determine their actions based on value function estimating a Q-value.