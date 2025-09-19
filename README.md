# Reinforcement-Learning-for-Rocket-Landing-Simulator

Three implementations of RL agents (Q-learning, DQN, and PG) to solve the LunarLander rocket trajectory optimization problem from the Gymnasium Python library.

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