| Animation | Method | Run | Episode seed | Outcome | Return | Chosen as |
| :--- | :--- | :--- | ---: | :--- | ---: | :--- |
| landing_grid, sb3_ppo | SB3 PPO | sb3_ppo/seed_107 | 10000 | landed | 252.8 | median run, first test episode |
| landing_grid, dqn | DQN | dqn/seed_100 | 10000 | time limit | 135.4 | median run, first test episode |
| landing_grid, sb3_dqn | SB3 DQN | sb3_dqn/seed_100 | 10000 | landed | 258.2 | median run, first test episode |
| landing_grid, q_learning | Q-learning | q_learning/seed_109 | 10000 | landed | 255.0 | median run, first test episode |
| landing_grid, reinforce | REINFORCE | reinforce/seed_103 | 10000 | landed | 270.3 | median run, first test episode |
| reinforce_failure | REINFORCE | reinforce/seed_103 | 10005 | out of bounds | -167.1 | median run, first test episode it failed |
| wind_grid, sb3_ppo | SB3 PPO | sb3_ppo/seed_107 | 10000 | crashed | 10.3 | median run, first test episode, wind |
| wind_grid, dqn | DQN | dqn/seed_100 | 10000 | landed | 279.4 | median run, first test episode, wind |
| wind_grid, sb3_dqn | SB3 DQN | sb3_dqn/seed_100 | 10000 | crashed | 4.3 | median run, first test episode, wind |
| wind_grid, q_learning | Q-learning | q_learning/seed_109 | 10000 | landed | 264.2 | median run, first test episode, wind |
| wind_grid, reinforce | REINFORCE | reinforce/seed_103 | 10000 | landed | 194.3 | median run, first test episode, wind |
