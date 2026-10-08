| Agent | Seeds | Env steps | Final return [95% CI] | IQM over seeds | Seed range | Success [95% CI] | Reached 200 | Best checkpoint return | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN | 3 | 500k | 195.8 [177.4, 231.4] | 195.8 | 177.4 to 231.4 | 68% [64%, 75%] | 3/3 runs, median 200k | 235.2 | 19 min |
| SB3 DQN | 3 | 500k | 116.9 [-24.7, 226.3] | 116.9 | -24.7 to 226.3 | 30% [3%, 84%] | 3/3 runs, median 75k | 242.4 | 21 min |
| SB3 PPO | 3 | 500k | 139.3 [21.2, 246.7] | 139.3 | 21.2 to 246.7 | 39% [4%, 96%] | 1/3 runs | 139.3 | 2 min |
