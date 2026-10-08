| Agent | Seeds | Env steps | Final return [95% CI] | IQM over seeds | Seed range | Success [95% CI] | Reached 200 | Best checkpoint return | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SB3 PPO | 10 | 1,000k | 258.8 [251.6, 265.0] | 260.3 | 233.2 to 274.4 | 99% [98%, 100%] | 10/10 runs, median 505k | 258.9 | 8 min |
| DQN | 10 | 1,000k | 234.1 [213.2, 254.5] | 235.1 | 185.9 to 282.8 | 79% [70%, 87%] | 10/10 runs, median 95k | 272.8 | 62 min |
| SB3 DQN | 10 | 1,000k | 183.2 [65.9, 253.6] | 237.9 | -303.6 to 276.2 | 76% [56%, 91%] | 10/10 runs, median 70k | 266.7 | 70 min |
| Q-learning | 10 | 1,000k | 211.5 [203.0, 218.6] | 213.4 | 182.0 to 226.9 | 76% [72%, 80%] | 10/10 runs, median 515k | 218.9 | 4 min |
| REINFORCE | 10 | 1,000k | 178.9 [141.7, 210.3] | 195.4 | 73.0 to 234.8 | 62% [44%, 76%] | 10/10 runs, median 385k | 232.2 | 5 min |
