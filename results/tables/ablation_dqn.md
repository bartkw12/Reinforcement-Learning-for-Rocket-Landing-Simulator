| Agent | Seeds | Env steps | Final return [95% CI] | IQM over seeds | Seed range | Success [95% CI] | Reached 200 | Best checkpoint return | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DQN (double=False, loss=huber) | 5 | 1,000k | 232.7 [209.8, 256.8] | 233.9 | 189.0 to 272.7 | 78% [68%, 88%] | 5/5 runs, median 50k | 275.8 | 70 min |
| DQN (double=True, loss=huber) | 5 | 1,000k | 257.3 [253.1, 264.3] | 254.3 | 252.3 to 271.2 | 88% [85%, 92%] | 5/5 runs, median 90k | 248.2 | 74 min |
| DQN (double=False, loss=mse) | 5 | 1,000k | -263.3 [-636.4, 109.7] | -228.8 | -899.6 to 269.2 | 33% [0%, 70%] | 5/5 runs, median 110k | 269.8 | 68 min |
| DQN (double=True, loss=mse) | 5 | 1,000k | -289.1 [-594.0, -7.7] | -249.6 | -829.8 to 133.0 | 14% [0%, 37%] | 5/5 runs, median 80k | 263.1 | 73 min |
