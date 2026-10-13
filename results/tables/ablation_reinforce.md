| Agent | Seeds | Env steps | Final return [95% CI] | IQM over seeds | Seed range | Success [95% CI] | Reached 200 | Best checkpoint return | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| REINFORCE (baseline=none, entropy_coef=0) | 10 | 1,000k | 133.5 [89.8, 175.5] | 131.5 | -12.1 to 239.7 | 32% [14%, 53%] | 5/10 runs | 175.2 | 6 min |
| REINFORCE (baseline=none, entropy_coef=0.01) | 10 | 1,000k | 41.8 [4.2, 79.4] | 37.0 | -58.0 to 141.2 | 9% [3%, 16%] | 8/10 runs, median 810k | 196.4 | 6 min |
| REINFORCE (baseline=normalize, entropy_coef=0) | 10 | 1,000k | 122.3 [73.5, 171.0] | 121.3 | -30.4 to 246.5 | 29% [11%, 52%] | 9/10 runs, median 655k | 198.9 | 5 min |
| REINFORCE (baseline=normalize, entropy_coef=0.01) | 10 | 1,000k | 178.9 [141.7, 210.3] | 195.4 | 73.0 to 234.8 | 62% [44%, 76%] | 10/10 runs, median 385k | 232.2 | 5 min |
| REINFORCE (baseline=value, entropy_coef=0) | 10 | 1,000k | 113.0 [66.6, 162.2] | 109.5 | 2.1 to 263.6 | 23% [6%, 45%] | 5/10 runs | 186.1 | 6 min |
| REINFORCE (baseline=value, entropy_coef=0.01) | 10 | 1,000k | 131.6 [88.1, 173.7] | 135.7 | -18.9 to 274.6 | 24% [10%, 44%] | 6/10 runs, median 490k | 170.4 | 6 min |
