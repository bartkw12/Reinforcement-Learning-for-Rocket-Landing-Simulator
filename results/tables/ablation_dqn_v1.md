| Agent | Seeds | Env steps | Final return [95% CI] | IQM over seeds | Seed range | Success [95% CI] | Reached 200 | Best checkpoint return | Training time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dqn_v1 (gamma=0.99, loss=huber) | 5 | 424k | 142.2 [91.5, 180.2] | 154.7 | 49.1 to 197.6 | 45% [28%, 57%] | 2/5 runs | 180.5 | 13 min |
| dqn_v1 (gamma=0.99, loss=mse) | 5 | 405k | 144.8 [104.5, 191.4] | 132.1 | 92.5 to 235.5 | 56% [46%, 71%] | 1/5 runs | 150.1 | 11 min |
| dqn_v1 (gamma=0.9, loss=huber) | 5 | 729k | -114.1 [-187.7, -64.5] | -89.6 | -254.4 to -47.1 | 2% [0%, 5%] | 0/5 runs | -74.3 | 21 min |
| dqn_v1 (gamma=0.9, loss=mse) | 5 | 658k | -128.6 [-163.1, -95.6] | -127.1 | -189.8 to -71.9 | 0% [0%, 0%] | 0/5 runs | -82.0 | 15 min |
