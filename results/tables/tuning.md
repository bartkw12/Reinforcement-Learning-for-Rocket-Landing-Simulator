#### DQN

| Candidate | Validation return [95% CI] | Seed range | Success | Best checkpoint |
| :--- | ---: | ---: | ---: | ---: |
| **lr=0.0003** (selected) | 237.0 [191.7, 280.7] | 191.7 to 280.7 | 70% | 281.3 |
| target_update_interval=1000 | 223.5 [156.6, 259.8] | 156.6 to 259.8 | 73% | 269.0 |
| zoo | -354.3 [-654.6, 189.1] | -654.6 to 189.1 | 24% | 254.3 |
| buffer_size=200000 | -959.9 [-3220.0, 207.0] | -3220.0 to 207.0 | 44% | 267.5 |

#### Q-learning

| Candidate | Validation return [95% CI] | Seed range | Success | Best checkpoint |
| :--- | ---: | ---: | ---: | ---: |
| **alpha=0.3, tiles_per_dim=6, decay_steps=100000** (selected) | 216.8 [201.0, 240.6] | 201.0 to 240.6 | 78% | 232.7 |
| alpha=0.3, tiles_per_dim=4, decay_steps=100000 | 208.2 [179.7, 239.0] | 179.7 to 239.0 | 75% | 211.8 |
| alpha=0.3, tiles_per_dim=4, decay_steps=300000 | 207.7 [198.0, 223.4] | 198.0 to 223.4 | 77% | 195.3 |
| alpha=0.1, tiles_per_dim=3, decay_steps=100000 | 190.6 [179.7, 197.0] | 179.7 to 197.0 | 64% | 187.6 |
| alpha=0.3, tiles_per_dim=3, decay_steps=300000 | 189.3 [184.3, 192.7] | 184.3 to 192.7 | 62% | 193.5 |
| alpha=0.3, tiles_per_dim=6, decay_steps=300000 | 185.1 [143.7, 206.9] | 143.7 to 206.9 | 68% | 208.3 |
| alpha=0.1, tiles_per_dim=4, decay_steps=100000 | 167.7 [157.0, 178.0] | 157.0 to 178.0 | 59% | 189.8 |
| alpha=0.1, tiles_per_dim=4, decay_steps=300000 | 165.9 [135.9, 181.3] | 135.9 to 181.3 | 57% | 169.5 |
| alpha=0.3, tiles_per_dim=3, decay_steps=100000 | 149.6 [137.4, 164.4] | 137.4 to 164.4 | 39% | 201.4 |
| alpha=0.1, tiles_per_dim=3, decay_steps=300000 | 133.8 [31.8, 199.0] | 31.8 to 199.0 | 53% | 171.2 |
| alpha=0.1, tiles_per_dim=6, decay_steps=300000 | 99.0 [72.8, 131.6] | 72.8 to 131.6 | 48% | 109.3 |
| alpha=0.1, tiles_per_dim=6, decay_steps=100000 | 72.9 [21.1, 117.1] | 21.1 to 117.1 | 6% | 35.6 |

#### REINFORCE

| Candidate | Validation return [95% CI] | Seed range | Success | Best checkpoint |
| :--- | ---: | ---: | ---: | ---: |
| **lr=0.003, baseline=normalize, entropy_coef=0.01** (selected) | 203.9 [157.1, 237.6] | 157.1 to 237.6 | 73% | 220.2 |
| lr=0.003, baseline=value, entropy_coef=0 | 188.1 [148.1, 260.7] | 148.1 to 260.7 | 52% | 210.8 |
| lr=0.001, baseline=normalize, entropy_coef=0.01 | 137.7 [32.6, 205.7] | 32.6 to 205.7 | 32% | 176.8 |
| lr=0.003, baseline=normalize, entropy_coef=0 | 127.0 [92.4, 146.2] | 92.4 to 146.2 | 7% | 178.1 |
| lr=0.003, baseline=value, entropy_coef=0.01 | 114.1 [109.6, 118.1] | 109.6 to 118.1 | 8% | 152.3 |
| lr=0.001, baseline=value, entropy_coef=0.01 | 40.6 [-26.6, 157.0] | -26.6 to 157.0 | 10% | 74.0 |
| lr=0.001, baseline=normalize, entropy_coef=0 | 4.9 [-35.5, 66.9] | -35.5 to 66.9 | 0% | 115.5 |
| lr=0.001, baseline=value, entropy_coef=0 | -23.4 [-35.9, -3.6] | -35.9 to -3.6 | 0% | 43.3 |
| lr=0.0003, baseline=value, entropy_coef=0 | -100.9 [-110.6, -94.9] | -110.6 to -94.9 | 0% | -74.8 |
| lr=0.0003, baseline=normalize, entropy_coef=0.01 | -130.9 [-153.1, -119.7] | -153.1 to -119.7 | 0% | -115.5 |
| lr=0.0003, baseline=value, entropy_coef=0.01 | -133.5 [-163.8, -94.2] | -163.8 to -94.2 | 0% | -46.2 |
| lr=0.0003, baseline=normalize, entropy_coef=0 | -139.5 [-146.1, -130.3] | -146.1 to -130.3 | 0% | -111.0 |
