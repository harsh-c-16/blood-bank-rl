# Evaluation Summary

## Multi mode (8 blood groups, with substitution)

| Policy | Total Reward | Shortage Rate | Expiry Rate |
|--------|-------------|---------------|-------------|
| Random | -1275.5 ± 18.7 | 0.0000 ± 0.0001 | 0.7261 ± 0.0054 |
| MeanDemand | -172.6 ± 8.7 | 0.0524 ± 0.0103 | 0.0373 ± 0.0090 |
| FixedOrder | -163.6 ± 7.0 | 0.0341 ± 0.0081 | 0.0481 ± 0.0088 |
| YesterdayDemand | -167.5 ± 4.4 | 0.0425 ± 0.0033 | 0.0437 ± 0.0035 |
| PPO | -151.4 ± 4.8 | 0.0288 ± 0.0043 | 0.0021 ± 0.0007 |


## Single mode (O+ only, no substitution)

| Policy | Total Reward | Shortage Rate | Expiry Rate |
|--------|-------------|---------------|-------------|
| Random | -102.7 ± 6.9 | 0.0055 ± 0.0048 | 0.3033 ± 0.0289 |
| MeanDemand | -51.3 ± 4.1 | 0.0348 ± 0.0141 | 0.0332 ± 0.0115 |
| FixedOrder | -51.1 ± 3.1 | 0.0214 ± 0.0114 | 0.0498 ± 0.0133 |
| YesterdayDemand | -44.0 ± 2.4 | 0.0171 ± 0.0048 | 0.0180 ± 0.0044 |
| DQN | -42.7 ± 1.7 | 0.0105 ± 0.0043 | 0.0197 ± 0.0072 |
| PPO | -41.5 ± 1.8 | 0.0097 ± 0.0049 | 0.0120 ± 0.0055 |
