# Model improvement experiments

All policy comparisons below use the same 30 evaluation seeds (1000–1029) and the same environment and reward. Reward is better when it is less negative. Confidence intervals are 95% normal intervals for paired per-seed differences.

## Multi-group PPO

| Checkpoint | Training setup | Reward | Shortage rate | Expiry rate |
|---|---|---:|---:|---:|
| Normalized retrain | 1,015,808 steps, [256, 256], learning rate 1e-4, entropy 0.005, VecNormalize | -473.5 ± 17.7 | 0.4110 ± 0.0121 | 0.0086 ± 0.0022 |
| Original checkpoint | 507,904 steps, [128, 128], learning rate 3e-4, entropy 0.01 | -234.8 ± 7.2 | 0.1306 ± 0.0060 | 0.0013 ± 0.0007 |
| Updated checkpoint | 1,007,616 steps, [128, 128], learning rate 3e-4, entropy 0.01, no observation normalization | **-151.4 ± 4.8** | **0.0288 ± 0.0043** | **0.0021 ± 0.0007** |
| Best baseline (FixedOrder) | Orders the average daily demand | -163.6 ± 7.0 | 0.0341 ± 0.0081 | 0.0481 ± 0.0088 |

The updated PPO gains 322.1 reward points versus the normalized retrain (95% paired CI: +316.2 to +328.0) and 83.4 versus the original checkpoint (CI: +80.5 to +86.2). Against the best baseline (FixedOrder), reward is 12.19 points higher (CI: +9.25 to +15.14) and PPO is better on 26 of 30 seeds.

## Single-group PPO and DQN

| Policy | Training steps | Reward | Shortage rate | Expiry rate |
|---|---:|---:|---:|---:|
| PPO initial | 204,800 | -41.9 ± 1.8 | 0.0124 ± 0.0043 | 0.0109 ± 0.0044 |
| PPO updated | 507,904 | **-41.5 ± 1.8** | **0.0097 ± 0.0049** | 0.0120 ± 0.0055 |
| DQN initial | 200,000 | -43.3 ± 2.1 | 0.0131 ± 0.0057 | **0.0178 ± 0.0071** |
| DQN updated | 500,000 | -42.7 ± 1.7 | **0.0105 ± 0.0043** | 0.0197 ± 0.0072 |

Longer training lowered shortage for both algorithms. Reward changes were small and their paired confidence intervals include zero (+0.43 for PPO, CI -0.31 to +1.16; +0.53 for DQN, CI -0.21 to +1.27). DQN's expiry rate also increased. These runs do not establish a clear reward improvement over the original checkpoints; the PPO model offers the best single-group reward mean, while the best baseline (YesterdayDemand) has a mean of -44.0, 2.5 points lower than PPO (CI: +1.8 to +3.3).

## Diagnosis and next step

The largest failure was the normalized multi-group PPO retrain: its action policy was overly conservative, producing a 41.1% shortage rate. Reusing the original observation scale and [128, 128] architecture, then training to about 1M steps, brought reward above every baseline and cut expiry sharply. Further work should tune multi-group PPO on the 500–519 tuning seeds and select the checkpoint by held-out reward and shortage, keeping evaluation seeds 1000–1029 untouched.
