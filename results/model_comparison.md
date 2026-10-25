# Model improvement experiments

All policy comparisons below use the same 30 evaluation seeds (1000–1029) and the same environment and reward. Reward is better when it is less negative. Confidence intervals are 95% normal intervals for paired per-seed differences.

## Multi-group PPO

| Checkpoint | Training setup | Reward | Shortage rate | Expiry rate |
|---|---|---:|---:|---:|
| Normalized retrain | 1,015,808 steps, [256, 256], learning rate 1e-4, entropy 0.005, VecNormalize | -473.5 ± 17.7 | 0.4110 ± 0.0121 | 0.0086 ± 0.0022 |
| Original checkpoint | 507,904 steps, [128, 128], learning rate 3e-4, entropy 0.01 | -234.8 ± 7.2 | 0.1306 ± 0.0060 | 0.0013 ± 0.0007 |
| Updated checkpoint | 1,007,616 steps, [128, 128], learning rate 3e-4, entropy 0.01, no observation normalization | **-151.4 ± 4.8** | **0.0288 ± 0.0043** | **0.0021 ± 0.0007** |
| Base-stock baseline | Tuned k=3.5 | -147.9 ± 4.1 | 0.0274 ± 0.0036 | 0.0281 ± 0.0029 |

The updated PPO gains 322.1 reward points versus the normalized retrain (95% paired CI: +316.2 to +328.0) and 83.4 versus the original checkpoint (CI: +80.5 to +86.2). Against base-stock, reward is 3.53 points lower (CI: -5.53 to -1.53), while expiry is 2.60 percentage points lower (CI: -2.71 to -2.49). The mean shortage difference is only +0.14 percentage points and its CI includes zero. PPO is therefore much improved and has a strong expiry trade-off, but base-stock still has the better mean reward.

## Single-group PPO and DQN

| Policy | Training steps | Reward | Shortage rate | Expiry rate |
|---|---:|---:|---:|---:|
| PPO initial | 204,800 | -41.9 ± 1.8 | 0.0124 ± 0.0043 | 0.0109 ± 0.0044 |
| PPO updated | 507,904 | **-41.5 ± 1.8** | **0.0097 ± 0.0049** | 0.0120 ± 0.0055 |
| DQN initial | 200,000 | -43.3 ± 2.1 | 0.0131 ± 0.0057 | **0.0178 ± 0.0071** |
| DQN updated | 500,000 | -42.7 ± 1.7 | **0.0105 ± 0.0043** | 0.0197 ± 0.0072 |

Longer training lowered shortage for both algorithms. Reward changes were small and their paired confidence intervals include zero (+0.43 for PPO, CI -0.31 to +1.16; +0.53 for DQN, CI -0.21 to +1.27). DQN's expiry rate also increased. These runs do not establish a clear reward improvement over the original checkpoints; the PPO model offers the best single-group reward mean, while the tuned base-stock mean is -42.0.

## Diagnosis and next step

The largest failure was the normalized multi-group PPO retrain: its action policy was overly conservative, producing a 41.1% shortage rate. Reusing the original observation scale and [128, 128] architecture, then training to about 1M steps, brought shortage close to base-stock and cut expiry sharply. Further work should tune multi-group PPO on the 500–519 tuning seeds and select the checkpoint by held-out reward and shortage, keeping evaluation seeds 1000–1029 untouched.
