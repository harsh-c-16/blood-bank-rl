"""Run one episode of the random and mean-demand policies in both modes and print the metrics.

Usage: python -m scripts.sanity_check
"""
from baselines import MeanDemandPolicy, RandomPolicy
from env import make_env
from env.rollout import run_episode

METRICS = ["total_reward", "demand", "shortage", "substituted", "expired", "avg_stock",
           "service_level", "expiry_rate"]


def main():
    print(f"{'mode':<7}{'policy':<12}" + "".join(f"{m:>14}" for m in METRICS))
    for mode in ["multi", "single"]:
        for name, policy in [("random", RandomPolicy()), ("mean", MeanDemandPolicy())]:
            out = run_episode(make_env(mode), policy, seed=0)
            row = "".join(f"{out[m]:>14.3f}" if isinstance(out[m], float) else f"{out[m]:>14}"
                          for m in METRICS)
            print(f"{mode:<7}{name:<12}{row}")


if __name__ == "__main__":
    main()
