"""Evaluate all policies (baselines + RL) on EVAL_SEEDS and produce result tables.

Usage:
    python -m evaluate
"""
import os
import warnings

import pandas as pd
from stable_baselines3 import PPO, DQN

from agents.sb3_policy import SB3Policy
from baselines.policies import FixedOrderPolicy, MeanDemandPolicy, RandomPolicy, YesterdayDemandPolicy
from env import make_env
from env.config import EVAL_SEEDS, SINGLE_GROUP_DEFAULT
from env.rollout import run_episode

METRICS = ["total_reward", "shortage_rate", "expiry_rate"]
SCALAR_KEYS = ["total_reward", "demand", "issued", "substituted", "shortage",
               "expired", "arrived", "ordered", "shortage_rate", "expiry_rate"]


def _load_model(cls, path: str):
    """Load an SB3 model, returning None (with a warning) if the file is missing."""
    if not os.path.exists(path):
        warnings.warn(f"Model {path} not found, skipping")
        return None
    return cls.load(path)


def _evaluate_mode(mode: str, policies: dict, seeds: list) -> pd.DataFrame:
    """Run all policies over all seeds, return a DataFrame of scalar results."""
    rows = []
    for name, policy in policies.items():
        if policy is None:
            continue
        print(f"  Evaluating {name} ...", end="", flush=True)
        for seed in seeds:
            env = make_env(mode, SINGLE_GROUP_DEFAULT)
            result = run_episode(env, policy, seed)
            row = {"policy": name, "seed": seed}
            row.update({k: result[k] for k in SCALAR_KEYS})
            rows.append(row)
            env.close()
        print(" done")
    return pd.DataFrame(rows)


def _summary_table(df: pd.DataFrame) -> str:
    """Produce a markdown table of mean +/- std for each policy."""
    lines = ["| Policy | Total Reward | Shortage Rate | Expiry Rate |",
             "|--------|-------------|---------------|-------------|"]
    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        vals = []
        for m in METRICS:
            mean = sub[m].mean()
            std = sub[m].std()
            if m != "total_reward":
                vals.append(f"{mean:.4f} ± {std:.4f}")
            else:
                vals.append(f"{mean:.1f} ± {std:.1f}")
        lines.append(f"| {policy} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    os.makedirs("results", exist_ok=True)

    # --- Multi mode ---
    print("\n=== Multi mode ===")
    ppo_multi_model = _load_model(PPO, os.path.join("models", "ppo_multi.zip"))

    multi_policies = {
        "Random": RandomPolicy(),
        "MeanDemand": MeanDemandPolicy(),
        "FixedOrder": FixedOrderPolicy(),
        "YesterdayDemand": YesterdayDemandPolicy(),
        "PPO": SB3Policy(ppo_multi_model) if ppo_multi_model else None,
    }

    df_multi = _evaluate_mode("multi", multi_policies, EVAL_SEEDS)
    df_multi.to_csv(os.path.join("results", "eval_multi.csv"), index=False)

    # --- Single mode ---
    print("\n=== Single mode ===")
    dqn_model = _load_model(DQN, os.path.join("models", "dqn_single.zip"))
    ppo_single_model = _load_model(PPO, os.path.join("models", "ppo_single.zip"))

    single_policies = {
        "Random": RandomPolicy(),
        "MeanDemand": MeanDemandPolicy(),
        "FixedOrder": FixedOrderPolicy(),
        "YesterdayDemand": YesterdayDemandPolicy(),
        "DQN": SB3Policy(dqn_model) if dqn_model else None,
        "PPO": SB3Policy(ppo_single_model) if ppo_single_model else None,
    }

    df_single = _evaluate_mode("single", single_policies, EVAL_SEEDS)
    df_single.to_csv(os.path.join("results", "eval_single.csv"), index=False)

    # --- Summary ---
    print("\n" + "=" * 60)
    summary_lines = ["# Evaluation Summary\n"]

    summary_lines.append("## Multi mode (8 blood groups, with substitution)\n")
    summary_lines.append(_summary_table(df_multi))
    summary_lines.append("")

    summary_lines.append(f"\n## Single mode ({SINGLE_GROUP_DEFAULT} only, no substitution)\n")
    summary_lines.append(_summary_table(df_single))
    summary_lines.append("")

    summary_text = "\n".join(summary_lines)
    print(summary_text)

    with open(os.path.join("results", "eval_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\nResults saved to results/eval_summary.md")


if __name__ == "__main__":
    main()
