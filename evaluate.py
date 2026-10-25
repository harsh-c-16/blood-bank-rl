"""Evaluate all policies (baselines + RL) on EVAL_SEEDS and produce result tables.

Usage:
    python -m evaluate
"""
import json
import os
import warnings

import pandas as pd
from stable_baselines3 import PPO, DQN

from agents.sb3_policy import SB3Policy
from baselines.policies import RandomPolicy, MeanDemandPolicy, BaseStockPolicy
from env import make_env
from env.config import EVAL_SEEDS, SINGLE_GROUP_DEFAULT
from env.rollout import run_episode

METRICS = ["total_reward", "shortage_rate", "expiry_rate", "avg_stock", "service_level"]
SCALAR_KEYS = ["total_reward", "demand", "issued", "substituted", "shortage",
               "expired", "arrived", "ordered", "avg_stock",
               "shortage_rate", "expiry_rate", "service_level"]


def _load_best_k() -> dict:
    path = os.path.join("results", "best_k.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    warnings.warn("results/best_k.json not found, using fallback k=2.5")
    return {"multi": 2.5, "single": 2.5}


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
    lines = ["| Policy | Total Reward | Shortage Rate | Expiry Rate | Avg Stock | Service Level |",
             "|--------|-------------|---------------|-------------|-----------|---------------|"]
    for policy in df["policy"].unique():
        sub = df[df["policy"] == policy]
        vals = []
        for m in METRICS:
            mean = sub[m].mean()
            std = sub[m].std()
            if m in ("shortage_rate", "expiry_rate", "service_level"):
                vals.append(f"{mean:.4f} ± {std:.4f}")
            else:
                vals.append(f"{mean:.1f} ± {std:.1f}")
        lines.append(f"| {policy} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _save_traces(mode: str, policies: dict, seed: int = 1000):
    """Save daily traces and per-group data for the trace seed."""
    trace_data = {}
    for name in ["PPO", "BaseStock"]:
        policy = policies.get(name)
        if policy is None:
            continue
        env = make_env(mode, SINGLE_GROUP_DEFAULT)
        result = run_episode(env, policy, seed)
        trace_data[name] = {
            "daily": result["daily"],
            "per_group": result["per_group"],
            "groups": result["groups"],
            "total_reward": result["total_reward"],
            "shortage_rate": result["shortage_rate"],
            "expiry_rate": result["expiry_rate"],
        }
        env.close()
    path = os.path.join("results", f"trace_{mode}.json")
    with open(path, "w") as f:
        json.dump(trace_data, f, indent=2)
    print(f"  Saved traces to {path}")


def main():
    os.makedirs("results", exist_ok=True)
    best_k = _load_best_k()

    # --- Multi mode ---
    print("\n=== Multi mode ===")
    k_multi = best_k.get("multi", 2.5)
    ppo_multi_model = _load_model(PPO, os.path.join("models", "ppo_multi.zip"))

    multi_policies = {
        "Random": RandomPolicy(),
        "MeanDemand": MeanDemandPolicy(),
        "BaseStock": BaseStockPolicy(k_multi),
        "PPO": SB3Policy(ppo_multi_model) if ppo_multi_model else None,
    }

    df_multi = _evaluate_mode("multi", multi_policies, EVAL_SEEDS)
    df_multi.to_csv(os.path.join("results", "eval_multi.csv"), index=False)

    _save_traces("multi", multi_policies)

    # --- Single mode ---
    print("\n=== Single mode ===")
    k_single = best_k.get("single", 2.5)
    dqn_model = _load_model(DQN, os.path.join("models", "dqn_single.zip"))
    ppo_single_model = _load_model(PPO, os.path.join("models", "ppo_single.zip"))

    single_policies = {
        "Random": RandomPolicy(),
        "MeanDemand": MeanDemandPolicy(),
        "BaseStock": BaseStockPolicy(k_single),
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
