"""Grid search of the base-stock level k on the tuning seeds, for both modes.

Usage: python -m baselines.tune_base_stock
Writes results/base_stock_sweep.csv and results/best_k.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.policies import BaseStockPolicy
from env import make_env
from env import config as C
from env.rollout import run_episode

K_GRID = np.round(np.arange(1.0, 5.0 + 1e-9, 0.25), 2)
COLUMNS = ["total_reward", "shortage_rate", "expiry_rate", "avg_stock"]
RESULTS = Path("results")


def sweep(mode: str) -> pd.DataFrame:
    env = make_env(mode)
    rows = []
    for k in K_GRID:
        runs = [run_episode(env, BaseStockPolicy(k), seed) for seed in C.TUNE_SEEDS]
        row = {"mode": mode, "k": float(k)}
        row.update({c: float(np.mean([r[c] for r in runs])) for c in COLUMNS})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    table = pd.concat([sweep("multi"), sweep("single")], ignore_index=True)
    table.to_csv(RESULTS / "base_stock_sweep.csv", index=False)

    best = {}
    for mode, part in table.groupby("mode"):
        best[mode] = float(part.loc[part["total_reward"].idxmax(), "k"])
    with open(RESULTS / "best_k.json", "w") as f:
        json.dump(best, f, indent=2)

    print(table.to_string(index=False, float_format="%.3f"))
    print("\nbest k:", best)


if __name__ == "__main__":
    main()
