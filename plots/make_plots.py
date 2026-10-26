"""Generate all figures from evaluation and training results.

Usage:
    python -m plots.make_plots

Reads from results/ and writes to figures/.
"""
import json
import os
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG_DIR = "figures"
RES_DIR = "results"
DPI = 200
COL_WIDTH = 3.4  # inches, roughly one column of a two-column report


def _savefig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# --------------------------------------------------------------------------- #
# 1. Learning curves from Monitor CSVs
# --------------------------------------------------------------------------- #
def _read_monitor_csv(path: str) -> pd.DataFrame:
    """Read an SB3 Monitor CSV (header line starting with #, then CSV)."""
    with open(path) as f:
        # skip comment header line
        first = f.readline()
        if first.startswith("#"):
            pass  # skip
        else:
            f.seek(0)
        df = pd.read_csv(f)
    return df


def _load_monitor_data(log_dir: str) -> pd.DataFrame | None:
    """Load and concatenate all monitor CSVs in a directory."""
    files = sorted(glob.glob(os.path.join(log_dir, "*.monitor.csv")))
    if not files:
        files = sorted(glob.glob(os.path.join(log_dir, "monitor.csv")))
    if not files:
        return None
    dfs = []
    for f in files:
        try:
            df = _read_monitor_csv(f)
            dfs.append(df)
        except Exception:
            continue
    if not dfs:
        return None
    combined = pd.concat(dfs, ignore_index=True)
    # Monitor's elapsed-time column interleaves completed episodes from parallel workers.
    if "t" in combined.columns:
        combined = combined.sort_values("t", kind="stable").reset_index(drop=True)
    return combined


def plot_learning_curves(ref_multi: float | None = None,
                          ref_single: float | None = None):
    """Panel (a): PPO multi, Panel (b): DQN vs PPO single."""
    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH * 2 + 0.5, 2.8))

    # Panel (a): PPO multi
    ax = axes[0]
    data = _load_monitor_data(os.path.join(RES_DIR, "train_logs", "ppo_multi_final"))
    if data is not None and "r" in data.columns:
        rewards = data["r"].values
        smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()
        cumlen = data["l"].cumsum() if "l" in data.columns else np.arange(len(rewards))
        ax.plot(cumlen, smoothed, linewidth=0.8, color="#2196F3", label="PPO")
        if ref_multi is not None:
            ax.axhline(ref_multi, color="#FF5722", linestyle="--",
                       linewidth=0.8, label="MeanDemand")
        ax.legend(fontsize=6)
    ax.set_xlabel("Timesteps", fontsize=7)
    ax.set_ylabel("Episode Reward", fontsize=7)
    ax.set_title("(a) PPO — Multi (8 groups)", fontsize=8)
    ax.tick_params(labelsize=6)

    # Panel (b): DQN vs PPO single
    ax = axes[1]
    for tag, label, color in [("dqn_single_final", "DQN", "#4CAF50"),
                               ("ppo_single_final", "PPO", "#2196F3")]:
        data = _load_monitor_data(os.path.join(RES_DIR, "train_logs", tag))
        if data is not None and "r" in data.columns:
            rewards = data["r"].values
            smoothed = pd.Series(rewards).rolling(50, min_periods=1).mean()
            cumlen = data["l"].cumsum() if "l" in data.columns else np.arange(len(rewards))
            ax.plot(cumlen, smoothed, linewidth=0.8, color=color, label=label)
    if ref_single is not None:
        ax.axhline(ref_single, color="#FF5722", linestyle="--",
                   linewidth=0.8, label="MeanDemand")
    ax.legend(fontsize=6)
    ax.set_xlabel("Timesteps", fontsize=7)
    ax.set_ylabel("Episode Reward", fontsize=7)
    ax.set_title("(b) DQN vs PPO — Single (O+)", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.tight_layout()
    _savefig(fig, "learning_curves.png")


# --------------------------------------------------------------------------- #
# 2 & 3. Comparison bar charts
# --------------------------------------------------------------------------- #
def plot_comparison(csv_path: str, out_name: str, title_suffix: str):
    """Grouped bars: shortage_rate, expiry_rate, total_reward."""
    df = pd.read_csv(csv_path)
    policies = df["policy"].unique()
    n = len(policies)

    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH * 2 + 0.5, 2.8))

    # Panel 1: shortage & expiry rates
    ax = axes[0]
    x = np.arange(n)
    w = 0.35
    for i, (metric, label, color) in enumerate([
        ("shortage_rate", "Shortage Rate", "#E53935"),
        ("expiry_rate", "Expiry Rate", "#FB8C00"),
    ]):
        means = [df[df["policy"] == p][metric].mean() for p in policies]
        stds = [df[df["policy"] == p][metric].std() for p in policies]
        ax.bar(x + i * w - w / 2, means, w, yerr=stds, label=label,
               color=color, alpha=0.85, capsize=2, error_kw={"linewidth": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Rate", fontsize=7)
    ax.set_title(f"Shortage & Expiry — {title_suffix}", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=6)

    # Panel 2: total reward
    ax = axes[1]
    means = [df[df["policy"] == p]["total_reward"].mean() for p in policies]
    stds = [df[df["policy"] == p]["total_reward"].std() for p in policies]
    colors = ["#9E9E9E", "#2196F3", "#4CAF50", "#795548", "#FF9800", "#E91E63"][:n]
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=2,
           error_kw={"linewidth": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels(policies, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("Total Reward", fontsize=7)
    ax.set_title(f"Total Reward — {title_suffix}", fontsize=8)
    ax.tick_params(labelsize=6)

    fig.tight_layout()
    _savefig(fig, out_name)


# --------------------------------------------------------------------------- #
# 4. Daily trace plot
# --------------------------------------------------------------------------- #
def plot_trace(trace_path: str, out_name: str, days: int = 60):
    """60-day window of demand, stock, and shortage for PPO vs MeanDemand."""
    if not os.path.exists(trace_path):
        print(f"  Skipping trace plot: {trace_path} not found")
        return

    with open(trace_path) as f:
        traces = json.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(COL_WIDTH * 2, 3.5), sharex=True)

    t = np.arange(days)
    for i, (name, color_d, color_s) in enumerate([
        ("PPO", "#2196F3", "#90CAF9"),
        ("MeanDemand", "#FF5722", "#FFAB91"),
    ]):
        if name not in traces:
            continue
        daily = traces[name]["daily"]
        demand = np.array(daily["demand"][:days])
        stock = np.array(daily["stock_end"][:days])
        shortage = np.array(daily["shortage"][:days])

        ax = axes[i]
        ax.fill_between(t, stock, alpha=0.3, color=color_s, label="Stock")
        ax.plot(t, demand, linewidth=0.8, color=color_d, label="Demand")
        ax.plot(t, stock, linewidth=0.8, color=color_s)

        # Mark shortage days
        shortage_days = t[shortage > 0]
        if len(shortage_days) > 0:
            ax.scatter(shortage_days, demand[shortage > 0], marker="x",
                      color="#D32F2F", s=10, zorder=5, label="Shortage")

        ax.set_ylabel("Units", fontsize=7)
        ax.set_title(f"{name}", fontsize=8)
        ax.legend(fontsize=5, loc="upper right")
        ax.tick_params(labelsize=6)

    axes[-1].set_xlabel("Day", fontsize=7)
    fig.suptitle("Daily demand, stock & shortages (seed 1000, multi)", fontsize=8, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_name)


# --------------------------------------------------------------------------- #
# 5. Trade-off frontier
# --------------------------------------------------------------------------- #
def plot_tradeoff():
    """Expiry rate (x) vs shortage rate (y) of every evaluated policy."""
    eval_path = os.path.join(RES_DIR, "eval_multi.csv")

    if not os.path.exists(eval_path):
        print("  Skipping tradeoff plot: required files not found")
        return

    eval_df = pd.read_csv(eval_path)

    fig, ax = plt.subplots(figsize=(COL_WIDTH, COL_WIDTH * 0.8))

    # Evaluated policies as points
    colors = {"Random": "#9E9E9E", "MeanDemand": "#2196F3", "FixedOrder": "#4CAF50",
              "YesterdayDemand": "#795548", "PPO": "#E91E63", "DQN": "#FF9800"}
    markers = {"Random": "s", "MeanDemand": "D", "FixedOrder": "^", "YesterdayDemand": "v",
               "PPO": "*", "DQN": "p"}

    for policy in eval_df["policy"].unique():
        sub = eval_df[eval_df["policy"] == policy]
        ax.scatter(sub["expiry_rate"].mean(), sub["shortage_rate"].mean(),
                  color=colors.get(policy, "#000000"),
                  marker=markers.get(policy, "o"),
                  s=50, zorder=5, label=policy, edgecolors="black", linewidths=0.5)

    ax.set_xlabel("Expiry Rate", fontsize=7)
    ax.set_ylabel("Shortage Rate", fontsize=7)
    ax.set_title("Shortage vs Expiry Trade-off (Multi)", fontsize=8)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=6)

    fig.tight_layout()
    _savefig(fig, "tradeoff.png")


# --------------------------------------------------------------------------- #
# 6. Per-group comparison (optional)
# --------------------------------------------------------------------------- #
def plot_per_group():
    """Per blood group shortage & expiry for PPO vs MeanDemand."""
    trace_path = os.path.join(RES_DIR, "trace_multi.json")
    if not os.path.exists(trace_path):
        print("  Skipping per-group plot: trace_multi.json not found")
        return

    with open(trace_path) as f:
        traces = json.load(f)

    if "PPO" not in traces or "MeanDemand" not in traces:
        print("  Skipping per-group plot: PPO or MeanDemand trace missing")
        return

    groups = traces["PPO"]["groups"]
    G = len(groups)

    fig, axes = plt.subplots(1, 2, figsize=(COL_WIDTH * 2 + 0.5, 2.8))

    x = np.arange(G)
    w = 0.35

    for i, (metric, title) in enumerate([("shortage", "Shortage"), ("expired", "Expiry")]):
        ax = axes[i]
        for j, (name, color) in enumerate([("MeanDemand", "#4CAF50"), ("PPO", "#E91E63")]):
            vals = np.array(traces[name]["per_group"][metric], dtype=float)
            demand = np.array(traces[name]["per_group"]["demand"], dtype=float)
            rates = np.where(demand > 0, vals / demand, 0)
            ax.bar(x + j * w - w / 2, rates, w, color=color, alpha=0.85, label=name)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=5, rotation=45)
        ax.set_ylabel(f"{title} Rate", fontsize=7)
        ax.set_title(f"{title} by Blood Group", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    _savefig(fig, "per_group_multi.png")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Generating figures ...")

    # Get MeanDemand reward for learning curve reference lines
    ref_multi_reward, ref_single_reward = None, None
    eval_multi_path = os.path.join(RES_DIR, "eval_multi.csv")
    eval_single_path = os.path.join(RES_DIR, "eval_single.csv")

    if os.path.exists(eval_multi_path):
        df = pd.read_csv(eval_multi_path)
        ref = df[df["policy"] == "MeanDemand"]
        if len(ref) > 0:
            ref_multi_reward = ref["total_reward"].mean()

    if os.path.exists(eval_single_path):
        df = pd.read_csv(eval_single_path)
        ref = df[df["policy"] == "MeanDemand"]
        if len(ref) > 0:
            ref_single_reward = ref["total_reward"].mean()

    plot_learning_curves(ref_multi_reward, ref_single_reward)
    plot_comparison(eval_multi_path, "compare_multi.png", "Multi (8 groups)")
    plot_comparison(eval_single_path, "compare_single.png", f"Single (O+)")
    plot_trace(os.path.join(RES_DIR, "trace_multi.json"), "trace_multi.png")
    plot_tradeoff()
    plot_per_group()

    print("Done.")


if __name__ == "__main__":
    main()
