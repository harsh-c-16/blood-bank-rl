"""Plot the simulated demand against the published statistics it is calibrated to.

(a) mean total demand per weekday vs the UCLH values, (b) histogram of daily O+ demand.
Usage: python -m scripts.plot_demand  ->  figures/demand_profile.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import BloodBankEnv
from env import config as C

N_EPISODES = 30
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64", "#e4e3dd"


def simulate():
    """Demand per day (all groups) with zero orders; ordering does not affect demand."""
    env = BloodBankEnv(groups=C.GROUPS, substitution=True)
    totals, weekdays, o_pos = [], [], []
    for seed in range(N_EPISODES):
        env.reset(seed=seed)
        for _ in range(C.EPISODE_DAYS):
            info = env.step(np.zeros(len(C.GROUPS), dtype=int))[4]
            totals.append(info["demand"].sum())
            weekdays.append(info["weekday"])
            o_pos.append(info["demand"][C.GROUPS.index("O+")])
    return np.array(totals), np.array(weekdays), np.array(o_pos)


def main():
    totals, weekdays, o_pos = simulate()
    sim_mean = [totals[weekdays == w].mean() for w in range(7)]

    plt.rcParams.update({"font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.4, 1.8), constrained_layout=True)

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    x = np.arange(7)
    ax1.bar(x, sim_mean, width=0.6, color=BLUE, label="simulated", zorder=2)
    ax1.plot(x, C.WEEKDAY_MEAN_TOTAL, "o", ms=4, color=ORANGE, markeredgecolor="white",
             markeredgewidth=0.8, label="UCLH data", zorder=3)
    ax1.set_xticks(x, [d[0] for d in days])
    ax1.set_ylabel("mean units / day")
    ax1.set_xlabel("weekday")
    ax1.set_title("(a) total demand by weekday", fontsize=7, color=INK)
    ax1.set_ylim(0, 44)
    ax1.set_yticks([0, 10, 20, 30])
    ax1.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2, handlelength=1, columnspacing=1)

    bins = np.arange(o_pos.max() + 2) - 0.5
    ax2.hist(o_pos, bins=bins, color=BLUE, edgecolor="white", linewidth=0.4, zorder=2)
    ax2.set_xlabel("O+ units / day")
    ax2.set_ylabel("days")
    ax2.set_title("(b) daily O+ demand", fontsize=7, color=INK)
    ratio = o_pos.var() / o_pos.mean()
    ax2.text(0.97, 0.95, f"mean {o_pos.mean():.1f}\nvar/mean {ratio:.1f}", transform=ax2.transAxes,
             ha="right", va="top", fontsize=6, color=INK)

    for ax in (ax1, ax2):
        ax.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)

    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/demand_profile.png", dpi=300)
    print("saved figures/demand_profile.png")
    print("simulated weekday means:", np.round(sim_mean, 1))
    print(f"O+ mean {o_pos.mean():.2f}, var/mean {ratio:.2f}")


if __name__ == "__main__":
    main()
