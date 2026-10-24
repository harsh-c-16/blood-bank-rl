"""Simulation and reward constants for the blood bank environment."""
import numpy as np

GROUPS = ["O+", "O-", "A+", "A-", "B+", "B-", "AB+", "AB-"]
GROUP_SHARE = np.array([0.33, 0.025, 0.21, 0.015, 0.31, 0.02, 0.08, 0.01])  # sums to 1.0

WEEKDAY_MEAN_TOTAL = np.array([28.8, 33.4, 26.2, 28.4, 30.8, 18.6, 19.6])  # Mon..Sun, all groups
DISPERSION = 12.3 / 5.4  # variance / mean of daily demand

SHELF_LIFE = 5                                  # max days of life remaining
ARRIVAL_LIFE_VALUES = np.array([3, 4, 5])       # remaining life of a delivered unit
ARRIVAL_LIFE_PROBS = np.array([0.2, 0.3, 0.5])
INIT_DAYS_COVER = 2.0                           # initial stock = 2 days of mean demand

MAX_ORDER = 25          # max units ordered per group per day
HISTORY_DAYS = 7        # days of demand history in the observation
EPISODE_DAYS = 365

# reward = -REWARD_SCALE * (C_SHORT*shortage + C_SUB*substituted + C_EXPIRE*expired
#                           + C_HOLD*stock_end + C_ORDER*ordered)   (all summed over groups)
C_SHORT = 10.0
C_SUB = 0.5
C_EXPIRE = 3.0
C_HOLD = 0.1
C_ORDER = 1.0
REWARD_SCALE = 0.01

STOCK_NORM = 50.0
DEMAND_NORM = 40.0

# Donors allowed for each recipient, in preference order (own group is always tried first, separately).
COMPATIBLE_DONORS = {
    "O+": ["O-"],
    "O-": [],
    "A+": ["A-", "O+", "O-"],
    "A-": ["O-"],
    "B+": ["B-", "O+", "O-"],
    "B-": ["O-"],
    "AB+": ["AB-", "A+", "B+", "A-", "B-", "O+", "O-"],
    "AB-": ["A-", "B-", "O-"],
}

TUNE_SEEDS = list(range(500, 520))    # only for tuning the base-stock baseline
EVAL_SEEDS = list(range(1000, 1030))  # only for final evaluation (never for training or tuning)
SINGLE_GROUP_DEFAULT = "O+"
