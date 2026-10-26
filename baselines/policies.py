"""Simple ordering rules used as baselines for the RL agents.

Each policy has act(obs, env) and reads the true state from env.unwrapped.
They work in both modes: an int action for Discrete, an int array for MultiDiscrete.
"""
import numpy as np
from gymnasium import spaces

from env import config as C


def _to_action(order: np.ndarray, env):
    """Clip orders to the allowed range and match the env's action space type."""
    order = np.clip(np.round(order), 0, C.MAX_ORDER).astype(np.int64)
    if isinstance(env.action_space, spaces.Discrete):
        return int(order[0])
    return order


class RandomPolicy:
    """Orders a uniformly random amount per group."""

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def act(self, obs, env):
        n_groups = len(env.unwrapped.groups)
        return _to_action(self.rng.integers(0, C.MAX_ORDER + 1, size=n_groups), env)


class MeanDemandPolicy:
    """Orders the mean demand of the day the order will arrive (tomorrow)."""

    def act(self, obs, env):
        base = env.unwrapped
        return _to_action(base.expected_demand((base.day + 1) % 7), env)


class FixedOrderPolicy:
    """Orders the same amount every day: the average daily demand of each group."""

    def act(self, obs, env):
        base = env.unwrapped
        mean_daily = np.mean([base.expected_demand(w) for w in range(7)], axis=0)
        return _to_action(mean_daily, env)


class YesterdayDemandPolicy:
    """Orders as many units as were demanded yesterday."""

    def act(self, obs, env):
        return _to_action(env.unwrapped.history[:, -1], env)
