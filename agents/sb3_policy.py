"""Wrapper that gives an SB3 model the same .act(obs, env) interface as the baselines."""
import numpy as np
from gymnasium import spaces


class SB3Policy:
    """Wraps a trained Stable-Baselines3 model to match the policy contract.

    policy.act(obs, env) returns:
      - int           for Discrete action spaces
      - np.ndarray    for MultiDiscrete action spaces
    """

    def __init__(self, model):
        self.model = model

    def act(self, obs, env):
        action, _ = self.model.predict(obs, deterministic=True)
        if isinstance(env.action_space, spaces.Discrete):
            return int(action)
        return np.asarray(action, dtype=int)
