"""Action wrappers for the blood bank environment."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import config as C


class DiscreteActionWrapper(gym.ActionWrapper):
    """Turns the single-group MultiDiscrete action into Discrete(MAX_ORDER+1), for DQN."""

    def __init__(self, env):
        super().__init__(env)
        assert len(env.unwrapped.groups) == 1, "DiscreteActionWrapper needs exactly one blood group"
        self.action_space = spaces.Discrete(C.MAX_ORDER + 1)

    def action(self, action):
        return np.array([int(action)], dtype=np.int64)
