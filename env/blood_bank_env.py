"""Platelet blood bank inventory environment (Gymnasium API)."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from env import config as C
from env.demand import DemandModel


class BloodBankEnv(gym.Env):
    """Daily ordering of platelets for one or more blood groups.

    stock[g, l] holds units of group g with l+1 days of life left (column 0 expires tonight).
    pipeline[g] is today's order, delivered at the start of the next day.
    """

    metadata = {"render_modes": []}

    def __init__(self, groups: list[str], substitution: bool):
        self.groups = list(groups)
        self.substitution = substitution
        G = len(self.groups)
        self.share = np.array([C.GROUP_SHARE[C.GROUPS.index(g)] for g in self.groups])
        self.mean_daily = C.WEEKDAY_MEAN_TOTAL.mean() * self.share

        self.action_space = spaces.MultiDiscrete([C.MAX_ORDER + 1] * G)
        obs_size = G * C.SHELF_LIFE + G + G * C.HISTORY_DAYS + 7
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.stock = np.zeros((G, C.SHELF_LIFE), dtype=np.int64)
        self.pipeline = np.zeros(G, dtype=np.int64)
        self.history = np.zeros((G, C.HISTORY_DAYS))
        self.day = 0
        self.rng = np.random.default_rng()
        self.demand_model = DemandModel(self.groups, self.rng)

    def expected_demand(self, weekday: int) -> np.ndarray:
        """Mean demand per group for the given weekday (0 = Monday)."""
        return self.demand_model.expected(weekday)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.demand_model.rng = self.rng
        self.day = 0

        # initial stock spread evenly over 3, 4 and 5 days of life, remainder to the freshest column
        self.stock[:] = 0
        init = np.round(C.INIT_DAYS_COVER * self.mean_daily).astype(np.int64)
        self.stock[:, 2:5] = (init // 3)[:, None]
        self.stock[:, 4] += init % 3

        self.pipeline = np.round(self.expected_demand(0)).astype(np.int64)
        self.history[:] = self.mean_daily[:, None]
        return self._obs(), {}

    def step(self, action):
        weekday = self.day % 7
        arrived = self._arrive()
        ordered = np.clip(np.asarray(action, dtype=np.int64).reshape(-1), 0, C.MAX_ORDER)
        self.pipeline = ordered

        demand = self.demand_model.sample(weekday)
        issued = self._issue_own(demand)
        substituted = np.zeros_like(demand)
        shortage = demand - issued - substituted
        expired = self._expire_and_age()
        self.history = np.concatenate([self.history[:, 1:], demand[:, None]], axis=1)

        stock_end = self.stock.sum(axis=1)
        cost = (C.C_SHORT * shortage.sum() + C.C_SUB * substituted.sum() + C.C_EXPIRE * expired.sum()
                + C.C_HOLD * stock_end.sum() + C.C_ORDER * ordered.sum())
        reward = -C.REWARD_SCALE * float(cost)

        self.day += 1
        truncated = self.day == C.EPISODE_DAYS
        info = {"demand": demand, "issued": issued, "substituted": substituted, "shortage": shortage,
                "expired": expired, "arrived": arrived, "ordered": ordered, "stock_end": stock_end,
                "weekday": weekday}
        return self._obs(), reward, False, truncated, info

    def _arrive(self) -> np.ndarray:
        """Move yesterday's order into stock. Each unit arrives with 3, 4 or 5 days of life left."""
        arrived = self.pipeline.copy()
        for g in range(len(self.groups)):
            # drawing a life for each of the n units independently = one multinomial draw of the counts
            counts = self.rng.multinomial(arrived[g], C.ARRIVAL_LIFE_PROBS)
            self.stock[g, C.ARRIVAL_LIFE_VALUES - 1] += counts
        return arrived

    def _issue_own(self, demand: np.ndarray) -> np.ndarray:
        """Serve each group's demand from its own stock, oldest units first."""
        issued = np.zeros_like(demand)
        for g in range(len(self.groups)):
            for life in range(C.SHELF_LIFE):
                take = min(self.stock[g, life], demand[g] - issued[g])
                self.stock[g, life] -= take
                issued[g] += take
        return issued

    def _expire_and_age(self) -> np.ndarray:
        """Discard units expiring today and shift the rest one day older."""
        expired = self.stock[:, 0].copy()
        self.stock[:, :-1] = self.stock[:, 1:]
        self.stock[:, -1] = 0
        return expired

    def _obs(self) -> np.ndarray:
        weekday = np.zeros(7)
        weekday[self.day % 7] = 1.0
        return np.concatenate([
            self.stock.flatten() / C.STOCK_NORM,
            self.pipeline / C.MAX_ORDER,
            self.history.flatten() / C.DEMAND_NORM,
            weekday,
        ]).astype(np.float32)
