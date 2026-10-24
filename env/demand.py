"""Daily platelet demand: negative binomial, calibrated to published hospital statistics."""
import numpy as np

from env import config as C


class DemandModel:
    """Demand per blood group with weekday-dependent mean and variance = DISPERSION * mean."""

    def __init__(self, groups: list[str], rng: np.random.Generator):
        self.share = np.array([C.GROUP_SHARE[C.GROUPS.index(g)] for g in groups])
        self.rng = rng

    def expected(self, weekday: int) -> np.ndarray:
        """Mean demand per group for the given weekday (0 = Monday)."""
        return C.WEEKDAY_MEAN_TOTAL[weekday] * self.share

    def sample(self, weekday: int) -> np.ndarray:
        """One day of demand per group. NB(r, p) with r = m/(D-1), p = r/(r+m) has mean m, variance D*m."""
        out = np.zeros(len(self.share), dtype=np.int64)
        for g, m in enumerate(self.expected(weekday)):
            if m > 0:
                r = m / (C.DISPERSION - 1)
                out[g] = self.rng.negative_binomial(r, r / (r + m))
        return out
