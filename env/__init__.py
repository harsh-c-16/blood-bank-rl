"""Blood bank environment package."""
from env.blood_bank_env import BloodBankEnv
from env.config import GROUPS
from env.wrappers import DiscreteActionWrapper

__all__ = ["BloodBankEnv", "make_env"]


def make_env(mode: str = "multi", group: str = "O+", substitution: bool | None = None):
    """'multi': all 8 groups, MultiDiscrete actions (PPO). 'single': one group, Discrete actions (DQN)."""
    if mode == "multi":
        return BloodBankEnv(groups=GROUPS, substitution=True if substitution is None else substitution)
    if mode == "single":
        return DiscreteActionWrapper(BloodBankEnv(groups=[group], substitution=False))
    raise ValueError(f"unknown mode {mode!r}, expected 'multi' or 'single'")
