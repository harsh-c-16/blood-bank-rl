import numpy as np
import pytest
from gymnasium import spaces

from env import BloodBankEnv, make_env
from env import config as C
from env.rollout import run_episode


class RandomActions:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, env):
        n = len(env.unwrapped.groups)
        a = self.rng.integers(0, C.MAX_ORDER + 1, size=n)
        return int(a[0]) if isinstance(env.action_space, spaces.Discrete) else a


@pytest.mark.parametrize("mode,size", [("multi", 111), ("single", 20)])
def test_observation(mode, size):
    env = make_env(mode)
    obs, info = env.reset(seed=0)
    assert info == {}
    assert env.observation_space.shape == (size,)
    policy = RandomActions()
    for _ in range(50):
        assert obs.shape == (size,) and obs.dtype == np.float32
        assert not np.isnan(obs).any() and (obs >= 0).all()
        assert env.observation_space.contains(obs)
        obs, *_ = env.step(policy.act(obs, env))


def test_action_spaces():
    multi = make_env("multi")
    single = make_env("single")
    assert isinstance(multi.action_space, spaces.MultiDiscrete)
    assert list(multi.action_space.nvec) == [C.MAX_ORDER + 1] * 8
    assert isinstance(single.action_space, spaces.Discrete)
    assert single.action_space.n == C.MAX_ORDER + 1
    assert single.unwrapped.groups == [C.SINGLE_GROUP_DEFAULT]


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_stock_never_negative_and_conservation(mode):
    env = make_env(mode)
    obs, _ = env.reset(seed=3)
    base = env.unwrapped
    start = base.stock.sum()
    totals = {k: 0 for k in ["arrived", "issued", "substituted", "expired"]}
    policy = RandomActions(1)
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step(policy.act(obs, env))
        assert (base.stock >= 0).all()
        assert (info["shortage"] >= 0).all()
        assert (info["stock_end"] == base.stock.sum(axis=1)).all()
        for k in totals:
            totals[k] += info[k].sum()
    lhs = start + totals["arrived"]
    rhs = totals["issued"] + totals["substituted"] + totals["expired"] + base.stock.sum()
    assert lhs == rhs


def test_no_substitution_when_disabled():
    for env in [make_env("multi", substitution=False), make_env("single")]:
        obs, _ = env.reset(seed=4)
        policy = RandomActions(2)
        for _ in range(C.EPISODE_DAYS):
            obs, _, _, _, info = env.step(policy.act(obs, env))
            assert (info["substituted"] == 0).all()


def test_substitution_happens():
    # with nothing ordered, rare groups run out and borrow from compatible donors
    env = make_env("multi")
    env.reset(seed=5)
    used = 0
    for _ in range(30):
        used += env.step(np.zeros(8, dtype=int))[4]["substituted"].sum()
    assert used > 0


def test_seeding():
    env = make_env("multi")
    a = run_episode(env, RandomActions(0), seed=7)
    b = run_episode(env, RandomActions(0), seed=7)
    c = run_episode(env, RandomActions(0), seed=8)
    assert a["daily"] == b["daily"] and a["total_reward"] == b["total_reward"]
    assert a["daily"]["demand"] != c["daily"]["demand"]


@pytest.mark.parametrize("mode", ["multi", "single"])
def test_truncation(mode):
    env = make_env(mode)
    obs, _ = env.reset(seed=0)
    policy = RandomActions()
    for day in range(1, C.EPISODE_DAYS + 1):
        obs, _, terminated, truncated, info = env.step(policy.act(obs, env))
        assert not terminated
        assert truncated == (day == C.EPISODE_DAYS)
        assert info["weekday"] == (day - 1) % 7


def test_run_episode_keys():
    out = run_episode(make_env("multi"), RandomActions(), seed=0)
    scalars = ["total_reward", "demand", "issued", "substituted", "shortage", "expired", "arrived",
               "ordered", "avg_stock", "shortage_rate", "expiry_rate", "service_level"]
    for k in scalars:
        assert np.isscalar(out[k]), k
    assert set(out["daily"]) == {"demand", "shortage", "expired", "stock_end", "ordered"}
    assert all(len(v) == C.EPISODE_DAYS for v in out["daily"].values())
    assert set(out["per_group"]) == {"demand", "shortage", "expired"}
    assert all(len(v) == 8 for v in out["per_group"].values())
    assert out["groups"] == C.GROUPS
    assert out["service_level"] == pytest.approx(1 - out["shortage_rate"])


def test_mean_demand_matches_target():
    env = BloodBankEnv(groups=C.GROUPS, substitution=True)
    days = []
    for seed in range(30):
        env.reset(seed=seed)
        for _ in range(C.EPISODE_DAYS):
            days.append(env.step(np.zeros(8, dtype=int))[4]["demand"].sum())
    assert np.mean(days) == pytest.approx(C.WEEKDAY_MEAN_TOTAL.mean(), rel=0.05)


def test_demand_overdispersed():
    env = BloodBankEnv(groups=["O+"], substitution=False)
    monday = []
    for seed in range(40):
        env.reset(seed=seed)
        for _ in range(C.EPISODE_DAYS):
            info = env.step(np.zeros(1, dtype=int))[4]
            if info["weekday"] == 0:
                monday.append(info["demand"][0])
    ratio = np.var(monday) / np.mean(monday)
    assert ratio == pytest.approx(C.DISPERSION, rel=0.15)


def test_unseeded_reset_continues_seeded_stream():
    # training libraries seed only the first reset, later episodes must still be reproducible
    def demands(env):
        env.reset(seed=11)
        out = []
        for _ in range(3):
            env.reset()
            out.append([env.step(np.zeros(8, dtype=int))[4]["demand"].tolist() for _ in range(5)])
        return out
    assert demands(make_env("multi")) == demands(make_env("multi"))
