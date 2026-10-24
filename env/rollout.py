"""Run one full episode of a policy and collect metrics."""
import numpy as np

COUNT_KEYS = ["demand", "issued", "substituted", "shortage", "expired", "arrived", "ordered"]
DAILY_KEYS = ["demand", "shortage", "expired", "stock_end", "ordered"]
GROUP_KEYS = ["demand", "shortage", "expired"]


def run_episode(env, policy, seed: int) -> dict:
    """Reset with `seed`, act with `policy.act(obs, env)` until truncation, return episode metrics."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    daily = {k: [] for k in DAILY_KEYS}
    per_group = {k: 0 for k in COUNT_KEYS}

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs, env))
        total_reward += reward
        for k in DAILY_KEYS:
            daily[k].append(int(info[k].sum()))
        for k in per_group:
            per_group[k] = per_group[k] + info[k]
        done = terminated or truncated

    out = {"total_reward": total_reward}
    out.update({k: int(per_group[k].sum()) for k in COUNT_KEYS})
    out["avg_stock"] = float(np.mean(daily["stock_end"]))
    out["shortage_rate"] = out["shortage"] / out["demand"] if out["demand"] else 0.0
    out["expiry_rate"] = out["expired"] / out["arrived"] if out["arrived"] else 0.0
    out["service_level"] = 1.0 - out["shortage_rate"]
    out["daily"] = daily
    out["per_group"] = {k: [int(x) for x in per_group[k]] for k in GROUP_KEYS}
    out["groups"] = list(env.unwrapped.groups)
    return out
