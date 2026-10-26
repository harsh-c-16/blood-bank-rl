"""Run one full episode of a policy and collect metrics."""

COUNT_KEYS = ["demand", "issued", "substituted", "shortage", "expired", "arrived", "ordered"]


def run_episode(env, policy, seed: int) -> dict:
    """Reset with `seed`, act with `policy.act(obs, env)` until truncation, return episode metrics."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    counts = {k: 0 for k in COUNT_KEYS}

    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs, env))
        total_reward += reward
        for k in counts:
            counts[k] += int(info[k].sum())
        done = terminated or truncated

    out = {"total_reward": total_reward, **counts}
    out["shortage_rate"] = out["shortage"] / out["demand"] if out["demand"] else 0.0
    out["expiry_rate"] = out["expired"] / out["arrived"] if out["arrived"] else 0.0
    return out
