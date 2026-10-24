"""Train DQN on the single-group blood bank environment.

Usage:
    python -m agents.train_dqn
    python -m agents.train_dqn --group O+ --timesteps 200000
"""
import argparse
import os
import time

from stable_baselines3 import DQN

from env import make_env
from env.config import SINGLE_GROUP_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Train DQN on single-group blood bank env")
    parser.add_argument("--group", default=SINGLE_GROUP_DEFAULT)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tag = "dqn_single"
    log_dir = os.path.join("results", "train_logs", tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"[DQN] group={args.group}, timesteps={args.timesteps}, seed={args.seed}")

    env = make_env("single", args.group)

    # Wrap with Monitor for logging
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env, filename=os.path.join(log_dir, "monitor"))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2_000,
        exploration_fraction=0.3,
        exploration_final_eps=0.02,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.perf_counter() - t0

    save_path = os.path.join("models", tag)
    model.save(save_path)
    print(f"[DQN] Saved to {save_path}.zip  ({elapsed:.1f}s wall time)")

    env.close()


if __name__ == "__main__":
    main()
