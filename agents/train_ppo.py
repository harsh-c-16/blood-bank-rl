"""Train PPO on the blood bank environment (multi or single mode).

Usage:
    python -m agents.train_ppo --mode multi
    python -m agents.train_ppo --mode single --group O+
    python -m agents.train_ppo --mode multi --timesteps 1000000
"""
import argparse
import os
import time

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from env import make_env
from env.config import SINGLE_GROUP_DEFAULT


def main():
    parser = argparse.ArgumentParser(description="Train PPO on the blood bank env")
    parser.add_argument("--mode", choices=["multi", "single"], default="multi")
    parser.add_argument("--group", default=SINGLE_GROUP_DEFAULT)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    default_ts = 500_000 if args.mode == "multi" else 200_000
    timesteps = args.timesteps or default_ts
    tag = f"ppo_{args.mode}"

    log_dir = os.path.join("results", "train_logs", tag)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"[PPO] mode={args.mode}, group={args.group}, timesteps={timesteps}, seed={args.seed}")

    vec_env = make_vec_env(
        lambda: make_env(args.mode, args.group),
        n_envs=8,
        seed=args.seed * 100,
        monitor_dir=log_dir,
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[128, 128]),
        seed=args.seed,
        verbose=1,
    )

    t0 = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    elapsed = time.perf_counter() - t0

    save_path = os.path.join("models", f"{tag}")
    model.save(save_path)
    print(f"[PPO] Saved to {save_path}.zip  ({elapsed:.1f}s wall time)")

    vec_env.close()


if __name__ == "__main__":
    main()
