#!/usr/bin/env bash
# run_all.sh — run the entire pipeline from scratch.
# Use SKIP_TRAIN=1 bash run_all.sh to skip training and use committed models.
set -e

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Running tests ==="
python -m pytest -q

echo "=== Tuning base-stock baseline ==="
python -m baselines.tune_base_stock

echo "=== Generating demand profile figure ==="
python -m scripts.plot_demand

if [ "$SKIP_TRAIN" != "1" ]; then
    echo "=== Training PPO (multi) ==="
    python -m agents.train_ppo --mode multi

    echo "=== Training PPO (single) ==="
    python -m agents.train_ppo --mode single

    echo "=== Training DQN (single) ==="
    python -m agents.train_dqn
else
    echo "=== SKIP_TRAIN=1: using committed models ==="
fi

echo "=== Evaluating all policies ==="
python -m evaluate

echo "=== Generating plots ==="
python -m plots.make_plots

echo "=== All done ==="
