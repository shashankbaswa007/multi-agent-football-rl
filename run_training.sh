#!/bin/bash
# Convenient script to run training with the correct virtual environment

# Activate virtual environment
source .venv/bin/activate

# Run training with emergency fix config
python training/train_ppo.py --config configs/emergency_fix.yaml "$@"
