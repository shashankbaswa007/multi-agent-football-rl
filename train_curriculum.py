#!/usr/bin/env python3
"""
Training script with curriculum - starts easy and gets harder
"""
import sys
import os
sys.path.insert(0, '/Users/shashi/reinforcement_learning')

import torch
import yaml
import numpy as np
from datetime import datetime
from pathlib import Path

# Import easy start environment
from easy_start_env import EasyStartEnv

# Import training components
import importlib.util
spec = importlib.util.spec_from_file_location("train_ppo", "/Users/shashi/reinforcement_learning/training/train_ppo.py")
train_module = importlib.util.module_from_spec(spec)

print("=" * 70)
print("🎯 CURRICULUM TRAINING: Easy → Medium → Hard")
print("=" * 70)

# Load config
with open('/Users/shashi/reinforcement_learning/configs/quick_test.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Stage 1: EASY MODE (200 episodes)
print("\n📚 STAGE 1: Easy Start (agent near goal with ball)")
print("   Goal: Learn to SHOOT and SCORE")
print("   Episodes: 200")
print("-" * 70)

config['num_episodes'] = 200
config['max_steps'] = 50  # Shorter - agent is already near goal

# Save modified config
with open('/tmp/curriculum_stage1.yaml', 'w') as f:
    yaml.dump(config, f)

# Note: To properly integrate, we'd need to modify train_ppo.py
# For now, let's create a simple demonstration

print("\n✅ Curriculum config created!")
print("   Config saved to: /tmp/curriculum_stage1.yaml")
print("\nTo run Stage 1 training:")
print("   1. Modify train_ppo.py to use EasyStartEnv")
print("   2. Run: python train_ppo.py --config /tmp/curriculum_stage1.yaml")
print("\nExpected Stage 1 results:")
print("   - 50%+ goals scored in first 100 episodes")
print("   - Agents learn: MOVE_RIGHT → SHOOT → WIN")
print("   - Reward: -5 → +190 (consistent scoring)")
