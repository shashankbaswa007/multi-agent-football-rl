#!/usr/bin/env python3
"""
Comprehensive System Check - Verify all components work correctly
"""

import sys
import os
import torch
import numpy as np
import yaml
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_section(text):
    print("\n" + "-"*70)
    print(f"  {text}")
    print("-"*70)

def check_imports():
    """Check all required imports"""
    print_section("Checking Imports")
    
    checks = []
    
    try:
        from env.football_env import FootballEnv
        checks.append(("FootballEnv", True, "Core environment"))
    except Exception as e:
        checks.append(("FootballEnv", False, str(e)))
    
    try:
        from easy_start_env import EasyStartEnv
        checks.append(("EasyStartEnv", True, "Curriculum wrapper"))
    except Exception as e:
        checks.append(("EasyStartEnv", False, str(e)))
    
    try:
        from models.ppo_agent import PPOAgent
        checks.append(("PPOAgent", True, "RL agent"))
    except Exception as e:
        checks.append(("PPOAgent", False, str(e)))
    
    try:
        from training.buffer import MultiAgentBuffer
        checks.append(("MultiAgentBuffer", True, "Replay buffer"))
    except Exception as e:
        checks.append(("MultiAgentBuffer", False, str(e)))
    
    for name, success, msg in checks:
        status = "✅" if success else "❌"
        print(f"  {status} {name:20s} - {msg}")
    
    return all(s for _, s, _ in checks)

def check_environment():
    """Check environment mechanics"""
    print_section("Checking Environment Mechanics")
    
    from easy_start_env import EasyStartEnv
    
    env = EasyStartEnv(num_agents_per_team=1, grid_width=8, grid_height=4,
                       max_steps=100, difficulty='easy')
    obs, _ = env.reset()
    
    checks = []
    
    # Check 1: One-block movement
    agent = 'team_0_agent_0'
    pos1 = env.agent_positions[agent].copy()
    env._execute_action(agent, 4)  # MOVE_RIGHT
    pos2 = env.agent_positions[agent].copy()
    dist = np.linalg.norm(pos2 - pos1)
    checks.append(("One-block movement", 0.9 <= dist <= 1.1, f"{dist:.2f} units"))
    
    # Check 2: Shooting distance limit
    env.agent_positions[agent] = np.array([2.0, 2.0])
    env.ball_possession = agent
    reward_far = env._execute_action(agent, 6)  # SHOOT from far
    checks.append(("Shooting distance limit", reward_far < 0, f"Reward: {reward_far:.1f}"))
    
    # Check 3: Starting position realistic
    env.reset()
    pos = env.agent_positions[agent]
    goal = np.array([env.grid_width - 1, env.grid_height // 2])
    start_dist = np.linalg.norm(pos - goal)
    checks.append(("Realistic start position", 3.0 <= start_dist <= 6.0, f"{start_dist:.1f} units from goal"))
    
    # Check 4: Can score from close
    env.agent_positions[agent] = np.array([env.grid_width - 2, env.grid_height // 2])
    env.ball_possession = agent
    reward_close = env._execute_action(agent, 6)  # SHOOT from close
    checks.append(("Close-range shooting works", reward_close > 0, f"Reward: {reward_close:.1f}"))
    
    for name, success, msg in checks:
        status = "✅" if success else "❌"
        print(f"  {status} {name:25s} - {msg}")
    
    return all(s for _, s, _ in checks)

def check_configs():
    """Check configuration files"""
    print_section("Checking Configuration Files")
    
    configs = [
        "configs/stage1_stable.yaml",
        "configs/stage1_realistic.yaml",
        "configs/default_config.yaml"
    ]
    
    checks = []
    for config_path in configs:
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                required_keys = ['num_agents_per_team', 'grid_width', 'ppo_params']
                has_all = all(k in config for k in required_keys)
                checks.append((config_path, has_all, "Valid" if has_all else "Missing keys"))
            except Exception as e:
                checks.append((config_path, False, str(e)))
        else:
            checks.append((config_path, False, "File not found"))
    
    for name, success, msg in checks:
        status = "✅" if success else "❌"
        print(f"  {status} {Path(name).name:25s} - {msg}")
    
    return all(s for _, s, _ in checks)

def check_trained_models():
    """Check for trained models"""
    print_section("Checking Trained Models")
    
    runs_dir = Path("runs")
    
    if not runs_dir.exists():
        print("  ❌ No 'runs' directory found")
        return False
    
    run_dirs = sorted(runs_dir.glob("run_*"))
    
    if not run_dirs:
        print("  ⚠️  No training runs found")
        print("  ℹ️  Train a model with: python train_ppo.py --config configs/stage1_realistic.yaml")
        return False
    
    for run_dir in run_dirs:
        checkpoint_dir = run_dir / "checkpoints"
        if checkpoint_dir.exists():
            checkpoints = list(checkpoint_dir.glob("*.pt"))
            print(f"  📁 {run_dir.name}: {len(checkpoints)} checkpoint(s)")
            
            # Check if best_model exists
            if (checkpoint_dir / "best_model.pt").exists():
                print(f"     ✅ best_model.pt available")
            if (checkpoint_dir / "final_model.pt").exists():
                print(f"     ✅ final_model.pt available")
    
    return True

def check_training_compatibility():
    """Check if old models work with new mechanics"""
    print_section("Checking Model Compatibility")
    
    runs_dir = Path("runs")
    if not runs_dir.exists() or not list(runs_dir.glob("run_*/checkpoints/*.pt")):
        print("  ℹ️  No trained models to check")
        return True
    
    print("  ⚠️  WARNING: Existing models trained on OLD mechanics")
    print("  ℹ️  Old mechanics:")
    print("     - Agents started very close to goal (2-3 units)")
    print("     - Could shoot from anywhere")
    print("     - No shot distance limits")
    print()
    print("  ✅ New REALISTIC mechanics:")
    print("     - Agents start at midfield (4-5 units)")
    print("     - Must dribble forward step-by-step")
    print("     - Can only shoot within 4 units of goal")
    print()
    print("  📝 Recommendation: Retrain with new mechanics")
    print("     python train_ppo.py --config configs/stage1_realistic.yaml")
    
    return True

def check_scripts():
    """Check test scripts"""
    print_section("Checking Test Scripts")
    
    scripts = [
        ("test_realistic_football.py", "Comprehensive mechanics tests"),
        ("demo_realistic_mechanics.py", "Visual demonstration"),
        ("test_trained_model.py", "Model testing"),
        ("demo_model.py", "Quick model stats"),
        ("app.py", "Streamlit visualization"),
    ]
    
    for script, description in scripts:
        exists = Path(script).exists()
        status = "✅" if exists else "❌"
        print(f"  {status} {script:30s} - {description}")
    
    return True

def main():
    print_header("🏃‍♂️ COMPREHENSIVE SYSTEM CHECK")
    
    checks = [
        ("Imports", check_imports),
        ("Environment", check_environment),
        ("Configurations", check_configs),
        ("Trained Models", check_trained_models),
        ("Compatibility", check_training_compatibility),
        ("Scripts", check_scripts),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("📊 SUMMARY")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    
    print(f"\n  Total: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n  Next Steps:")
        print("  1. Train new model: python train_ppo.py --config configs/stage1_realistic.yaml")
        print("  2. Test training: python test_realistic_football.py")
        print("  3. Visualize: streamlit run app.py")
    else:
        print(f"\n  ⚠️  {total - passed} check(s) need attention")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
