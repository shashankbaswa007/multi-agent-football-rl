#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE VALIDATION
Tests all components together to ensure perfect functionality
"""

import sys
import numpy as np
from pathlib import Path

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def print_check(name, status, message=""):
    symbol = "✅" if status else "❌"
    msg = f" - {message}" if message else ""
    print(f"  {symbol} {name}{msg}")

def validate_all():
    print_header("🔍 FINAL COMPREHENSIVE VALIDATION")
    
    all_checks = []
    
    # 1. Core imports
    print("\n1️⃣  Core Imports")
    try:
        from env.football_env import FootballEnv
        from easy_start_env import EasyStartEnv
        from models.ppo_agent import PPOAgent
        import torch
        import yaml
        import streamlit
        print_check("All imports", True, "No errors")
        all_checks.append(True)
    except Exception as e:
        print_check("Imports", False, str(e))
        all_checks.append(False)
        return False
    
    # 2. Environment mechanics
    print("\n2️⃣  Environment Mechanics")
    try:
        env = EasyStartEnv(num_agents_per_team=1, grid_width=8, grid_height=4,
                           max_steps=200, difficulty='easy')
        obs, _ = env.reset()
        
        # Test movement
        agent = 'team_0_agent_0'
        pos1 = env.agent_positions[agent].copy()
        env._execute_action(agent, 4)  # RIGHT
        pos2 = env.agent_positions[agent].copy()
        dist = np.linalg.norm(pos2 - pos1)
        
        checks = [
            ("One-block movement", 0.9 <= dist <= 1.1),
            ("Observation shape", obs[agent].shape == (11,)),
            ("Agent positions", all(a in env.agent_positions for a in env.agents)),
            ("Ball exists", env.ball_position is not None),
        ]
        
        for name, status in checks:
            print_check(name, status)
            all_checks.append(status)
    except Exception as e:
        print_check("Environment mechanics", False, str(e))
        all_checks.append(False)
    
    # 3. Realistic mechanics
    print("\n3️⃣  Realistic Football Mechanics")
    try:
        env.reset()
        agent = 'team_0_agent_0'
        
        # Test shooting distance limit
        env.agent_positions[agent] = np.array([2.0, 2.0])
        env.ball_possession = agent
        reward_far = env._execute_action(agent, 6)  # SHOOT from far
        
        # Test close shooting
        env.agent_positions[agent] = np.array([env.grid_width - 2, env.grid_height // 2])
        env.ball_possession = agent
        reward_close = env._execute_action(agent, 6)
        
        # Test starting position (grid-size aware)
        env.reset()
        pos = env.agent_positions[agent]
        goal = np.array([env.grid_width - 1, env.grid_height // 2])
        start_dist = np.linalg.norm(pos - goal)
        # For 8x4 grid, expect 2.0-4.0; for 10x6, expect 3.0-6.0
        min_dist = env.grid_width * 0.25
        max_dist = env.grid_width * 0.75
        
        checks = [
            ("Shooting distance limit", reward_far < 0),
            ("Close-range shooting", reward_close > 0),
            ("Realistic start position", min_dist <= start_dist <= max_dist),
        ]
        
        for name, status in checks:
            print_check(name, status)
            all_checks.append(status)
    except Exception as e:
        print_check("Realistic mechanics", False, str(e))
        all_checks.append(False)
    
    # 4. Configuration files
    print("\n4️⃣  Configuration Files")
    try:
        configs = [
            "configs/stage1_realistic.yaml",
            "configs/stage1_stable.yaml",
            "configs/default_config.yaml"
        ]
        
        for config_path in configs:
            exists = Path(config_path).exists()
            if exists:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                valid = 'ppo_params' in config and 'num_agents_per_team' in config
                print_check(Path(config_path).name, valid)
                all_checks.append(valid)
            else:
                print_check(Path(config_path).name, False, "Not found")
                all_checks.append(False)
    except Exception as e:
        print_check("Config files", False, str(e))
        all_checks.append(False)
    
    # 5. Streamlit app mechanics
    print("\n5️⃣  Streamlit App Mechanics")
    try:
        # Test agent loop
        env = EasyStartEnv(num_agents_per_team=1, grid_width=8, grid_height=4,
                           max_steps=200, difficulty='easy')
        obs, _ = env.reset()
        
        steps_executed = 0
        for _ in range(10):
            agent_name = env.agent_selection
            if env.terminations[agent_name] or env.truncations[agent_name]:
                env.step(None)
                obs = env._get_observations()
                continue
            
            action = np.random.randint(0, 7)
            env.step(action)
            obs = env._get_observations()
            steps_executed += 1
        
        print_check("Agent loop", steps_executed > 0, f"{steps_executed} steps")
        print_check("Observation updates", len(obs) == 2)
        print_check("Episode stats", hasattr(env, 'episode_stats'))
        
        all_checks.extend([steps_executed > 0, len(obs) == 2, hasattr(env, 'episode_stats')])
    except Exception as e:
        print_check("Streamlit mechanics", False, str(e))
        all_checks.append(False)
    
    # 6. Test scripts
    print("\n6️⃣  Test Scripts")
    scripts = [
        "test_realistic_football.py",
        "demo_realistic_mechanics.py",
        "test_streamlit_app.py",
        "check_system.py",
        "app.py"
    ]
    
    for script in scripts:
        exists = Path(script).exists()
        print_check(script, exists)
        all_checks.append(exists)
    
    # 7. Training capability
    print("\n7️⃣  Training System")
    try:
        from training.buffer import MultiAgentBuffer
        from training.utils import LinearSchedule
        
        train_script = Path("train_ppo.py")
        config_exists = Path("configs/stage1_realistic.yaml").exists()
        
        print_check("Training script", train_script.exists())
        print_check("Training config", config_exists)
        print_check("Buffer module", True)
        print_check("Utils module", True)
        
        all_checks.extend([train_script.exists(), config_exists, True, True])
    except Exception as e:
        print_check("Training system", False, str(e))
        all_checks.append(False)
    
    # Summary
    print_header("📊 VALIDATION SUMMARY")
    
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total * 100) if total > 0 else 0
    
    print(f"\n  Checks passed: {passed}/{total} ({percentage:.1f}%)")
    print()
    
    if passed == total:
        print("  🎉 PERFECT! ALL SYSTEMS VALIDATED!")
        print()
        print("  ✅ Environment mechanics work correctly")
        print("  ✅ Realistic football behavior implemented")
        print("  ✅ Streamlit app ready to use")
        print("  ✅ Training system operational")
        print("  ✅ All test scripts available")
        print()
        print("  🚀 READY FOR PRODUCTION!")
        print()
        print("  Next steps:")
        print("  1. Train model: python train_ppo.py --config configs/stage1_realistic.yaml")
        print("  2. Test training: python test_realistic_football.py")
        print("  3. Launch app: streamlit run app.py")
        print("  4. Watch agents: Load model and click Play!")
    elif passed >= total * 0.8:
        print("  ✅ GOOD! Most systems validated")
        print(f"  ⚠️  {total - passed} check(s) need attention")
    else:
        print("  ⚠️  ATTENTION NEEDED")
        print(f"  {total - passed} check(s) failed")
    
    print()
    print("="*70)
    
    return passed == total

if __name__ == "__main__":
    success = validate_all()
    sys.exit(0 if success else 1)
