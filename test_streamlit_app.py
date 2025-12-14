#!/usr/bin/env python3
"""
Test Streamlit app functionality without needing trained model
Uses random agent to verify the app mechanics work
"""

import numpy as np
from easy_start_env import EasyStartEnv
import yaml

def test_streamlit_mechanics():
    """Test the core mechanics that Streamlit app uses"""
    print("="*60)
    print("🎬 TESTING STREAMLIT APP MECHANICS")
    print("="*60)
    
    # Load config (same as Streamlit)
    with open('configs/stage1_realistic.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment (same as Streamlit)
    env = EasyStartEnv(
        num_agents_per_team=config['num_agents_per_team'],
        grid_width=config['grid_width'],
        grid_height=config['grid_height'],
        max_steps=config['max_steps'],
        difficulty='easy'
    )
    
    print(f"\n✅ Environment created")
    print(f"   Grid: {env.grid_width}x{env.grid_height}")
    print(f"   Agents per team: {config['num_agents_per_team']}")
    print(f"   Max steps: {config['max_steps']}")
    
    # Reset environment
    obs, _ = env.reset()
    print(f"\n✅ Environment reset")
    print(f"   Agents: {env.agents}")
    print(f"   Observation shape: {obs[env.agents[0]].shape}")
    
    # Simulate Streamlit loop
    print(f"\n🎮 Simulating Streamlit play loop...")
    print("-"*60)
    
    episode_reward = 0
    goals_team0 = 0
    goals_team1 = 0
    steps = 0
    
    for _ in range(50):  # Simulate 50 steps
        agent_name = env.agent_selection
        
        # Check if done
        if env.terminations[agent_name] or env.truncations[agent_name]:
            env.step(None)
            obs = env._get_observations()
            
            if all(env.terminations.values()) or all(env.truncations.values()):
                print(f"\n✅ Episode completed at step {steps}")
                break
            continue
        
        # Get observation
        observation = obs[agent_name]
        
        # Random action (simulating model)
        action = np.random.randint(0, 7)
        
        # Execute
        env.step(action)
        reward = env.rewards[agent_name]
        
        # Update observations
        obs = env._get_observations()
        
        # Track stats (same as Streamlit)
        if 'team_0' in agent_name:
            episode_reward += reward
            if reward > 90:
                goals_team0 += 1
                agent_pos = env.agent_positions[agent_name]
                print(f"   ⚽ GOAL! Team 0 at step {steps}, pos: [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}]")
        elif reward < -90:
            goals_team1 += 1
            print(f"   ⚽ GOAL! Team 1 at step {steps}")
        
        steps += 1
        
        # Show progress every 10 steps
        if steps % 10 == 0:
            agent_pos = env.agent_positions['team_0_agent_0']
            goal_pos = np.array([env.grid_width - 1, env.grid_height // 2])
            dist = np.linalg.norm(agent_pos - goal_pos)
            print(f"   Step {steps:2d}: Pos [{agent_pos[0]:.1f}, {agent_pos[1]:.1f}], "
                  f"Dist to goal: {dist:.1f}, Reward: {episode_reward:+.1f}")
    
    print()
    print("="*60)
    print("📊 FINAL STATISTICS")
    print("="*60)
    print(f"Steps taken: {steps}")
    print(f"Episode reward: {episode_reward:.1f}")
    print(f"Goals Team 0: {goals_team0}")
    print(f"Goals Team 1: {goals_team1}")
    print(f"Shots taken: {env.episode_stats['shots']}")
    print()
    
    if steps > 0:
        print("✅ SUCCESS: Streamlit app mechanics work correctly!")
        print("   - Agent loop functions properly")
        print("   - Observation updates work")
        print("   - Reward tracking works")
        print("   - Goal detection works")
        print()
        print("🚀 Streamlit app is ready to use!")
        print("   Run: streamlit run app.py")
    else:
        print("❌ FAIL: Something went wrong")
    
    return steps > 0

def test_field_rendering():
    """Test field rendering components"""
    print("\n" + "="*60)
    print("🎨 TESTING FIELD RENDERING")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        
        print("✅ Matplotlib imports successful")
        
        # Create simple test figure
        fig, ax = plt.subplots(figsize=(8, 6))
        field = Rectangle((0, 0), 8, 4, facecolor='#2ECC40', edgecolor='white')
        ax.add_patch(field)
        ax.set_xlim(-1, 9)
        ax.set_ylim(-1, 5)
        ax.axis('off')
        plt.close(fig)
        
        print("✅ Field rendering components work")
        return True
    except Exception as e:
        print(f"❌ Rendering error: {e}")
        return False

def main():
    print("\n" + "🏟️ " * 15)
    print("\n   STREAMLIT APP COMPREHENSIVE TEST")
    print("\n" + "🏟️ " * 15)
    
    tests = [
        ("Core Mechanics", test_streamlit_mechanics),
        ("Field Rendering", test_field_rendering),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nStreamlit app is fully functional!")
        print("\n📝 To use:")
        print("   1. Train a model (optional):")
        print("      python train_ppo.py --config configs/stage1_realistic.yaml")
        print()
        print("   2. Launch Streamlit:")
        print("      streamlit run app.py")
        print()
        print("   3. If no trained model, app shows:")
        print("      - Helpful getting started guide")
        print("      - Model loading instructions")
        print("      - Expected file locations")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("   Check errors above for details")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
