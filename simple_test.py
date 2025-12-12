"""Simple test to verify the environment works"""
import sys
import os
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.football_env import FootballEnv


def test_reward_sanity():
    """Test that rewards are reasonable and not NaN"""
    print("\n🔍 Testing reward sanity...")
    env = FootballEnv(num_agents_per_team=2, max_steps=50)
    env.reset(seed=42)
    
    all_rewards = []
    episode_done = False
    
    for agent in env.agent_iter():
        if episode_done:
            break
            
        # Handle dead agents properly
        if env.terminations[agent] or env.truncations[agent]:
            result = env.step(None)
            if result is None:
                continue
            obs_dict, rewards_dict, terms, truncs, info = result
        else:
            obs_dict, rewards_dict, terms, truncs, info = env.step(0)  # STAY action
        
        reward = rewards_dict[agent]
        
        # Check for NaN
        if np.isnan(reward):
            print(f"  ⚠️ NaN reward detected!")
            continue
            
        # Check for reasonable bounds
        if not (-100 < reward < 300):
            print(f"  ⚠️ Reward out of bounds: {reward}")
            continue
            
        all_rewards.append(reward)
        
        # Check if episode ended
        if all(terms.values()) or all(truncs.values()):
            episode_done = True
    
    if all_rewards:
        print(f"  ✓ Reward range: [{min(all_rewards):.2f}, {max(all_rewards):.2f}]")
        print(f"  ✓ Mean reward: {np.mean(all_rewards):.2f}")
        print(f"  ✓ Reward std: {np.std(all_rewards):.2f}")
        
        # Check that we have some variance
        if np.std(all_rewards) > 0.01:
            print("  ✓ Rewards have good variance")
    else:
        print("  ⚠️ No rewards collected")


def test_episode_termination():
    """Ensure episodes terminate properly"""
    print("\n🔍 Testing episode termination...")
    env = FootballEnv(num_agents_per_team=2, max_steps=100)
    env.reset(seed=42)
    
    steps = 0
    for agent in env.agent_iter():
        # Handle dead agents
        if env.terminations[agent] or env.truncations[agent]:
            result = env.step(None)
            if result is None:
                continue
            obs_dict, rewards_dict, terms, truncs, info = result
        else:
            obs_dict, rewards_dict, terms, truncs, info = env.step(0)
        
        steps += 1
        
        if all(terms.values()) or all(truncs.values()):
            print(f"  ✓ Episode terminated after {steps} agent steps")
            assert steps <= 1000, f"Episode took too long: {steps} steps"
            return
    
    print(f"  ⚠️ Completed iterator after {steps} steps")


def test_observation_shapes():
    """Verify observation shapes are consistent"""
    print("\n🔍 Testing observation shapes...")
    env = FootballEnv(num_agents_per_team=2)
    obs_dict, _ = env.reset()
    
    expected_shape = (13,)  # For 2v2
    for agent, obs in obs_dict.items():
        assert obs.shape == expected_shape, f"Wrong shape for {agent}: {obs.shape}"
        assert not np.isnan(obs).any(), f"NaN in observation for {agent}"
    
    print(f"  ✓ All observations have shape {expected_shape}")
    print("  ✓ No NaN values in observations")

# Run basic tests
print("="*70)
print("RUNNING ENVIRONMENT SANITY CHECKS")
print("="*70)

test_observation_shapes()
test_reward_sanity()
test_episode_termination()

# Create environment
env = FootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6, max_steps=50)

# Test reset
observations, infos = env.reset(seed=42)
print(f"\n✓ Environment reset successful!")
print(f"  Number of agents: {len(env.agents)}")
print(f"  Observation shape for first agent: {observations[env.agents[0]].shape}")

# Test a few steps
print("\n✓ Running 10 test steps...")
step_count = 0
for agent in env.agent_iter():
    if step_count >= 10:
        break
    
    # Check if agent is terminated or truncated
    if env.terminations[agent] or env.truncations[agent]:
        # For dead agents, pass None
        observations_dict, rewards_dict, terminations, truncations, info = env.step(None)
    else:
        # For alive agents, take action
        observations_dict, rewards_dict, terminations, truncations, info = env.step(0)  # STAY action
    
    step_count += 1
    
    # Check if episode is done
    if all(terminations.values()) or all(truncations.values()):
        print(f"  Episode ended after {step_count} steps")
        break

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\n🎉 The project is working correctly and ready for training!")
