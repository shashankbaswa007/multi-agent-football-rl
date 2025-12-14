#!/usr/bin/env python3
"""Test that agents receive appropriate rewards for goal-directed behavior"""

import sys
sys.path.insert(0, '/Users/shashi/reinforcement_learning')

from env.football_env import FootballEnv
import numpy as np

def test_reward_structure():
    print("=" * 70)
    print("🧪 TESTING REWARD STRUCTURE")
    print("=" * 70)
    
    env = FootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = 'team_0_agent_0'
    
    # Test 1: Reward for moving toward goal with ball
    print("\n📊 Test 1: Moving toward goal with ball")
    env.agent_positions[agent] = np.array([2.0, 3.0])
    env.ball_possession = agent
    env.ball_position = env.agent_positions[agent].copy()
    
    # Move right (toward team 0's goal)
    reward = env._execute_action(agent, 4)  # MOVE_RIGHT
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: Large positive (>5.0) for advancing with ball")
    
    # Test 2: Reward for gaining possession
    print("\n📊 Test 2: Gaining possession of loose ball")
    env.agent_positions[agent] = np.array([4.0, 3.0])
    env.ball_possession = None
    env.ball_position = np.array([4.5, 3.0])
    
    reward = env._execute_action(agent, 4)  # MOVE_RIGHT toward ball
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: ~10.0 for gaining possession")
    
    # Test 3: Reward for stealing from opponent
    print("\n📊 Test 3: Stealing ball from opponent")
    env.agent_positions[agent] = np.array([5.0, 3.0])
    env.agent_positions['team_1_agent_0'] = np.array([5.8, 3.0])
    env.ball_possession = 'team_1_agent_0'
    env.ball_position = env.agent_positions['team_1_agent_0'].copy()
    
    reward = env._execute_action(agent, 4)  # MOVE_RIGHT toward opponent
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: ~25.0 for stealing ball")
    
    # Test 4: Reward for shooting close to goal
    print("\n📊 Test 4: Shooting near goal")
    env.agent_positions[agent] = np.array([8.0, 3.0])  # Close to goal
    env.ball_possession = agent
    env.ball_position = env.agent_positions[agent].copy()
    
    reward = env._execute_action(agent, 6)  # SHOOT
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: >5.0 for close shot attempt, 200+ if scored")
    
    # Test 5: Penalty for staying still
    print("\n📊 Test 5: Staying still (do-nothing policy)")
    env.agent_positions[agent] = np.array([3.0, 3.0])
    env.ball_possession = agent
    env.ball_position = env.agent_positions[agent].copy()
    
    reward = env._execute_action(agent, 0)  # STAY
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: Negative (~-2.0) to discourage inaction")
    
    # Test 6: Chasing opponent with ball
    print("\n📊 Test 6: Chasing opponent who has ball")
    env.agent_positions[agent] = np.array([4.0, 3.0])
    env.agent_positions['team_1_agent_0'] = np.array([5.5, 3.0])
    env.ball_possession = 'team_1_agent_0'
    env.ball_position = env.agent_positions['team_1_agent_0'].copy()
    
    reward = env._execute_action(agent, 4)  # MOVE_RIGHT toward opponent
    print(f"   Reward: {reward:.2f}")
    print(f"   ✅ EXPECTED: Positive (>0.5) for chasing opponent")
    
    print("\n" + "=" * 70)
    print("✅ Reward structure test complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_reward_structure()
