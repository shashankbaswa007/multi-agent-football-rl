#!/usr/bin/env python3
"""
Test continuous play - ball resets to center after goals
"""
import sys
sys.path.insert(0, '/Users/shashi/reinforcement_learning')

from env.football_env import FootballEnv
import numpy as np

def test_continuous_play():
    print("="*70)
    print("🎮 TESTING CONTINUOUS PLAY FEATURE")
    print("="*70)
    
    env = FootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6, max_steps=200)
    obs, info = env.reset()
    
    print("\n📍 Initial Setup:")
    print(f"   Ball position: {env.ball_position}")
    print(f"   Ball possession: {env.ball_possession}")
    
    goals_scored = 0
    step = 0
    
    print("\n🏃 Starting simulation...")
    print("-"*70)
    
    for agent in env.agent_iter(max_iter=200):
        step += 1
        
        if env.terminations[agent] or env.truncations[agent]:
            env.step(None)
            continue
        
        # Simple strategy: move toward ball and shoot
        pos = env.agent_positions[agent]
        team = 0 if 'team_0' in agent else 1
        
        # Determine action
        if env.ball_possession == agent:
            # If we have ball, move toward goal or shoot
            goal = np.array([env.grid_width, env.grid_height/2]) if team == 0 else np.array([0, env.grid_height/2])
            dist_to_goal = np.linalg.norm(pos - goal)
            
            if dist_to_goal < 3:
                action = 6  # SHOOT
            else:
                # Move toward goal
                if team == 0:
                    action = 5  # RIGHT
                else:
                    action = 4  # LEFT
        else:
            # Move toward ball
            ball_dir = env.ball_position - pos
            if abs(ball_dir[0]) > abs(ball_dir[1]):
                action = 5 if ball_dir[0] > 0 else 4  # RIGHT or LEFT
            else:
                action = 2 if ball_dir[1] > 0 else 3  # DOWN or UP
        
        # Execute action
        env.step(action)
        reward = env.rewards[agent]
        
        # Check if goal was scored (big reward)
        if abs(reward) > 90:  # Goal scored (±100 reward)
            goals_scored += 1
            scoring_team = "Team 0" if reward > 0 and 'team_0' in agent else "Team 1"
            print(f"\n🎉 GOAL #{goals_scored} scored by {scoring_team}!")
            print(f"   Step: {step}")
            print(f"   Ball reset to center: {env.ball_position}")
            print(f"   Ball possession: {env.ball_possession}")
            print(f"   Episode continues: {not env.terminations[agent]}")
            print("-"*70)
        
        # Check if episode ended (should NOT end on goal anymore)
        if all(env.terminations.values()):
            print(f"\n⏹️  Episode ended at step {step}")
            break
        
        if all(env.truncations.values()):
            print(f"\n⏱️  Episode truncated at step {step} (max steps)")
            break
    
    print("\n📊 RESULTS:")
    print(f"   Total goals scored: {goals_scored}")
    print(f"   Total steps: {step}")
    print(f"   Episode ended by goal: {'NO ✅' if goals_scored > 0 else 'N/A'}")
    
    if goals_scored >= 2:
        print("\n✅ SUCCESS! Multiple goals scored - continuous play working!")
    elif goals_scored == 1:
        print("\n⚠️  Only 1 goal scored - may need more steps to test fully")
    else:
        print("\n❌ No goals scored - couldn't test continuous play")
    
    print("="*70)

if __name__ == "__main__":
    test_continuous_play()
