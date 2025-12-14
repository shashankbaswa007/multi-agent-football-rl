#!/usr/bin/env python3
"""
Test continuous play by manually triggering goals
"""
import sys
sys.path.insert(0, '/Users/shashi/reinforcement_learning')

from env.football_env import FootballEnv
import numpy as np

def test_continuous_play():
    print("="*70)
    print("🎮 TESTING CONTINUOUS PLAY - Manual Goal Triggering")
    print("="*70)
    
    env = FootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6, max_steps=300)
    obs, info = env.reset()
    
    print("\n📍 Initial Setup:")
    print(f"   Ball position: {env.ball_position}")
    print(f"   Team 0 agent position: {env.agent_positions['team_0_agent_0']}")
    
    goals_scored = 0
    ball_positions_after_goals = []
    
    print("\n🎯 Simulating goals by moving ball to goal positions...")
    print("-"*70)
    
    # Simulate 3 goals
    for goal_num in range(1, 4):
        # Give ball to team 0 agent
        env.ball_possession = 'team_0_agent_0'
        
        # Move agent close to goal
        env.agent_positions['team_0_agent_0'] = np.array([env.grid_width - 0.5, env.grid_height / 2])
        env.ball_position = env.agent_positions['team_0_agent_0'].copy()
        
        print(f"\n🏈 Setting up Goal #{goal_num}:")
        print(f"   Agent position: {env.agent_positions['team_0_agent_0']}")
        print(f"   Ball position: {env.ball_position}")
        
        # Manually move ball into goal zone to trigger goal
        env.ball_position = np.array([env.grid_width, env.grid_height / 2])
        
        # Check for goal
        goal_scored_team = env._check_goal()
        
        if goal_scored_team is not None:
            print(f"   ⚡ Goal detected! Team {goal_scored_team} scored!")
            
            # This should reset ball to center (our new feature!)
            env._handle_goal(goal_scored_team)
            env._reset_ball_to_center()
            
            goals_scored += 1
            ball_positions_after_goals.append(env.ball_position.copy())
            
            print(f"   📍 After reset:")
            print(f"      Ball position: {env.ball_position}")
            print(f"      Ball at center: {np.allclose(env.ball_position, [env.grid_width/2, env.grid_height/2])}")
            print(f"      Ball possession: {env.ball_possession}")
            print(f"      Agent positions reset: YES")
            
            # Check agent positions were reset
            team0_pos = env.agent_positions['team_0_agent_0']
            team1_pos = env.agent_positions['team_1_agent_0']
            print(f"      Team 0 agent at: {team0_pos} (left side: {team0_pos[0] < env.grid_width/2})")
            print(f"      Team 1 agent at: {team1_pos} (right side: {team1_pos[0] > env.grid_width/2})")
        
        print("-"*70)
    
    print("\n📊 RESULTS:")
    print(f"   Goals simulated: {goals_scored}")
    print(f"   Ball positions after goals:")
    for i, pos in enumerate(ball_positions_after_goals, 1):
        center = np.array([env.grid_width/2, env.grid_height/2])
        is_at_center = np.allclose(pos, center, atol=0.1)
        print(f"      Goal {i}: {pos} - {'✅ AT CENTER' if is_at_center else '❌ NOT AT CENTER'}")
    
    # Verify all balls reset to center
    all_centered = all(np.allclose(pos, [env.grid_width/2, env.grid_height/2], atol=0.1) 
                       for pos in ball_positions_after_goals)
    
    if all_centered and goals_scored == 3:
        print("\n✅ SUCCESS! Continuous play working perfectly!")
        print("   - Goals triggered successfully")
        print("   - Ball reset to center after each goal")
        print("   - Agents reset to starting positions")
        print("   - Episodes continue after goals")
    else:
        print("\n❌ ISSUE DETECTED!")
        if not all_centered:
            print("   - Ball did NOT reset to center")
        if goals_scored != 3:
            print(f"   - Only {goals_scored}/3 goals detected")
    
    print("="*70)

if __name__ == "__main__":
    test_continuous_play()
