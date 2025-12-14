#!/usr/bin/env python3
"""
Visual demonstration of realistic football mechanics
Shows step-by-step progression toward goal
"""

import numpy as np
from easy_start_env import EasyStartEnv
import time

def draw_field(env, agent='team_0_agent_0'):
    """Draw ASCII representation of field"""
    width = int(env.grid_width)
    height = int(env.grid_height)
    
    # Create field
    field = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Add goal markers
    goal_y = height // 2
    field[goal_y][0] = '🥅'
    field[goal_y][width-1] = '🥅'
    
    # Add center line
    for y in range(height):
        field[y][width//2] = '|'
    
    # Add agents
    for agent_name in env.agents:
        pos = env.agent_positions[agent_name]
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x < width and 0 <= y < height:
            if 'team_0' in agent_name:
                field[y][x] = '🔴'  # Team 0 (Red)
            else:
                field[y][x] = '🔵'  # Team 1 (Blue)
    
    # Add ball
    if env.ball_possession is None:
        ball_pos = env.ball_position
        x, y = int(ball_pos[0]), int(ball_pos[1])
        if 0 <= x < width and 0 <= y < height:
            field[y][x] = '⚽'
    
    # Print field (top to bottom)
    print("  " + "─" * width)
    for y in reversed(range(height)):
        print(" │" + "".join(field[y]) + "│")
    print("  " + "─" * width)
    print("  " + "".join([str(i) for i in range(width)]))

def demo_realistic_attack():
    """Demonstrate realistic attacking sequence"""
    print("\n" + "="*70)
    print("⚽ REALISTIC FOOTBALL ATTACK DEMONSTRATION")
    print("="*70)
    
    env = EasyStartEnv(num_agents_per_team=1, difficulty='easy', 
                       grid_width=10, grid_height=6, max_steps=200)
    obs, _ = env.reset()
    
    agent = 'team_0_agent_0'
    opponent = 'team_1_agent_0'
    
    print("\n📍 STARTING POSITION (Realistic: Midfield)")
    print("-" * 70)
    pos = env.agent_positions[agent]
    goal_pos = np.array([env.grid_width - 1, env.grid_height // 2])
    dist = np.linalg.norm(pos - goal_pos)
    print(f"Agent position: [{pos[0]:.1f}, {pos[1]:.1f}]")
    print(f"Goal position: [{goal_pos[0]:.1f}, {goal_pos[1]:.1f}]")
    print(f"Distance to goal: {dist:.1f} units")
    print(f"Has ball: {'✅ Yes' if env.ball_possession == agent else '❌ No'}")
    print()
    draw_field(env)
    
    print("\n🏃‍♂️ ATTACKING SEQUENCE - Watch the gradual progression!")
    print("-" * 70)
    
    step = 0
    total_reward = 0
    
    while dist > 1.5 and step < 10:
        step += 1
        
        # Decide action based on distance
        if dist > 4.0:
            action = 4  # MOVE_RIGHT (dribble toward goal)
            action_name = "DRIBBLE RIGHT"
        elif dist > 2.5:
            action = 4  # Continue dribbling
            action_name = "DRIBBLE CLOSER"
        else:
            action = 6  # SHOOT when close
            action_name = "SHOOT!"
        
        # Execute action
        reward = env._execute_action(agent, action)
        total_reward += reward
        
        # Update position
        new_pos = env.agent_positions[agent]
        new_dist = np.linalg.norm(new_pos - goal_pos)
        progress = dist - new_dist
        
        print(f"\n📍 Step {step}: {action_name}")
        print(f"   Position: [{new_pos[0]:.1f}, {new_pos[1]:.1f}]")
        print(f"   Distance to goal: {new_dist:.1f} units")
        print(f"   Progress: {progress:.2f} units forward")
        print(f"   Reward: {reward:+.1f} (Total: {total_reward:+.1f})")
        
        if action == 6:  # Shot
            if reward > 90:
                print("   🎉 GOAL SCORED! ⚽")
            else:
                print("   ❌ Shot missed (ball loose)")
        
        draw_field(env)
        time.sleep(0.5)
        
        dist = new_dist
        
        # Check if goal scored
        if env.episode_stats['goals_team_0'] > 0:
            break
    
    print("\n" + "="*70)
    print("📊 ATTACK SUMMARY")
    print("="*70)
    print(f"Steps taken: {step}")
    print(f"Total reward: {total_reward:+.1f}")
    print(f"Goals scored: {env.episode_stats['goals_team_0']}")
    print(f"Shots taken: {env.episode_stats['shots']}")
    print()
    
    if env.episode_stats['goals_team_0'] > 0:
        print("✅ SUCCESS: Realistic goal-scoring sequence!")
        print("   - Started at midfield (4-5 units away)")
        print("   - Dribbled forward step-by-step")
        print("   - Shot only when in range")
        print("   - Goal scored from close distance!")
    else:
        print("⚠️  Attack incomplete (shot from close range next time)")
    
    print("="*70)

def demo_shooting_limits():
    """Demonstrate realistic shooting distance limits"""
    print("\n" + "="*70)
    print("🎯 SHOOTING DISTANCE LIMITS DEMONSTRATION")
    print("="*70)
    
    env = EasyStartEnv(num_agents_per_team=1, difficulty='easy',
                       grid_width=10, grid_height=6)
    obs, _ = env.reset()
    
    agent = 'team_0_agent_0'
    goal_pos = np.array([env.grid_width - 1, env.grid_height // 2])
    
    test_distances = [
        (7.0, "Very far (impossible)"),
        (5.0, "Far (impossible)"),
        (3.5, "Medium range (low chance)"),
        (2.0, "Close range (good chance)"),
        (1.0, "Point-blank (high chance)")
    ]
    
    for dist, description in test_distances:
        # Position agent at specific distance
        env.agent_positions[agent] = goal_pos - np.array([dist, 0])
        env.ball_possession = agent
        
        actual_dist = np.linalg.norm(env.agent_positions[agent] - goal_pos)
        
        print(f"\n📍 Testing from {description}")
        print(f"   Distance: {actual_dist:.1f} units")
        
        # Try shooting
        reward = env._execute_action(agent, 6)  # SHOOT
        
        if reward < 0:
            print(f"   ❌ SHOT BLOCKED: Too far! (Reward: {reward:.1f})")
            print("   → Agent must dribble closer first")
        elif reward > 90:
            print(f"   🎉 GOAL! (Reward: {reward:.1f})")
        else:
            print(f"   ⚽ Shot attempted (Reward: {reward:.1f})")
            if env.episode_stats['goals_team_0'] > 0:
                print("   ✅ Goal scored!")
            else:
                print("   ❌ Shot missed")
        
        # Reset for next test
        env.episode_stats['goals_team_0'] = 0
    
    print("\n" + "="*70)
    print("Key Takeaway: Must be within 4 units to shoot (realistic!)")
    print("="*70)

def main():
    print("\n" + "🏟️ " * 15)
    print("\n      REALISTIC FOOTBALL MECHANICS - VISUAL DEMONSTRATION")
    print("\n" + "🏟️ " * 15)
    
    # Demo 1: Realistic attack
    demo_realistic_attack()
    
    input("\n\nPress Enter to see shooting limits demonstration...")
    
    # Demo 2: Shooting limits
    demo_shooting_limits()
    
    print("\n\n✅ DEMONSTRATIONS COMPLETE!")
    print("\nKey Points:")
    print("  1. Agents move ONE block at a time (no teleporting)")
    print("  2. Must dribble from midfield to goal (4-5 steps)")
    print("  3. Can only shoot within 4 units of goal")
    print("  4. Shot accuracy increases when closer (85% at point-blank)")
    print("  5. Realistic football progression!")
    print("\n🎮 Now load the trained model in Streamlit to see it in action!")
    print("   Run: streamlit run app.py")
    print()

if __name__ == "__main__":
    main()
