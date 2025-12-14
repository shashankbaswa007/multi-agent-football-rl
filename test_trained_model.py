#!/usr/bin/env python3
"""
Test the trained Stage 1 model and visualize agent behavior
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from easy_start_env import EasyStartEnv
from env.football_env import FootballEnv
from models.ppo_agent import PPOAgent
import time

def load_model(checkpoint_path, config_path):
    """Load trained model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    print(f"Using config: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment to get observation/action spaces
    env = EasyStartEnv(
        num_agents_per_team=config['num_agents_per_team'],
        grid_width=config['grid_width'],
        grid_height=config['grid_height'],
        max_steps=config['max_steps'],
        difficulty='easy'
    )
    
    obs, _ = env.reset()
    agent_name = env.agents[0]
    obs_dim = obs[agent_name].shape[0]
    action_dim = env.action_space(agent_name).n
    
    print(f"Observation dim: {obs_dim}, Action dim: {action_dim}")
    
    # Create agent
    ppo_params = config['ppo_params']
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=ppo_params['lr'],
        gamma=ppo_params['gamma'],
        clip_epsilon=ppo_params['clip_epsilon'],
        value_loss_coef=ppo_params['value_loss_coef'],
        entropy_coef=ppo_params['entropy_coef']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # The checkpoint stores agent states per team
    team_agent_state = checkpoint['team_0_agent']
    agent.actor.load_state_dict(team_agent_state['actor'])
    agent.critic.load_state_dict(team_agent_state['critic'])
    agent.actor.eval()
    agent.critic.eval()
    
    print(f"✅ Model loaded from episode {checkpoint['episode']}!")
    if 'best_win_rate' in checkpoint:
        print(f"   Best win rate: {checkpoint['best_win_rate']:.1f}%")
    return agent, env, config

def run_episode(agent, env, render=True, delay=0.5):
    """Run a single episode with the trained agent"""
    obs, info = env.reset()
    episode_reward = 0
    steps = 0
    
    print("\n" + "="*80)
    print("🏃 STARTING EPISODE")
    print("="*80)
    
    for agent_name in env.agent_iter(max_iter=env.max_steps * len(env.agents)):
        steps += 1
        
        # Get observation and action
        observation = obs[agent_name] if agent_name in obs else env.observe(agent_name)
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
        
        # Check if episode is done
        if env.terminations[agent_name] or env.truncations[agent_name]:
            env.step(None)
            continue
        
        # Get action from trained policy
        with torch.no_grad():
            action_probs = agent.actor(obs_tensor)
            action = torch.argmax(action_probs, dim=1).item()
        
        # Execute action
        env.step(action)
        
        # Get reward
        reward = env.rewards[agent_name]
        if 'team_0' in agent_name:
            episode_reward += reward
        
        # Render current state
        if render and 'team_0' in agent_name:  # Only render for team 0 turns
            print(f"\n{'─'*80}")
            print(f"Step {steps // len(env.agents)}/{env.max_steps}")
            print(f"{'─'*80}")
            
            # Show agent info
            agent_pos = env.agent_positions[agent_name]
            print(f"Agent: {agent_name}")
            print(f"Position: [{agent_pos[0]:.2f}, {agent_pos[1]:.2f}]")
            print(f"Action: {['STAY', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'SHOOT', 'PASS'][action]}")
            print(f"Reward: {reward:.2f}")
            
            # Show ball info
            if env.ball_possession:
                print(f"Ball: {env.ball_possession} has possession")
            else:
                ball_pos = env.ball_position
                print(f"Ball: Free at [{ball_pos[0]:.2f}, {ball_pos[1]:.2f}]")
            
            # Show goal proximity
            goal_pos = np.array([env.grid_width, env.grid_height / 2])
            dist_to_goal = np.linalg.norm(agent_pos - goal_pos)
            print(f"Distance to goal: {dist_to_goal:.2f} units")
            print(f"Team 0 Total Reward: {episode_reward:.2f}")
            
            # Show grid visualization
            print("\n" + render_grid(env))
            
            time.sleep(delay)
        
        # Check if episode ended
        if all(env.terminations.values()) or all(env.truncations.values()):
            break
    
    # Episode summary
    print("\n" + "="*80)
    print("📊 EPISODE SUMMARY")
    print("="*80)
    print(f"Steps: {steps}")
    print(f"Team 0 Total Reward: {episode_reward:.2f}")
    
    # Determine outcome
    if episode_reward > 180:
        print("🎉 GOAL SCORED! ✅")
    elif episode_reward > 50:
        print("🎯 Good progress (close to goal)")
    elif episode_reward > 0:
        print("⚠️ Some progress made")
    else:
        print("❌ Poor performance")
    
    return episode_reward, steps

def render_grid(env):
    """Simple ASCII visualization of the field"""
    grid = [[' ' for _ in range(env.grid_width + 1)] for _ in range(env.grid_height + 1)]
    
    # Draw agents
    for agent_name in env.agents:
        pos = env.agent_positions[agent_name]
        x, y = int(pos[0]), int(pos[1])
        if 0 <= x <= env.grid_width and 0 <= y <= env.grid_height:
            if 'team_0' in agent_name:
                grid[y][x] = '🔵'  # Blue team
            else:
                grid[y][x] = '🔴'  # Red team
    
    # Draw ball
    if env.ball_possession is None:
        ball_pos = env.ball_position
        x, y = int(ball_pos[0]), int(ball_pos[1])
        if 0 <= x <= env.grid_width and 0 <= y <= env.grid_height:
            grid[y][x] = '⚽'
    
    # Draw goals
    goal_y = int(env.grid_height / 2)
    grid[goal_y][0] = '🥅'  # Left goal
    grid[goal_y][env.grid_width] = '🥅'  # Right goal
    
    # Convert to string
    result = "┌" + "─" * (env.grid_width * 2 + 1) + "┐\n"
    for row in grid:
        result += "│" + "".join(f"{cell:2}" for cell in row) + "│\n"
    result += "└" + "─" * (env.grid_width * 2 + 1) + "┘"
    
    return result

def main():
    """Main testing function"""
    print("="*80)
    print("🧪 TESTING TRAINED MODEL")
    print("="*80)
    
    # Find latest checkpoint
    runs_dir = Path("runs")
    if not runs_dir.exists():
        print("❌ No runs directory found!")
        return
    
    # Look for most recent run
    run_dirs = sorted(runs_dir.glob("run_*"))
    if not run_dirs:
        print("❌ No training runs found!")
        return
    
    latest_run = run_dirs[-1]
    print(f"Using latest run: {latest_run}")
    
    # Try to find best or final model
    checkpoint_dir = latest_run / "checkpoints"
    if (checkpoint_dir / "best_model.pt").exists():
        checkpoint_path = checkpoint_dir / "best_model.pt"
        print("Using best model checkpoint")
    elif (checkpoint_dir / "final_model.pt").exists():
        checkpoint_path = checkpoint_dir / "final_model.pt"
        print("Using final model checkpoint")
    else:
        # Find latest episode checkpoint
        episode_checkpoints = sorted(checkpoint_dir.glob("episode_*.pt"))
        if episode_checkpoints:
            checkpoint_path = episode_checkpoints[-1]
            print(f"Using checkpoint: {checkpoint_path.name}")
        else:
            print("❌ No checkpoints found!")
            return
    
    # Load model
    config_path = "configs/stage1_stable.yaml"
    agent, env, config = load_model(checkpoint_path, config_path)
    
    # Run test episodes
    print("\n" + "="*80)
    print("🎮 RUNNING TEST EPISODES")
    print("="*80)
    print("Watch the agents play! They should move toward the goal and shoot.")
    print()
    
    num_episodes = 5
    total_rewards = []
    
    for i in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {i+1}/{num_episodes}")
        print(f"{'='*80}")
        
        reward, steps = run_episode(agent, env, render=True, delay=0.3)
        total_rewards.append(reward)
        
        print(f"\nPress Enter to continue to next episode...")
        input()
    
    # Final statistics
    print("\n" + "="*80)
    print("📈 FINAL STATISTICS")
    print("="*80)
    print(f"Total episodes: {num_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f}")
    print(f"Best reward: {np.max(total_rewards):.2f}")
    print(f"Worst reward: {np.min(total_rewards):.2f}")
    print(f"Goals scored: {sum(1 for r in total_rewards if r > 180)}/{num_episodes}")
    print(f"Win rate: {100 * sum(1 for r in total_rewards if r > 180) / num_episodes:.1f}%")
    
    if np.mean(total_rewards) > 180:
        print("\n🎉 EXCELLENT! Model is scoring consistently!")
    elif np.mean(total_rewards) > 50:
        print("\n✅ GOOD! Model is making progress toward goal")
    else:
        print("\n⚠️ Model needs more training")

if __name__ == "__main__":
    main()
