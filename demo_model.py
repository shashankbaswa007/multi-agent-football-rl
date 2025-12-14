#!/usr/bin/env python3
"""
Quick demo of trained model - simplified view
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from easy_start_env import EasyStartEnv
from models.ppo_agent import PPOAgent

def load_latest_model():
    """Load the most recent trained model"""
    runs_dir = Path("runs")
    run_dirs = sorted(runs_dir.glob("run_*"))
    latest_run = run_dirs[-1]
    
    checkpoint_dir = latest_run / "checkpoints"
    if (checkpoint_dir / "final_model.pt").exists():
        checkpoint_path = checkpoint_dir / "final_model.pt"
    elif (checkpoint_dir / "best_model.pt").exists():
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        episode_checkpoints = sorted(checkpoint_dir.glob("episode_*.pt"))
        checkpoint_path = episode_checkpoints[-1]
    
    print(f"Loading: {checkpoint_path.name}")
    
    # Load config
    with open("configs/stage1_stable.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Create environment
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
    
    # Create and load agent
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
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    team_agent_state = checkpoint['team_0_agent']
    agent.actor.load_state_dict(team_agent_state['actor'])
    agent.actor.eval()
    
    print(f"✅ Loaded from episode {checkpoint['episode']}")
    return agent, env

def run_demo():
    """Run quick demo"""
    print("="*60)
    print("🎮 TRAINED MODEL DEMO")
    print("="*60)
    
    agent, env = load_latest_model()
    
    num_episodes = 10
    goals_scored = 0
    total_reward = 0
    
    print(f"\nRunning {num_episodes} test episodes...")
    print("─"*60)
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        goals_this_ep = 0
        
        for agent_name in env.agent_iter(max_iter=env.max_steps * len(env.agents)):
            observation = obs[agent_name] if agent_name in obs else env.observe(agent_name)
            
            if env.terminations[agent_name] or env.truncations[agent_name]:
                env.step(None)
                continue
            
            # Get action
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                action_probs = agent.actor(obs_tensor)
                action = torch.argmax(action_probs, dim=1).item()
            
            env.step(action)
            reward = env.rewards[agent_name]
            
            if 'team_0' in agent_name:
                episode_reward += reward
                if reward > 99:  # Goal scored
                    goals_this_ep += 1
            
            if all(env.terminations.values()) or all(env.truncations.values()):
                break
        
        total_reward += episode_reward
        goals_scored += goals_this_ep
        
        status = "🎉 GOAL!" if goals_this_ep > 0 else "❌ Miss"
        print(f"Episode {ep+1:2d}: Reward={episode_reward:7.2f}, Goals={goals_this_ep}, {status}")
    
    # Final stats
    print("─"*60)
    print(f"\n📊 RESULTS:")
    print(f"   Episodes:  {num_episodes}")
    print(f"   Goals:     {goals_scored}/{num_episodes} ({100*goals_scored/num_episodes:.0f}%)")
    print(f"   Avg Reward: {total_reward/num_episodes:.2f}")
    print()
    
    if goals_scored >= num_episodes * 0.7:
        print("🏆 EXCELLENT! Model is scoring consistently!")
    elif goals_scored >= num_episodes * 0.4:
        print("✅ GOOD! Model is performing well")
    elif goals_scored > 0:
        print("⚠️  MODERATE - Some goals scored")
    else:
        print("❌ POOR - No goals scored")
    
    print("="*60)

if __name__ == "__main__":
    run_demo()
