"""
Main Training Script for Multi-Agent Football PPO
Implements curriculum learning and comprehensive logging
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import yaml
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.football_env import FootballEnv
from models.ppo_agent import PPOAgent
from training.buffer import MultiAgentBuffer
from training.utils import LinearSchedule, EpisodeStats


class CurriculumManager:
    """Manages curriculum learning progression"""
    
    def __init__(self, stages, threshold_win_rate=0.6, threshold_episodes=200):
        self.stages = stages  # List of dicts with stage configs
        self.current_stage = 0
        self.threshold_win_rate = threshold_win_rate
        self.threshold_episodes = threshold_episodes
        
        self.stage_episodes = 0
        self.stage_wins = 0
        self.recent_wins = []
    
    def should_advance(self):
        """Check if we should move to next curriculum stage"""
        if self.current_stage >= len(self.stages) - 1:
            return False
        
        if len(self.recent_wins) < self.threshold_episodes:
            return False
        
        win_rate = np.mean(self.recent_wins[-self.threshold_episodes:])
        return win_rate > self.threshold_win_rate
    
    def advance_stage(self):
        """Move to next curriculum stage"""
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            self.stage_episodes = 0
            self.stage_wins = 0
            self.recent_wins = []
            return True
        return False
    
    def update(self, won):
        """Update curriculum stats after episode"""
        self.stage_episodes += 1
        if won:
            self.stage_wins += 1
        self.recent_wins.append(1 if won else 0)
    
    def get_current_config(self):
        """Get current stage configuration"""
        return self.stages[self.current_stage]
    
    def get_stage_name(self):
        """Get current stage name"""
        return self.stages[self.current_stage].get('name', f'Stage {self.current_stage}')


class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        
        # Set device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config['use_gpu'] else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Create environment
        if config.get('use_improved_env', False):
            from env.improved_football_env import ImprovedFootballEnv
            self.env = ImprovedFootballEnv(
                num_agents_per_team=config['num_agents_per_team'],
                grid_width=config['grid_width'],
                grid_height=config['grid_height'],
                max_steps=config['max_steps'],
                movement_speed=config.get('movement_speed', 0.5),
                shoot_range=config.get('shoot_range', 3.0),
                pass_range=config.get('pass_range', 4.0),
                debug=False
            )
        else:
            self.env = FootballEnv(
                num_agents_per_team=config['num_agents_per_team'],
                grid_width=config['grid_width'],
                grid_height=config['grid_height'],
                max_steps=config['max_steps']
            )
        
        # Get observation and action dimensions
        agent = self.env.agents[0]
        self.obs_dim = self.env.observation_space(agent).shape[0]
        self.action_dim = self.env.action_space(agent).n
        
        # Create agents (shared parameters per team)
        self.team_0_agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            **config['ppo_params']
        )
        
        self.team_1_agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
            **config['ppo_params']
        )
        
        # Experience buffers
        self.team_0_buffer = MultiAgentBuffer(
            num_agents=config['num_agents_per_team'],
            buffer_size=config['buffer_size'],
            obs_dim=self.obs_dim,
            gamma=config['ppo_params']['gamma'],
            gae_lambda=config['ppo_params']['gae_lambda']
        )
        
        self.team_1_buffer = MultiAgentBuffer(
            num_agents=config['num_agents_per_team'],
            buffer_size=config['buffer_size'],
            obs_dim=self.obs_dim,
            gamma=config['ppo_params']['gamma'],
            gae_lambda=config['ppo_params']['gae_lambda']
        )
        
        # Curriculum
        if config.get('curriculum'):
            self.curriculum = CurriculumManager(
                stages=config['curriculum_stages'],
                threshold_win_rate=config['curriculum_threshold_win_rate'],
                threshold_episodes=config['curriculum_threshold_episodes']
            )
        else:
            self.curriculum = None
        
        # Entropy scheduling
        if config.get('entropy_decay'):
            self.entropy_schedule = LinearSchedule(
                start_value=config['ppo_params']['entropy_coef'],
                end_value=config['entropy_decay_target'],
                duration=config['entropy_decay_episodes']
            )
        else:
            self.entropy_schedule = None
        
        # Logging
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_dir = os.path.join(config['log_dir'], f'run_{timestamp}')
        self.writer = SummaryWriter(self.log_dir)
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Stats tracking
        self.episode_stats = EpisodeStats()
        self.best_win_rate = 0.0
        self.global_step = 0
    
    def collect_episode(self):
        """Collect one episode of experience"""
        observations, infos = self.env.reset()
        done = False
        episode_rewards = defaultdict(float)
        episode_steps = 0
        
        # Episode-level tracking
        passes_attempted = 0
        passes_successful = 0
        
        while not done:
            actions = {}
            
            # Get actions for all agents
            for agent in self.env.agents:
                obs = observations[agent]
                team_id = 0 if 'team_0' in agent else 1
                agent_obj = self.team_0_agent if team_id == 0 else self.team_1_agent
                
                action, log_prob, value, entropy = agent_obj.get_action(obs)
                actions[agent] = action
                
                # Store in buffer
                agent_idx = int(agent.split('_')[-1])
                buffer = self.team_0_buffer if team_id == 0 else self.team_1_buffer
                buffer.add(agent_idx, obs, action, 0, value, log_prob, False)
            
            # Step environment
            for agent in self.env.agent_iter():
                obs_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = self.env.step(
                    actions[agent]
                )
                
                # Extract this agent's data
                observation = obs_dict.get(agent, observations[agent])
                reward = rewards_dict.get(agent, 0)
                termination = terminations_dict.get(agent, False)
                truncation = truncations_dict.get(agent, False)
                info = infos_dict.get(agent, {})
                
                episode_rewards[agent] += reward
                observations[agent] = observation
                
                # Update buffer with actual reward
                team_id = 0 if 'team_0' in agent else 1
                agent_idx = int(agent.split('_')[-1])
                buffer = self.team_0_buffer if team_id == 0 else self.team_1_buffer
                
                # Update last transition reward
                buffer.buffers[agent_idx].rewards[buffer.buffers[agent_idx].ptr - 1] = reward
                
                if termination or truncation:
                    buffer.buffers[agent_idx].dones[buffer.buffers[agent_idx].ptr - 1] = 1
                    done = True
                
                if done:
                    break
            
            episode_steps += 1
            self.global_step += 1
        
        # Compute returns and advantages
        last_values_0 = [0] * self.config['num_agents_per_team']
        last_values_1 = [0] * self.config['num_agents_per_team']
        
        self.team_0_buffer.compute_returns_and_advantages(last_values_0)
        self.team_1_buffer.compute_returns_and_advantages(last_values_1)
        
        # Compute episode statistics
        team_0_reward = sum(episode_rewards[a] for a in self.env.agents if 'team_0' in a)
        team_1_reward = sum(episode_rewards[a] for a in self.env.agents if 'team_1' in a)
        
        stats = self.env.episode_stats
        
        return {
            'team_0_reward': team_0_reward,
            'team_1_reward': team_1_reward,
            'team_0_goals': stats.get('goals_team_0', 0),
            'team_1_goals': stats.get('goals_team_1', 0),
            'passes': stats.get('passes', 0),
            'successful_passes': stats.get('successful_passes', 0),
            'shots': stats.get('shots', stats.get('shots_team_0', 0) + stats.get('shots_team_1', 0)),
            'episode_steps': episode_steps
        }
    
    def update_agents(self, episode):
        """Update both teams' policies"""
        # Update entropy coefficient if scheduled
        if self.entropy_schedule:
            new_entropy_coef = self.entropy_schedule.value(episode)
            self.team_0_agent.entropy_coef = new_entropy_coef
            self.team_1_agent.entropy_coef = new_entropy_coef
        else:
            # Use agent's built-in entropy decay if configured
            self.team_0_agent.update_entropy_coefficient()
            self.team_1_agent.update_entropy_coefficient()
        
        # Update team 0
        buffer_0 = self.team_0_buffer.get_all_training_data()
        from training.buffer import DummyBuffer
        dummy_buffer_0 = DummyBuffer(*buffer_0)
        stats_0 = self.team_0_agent.update(dummy_buffer_0)
        
        # Update team 1
        buffer_1 = self.team_1_buffer.get_all_training_data()
        dummy_buffer_1 = DummyBuffer(*buffer_1)
        stats_1 = self.team_1_agent.update(dummy_buffer_1)
        
        # Clear buffers
        self.team_0_buffer.clear()
        self.team_1_buffer.clear()
        
        return stats_0, stats_1
    
    def train(self):
        """Main training loop"""
        num_episodes = self.config['num_episodes']
        update_interval = self.config['update_interval']
        
        print(f"Starting training for {num_episodes} episodes")
        print(f"Update interval: {update_interval} episodes")
        
        episode_buffer = []
        
        for episode in range(num_episodes):
            # Collect episode
            episode_data = self.collect_episode()
            episode_buffer.append(episode_data)
            
            # Update curriculum
            if self.curriculum:
                team_0_won = episode_data['team_0_goals'] > episode_data['team_1_goals']
                self.curriculum.update(team_0_won)
                
                if self.curriculum.should_advance():
                    self.curriculum.advance_stage()
                    print(f"\n{'='*60}")
                    print(f"CURRICULUM ADVANCED: {self.curriculum.get_stage_name()}")
                    print(f"{'='*60}\n")
                    
                    # Update environment parameters based on new stage
                    stage_config = self.curriculum.get_current_config()
                    if 'num_agents_per_team' in stage_config:
                        # Would need to recreate environment here
                        pass
            
            # Perform update
            if (episode + 1) % update_interval == 0:
                stats_0, stats_1 = self.update_agents(episode)
                
                # Log training stats
                self.writer.add_scalar('Train/policy_loss_team_0', 
                                      stats_0['policy_loss'], episode)
                self.writer.add_scalar('Train/value_loss_team_0',
                                      stats_0['value_loss'], episode)
                self.writer.add_scalar('Train/entropy_team_0',
                                      stats_0['entropy'], episode)
                
                # Compute episode statistics
                avg_team_0_reward = np.mean([e['team_0_reward'] for e in episode_buffer])
                avg_team_1_reward = np.mean([e['team_1_reward'] for e in episode_buffer])
                avg_passes = np.mean([e['passes'] for e in episode_buffer])
                avg_successful_passes = np.mean([e['successful_passes'] for e in episode_buffer])
                pass_success_rate = avg_successful_passes / (avg_passes + 1e-6)
                
                # Log episode stats
                self.writer.add_scalar('Episode/team_0_reward', avg_team_0_reward, episode)
                self.writer.add_scalar('Episode/team_1_reward', avg_team_1_reward, episode)
                self.writer.add_scalar('Episode/pass_success_rate', pass_success_rate, episode)
                self.writer.add_scalar('Episode/avg_passes', avg_passes, episode)
                
                # Win rate
                wins = sum(1 for e in episode_buffer 
                          if e['team_0_goals'] > e['team_1_goals'])
                win_rate = wins / len(episode_buffer)
                self.writer.add_scalar('Episode/win_rate', win_rate, episode)
                
                # Update best model
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    self.save_checkpoint(episode, 'best_model.pt')
                
                # Print progress
                if (episode + 1) % (update_interval * 5) == 0:
                    print(f"Episode {episode + 1}/{num_episodes}")
                    print(f"  Team 0 Reward: {avg_team_0_reward:.2f}")
                    print(f"  Win Rate: {win_rate:.2%}")
                    print(f"  Pass Success: {pass_success_rate:.2%}")
                    print(f"  Policy Loss: {stats_0['policy_loss']:.4f}")
                    if self.curriculum:
                        print(f"  Curriculum Stage: {self.curriculum.get_stage_name()}")
                
                episode_buffer = []
            
            # Periodic checkpoint
            if (episode + 1) % self.config['checkpoint_interval'] == 0:
                self.save_checkpoint(episode, f'episode_{episode+1}.pt')
        
        # Final save
        self.save_checkpoint(num_episodes, 'final_model.pt')
        self.writer.close()
        print("\nTraining complete!")
    
    def save_checkpoint(self, episode, filename):
        """Save model checkpoint"""
        path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'episode': episode,
            'team_0_agent': {
                'actor': self.team_0_agent.actor.state_dict(),
                'critic': self.team_0_agent.critic.state_dict(),
            },
            'team_1_agent': {
                'actor': self.team_1_agent.actor.state_dict(),
                'critic': self.team_1_agent.critic.state_dict(),
            },
            'config': self.config,
            'best_win_rate': self.best_win_rate
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default_config.yaml')
    parser.add_argument('--curriculum', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with CLI args
    if args.curriculum:
        config['curriculum'] = True
    
    # Create trainer
    trainer = Trainer(config)
    
    # Load checkpoint if specified
    if args.checkpoint:
        # Implementation for loading checkpoint
        pass
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()