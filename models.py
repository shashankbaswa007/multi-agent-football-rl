"""
Neural Network Models for Multi-Agent PPO
Includes Actor (policy) and Critic (value) networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np


def init_weights(module):
    """Orthogonal initialization for better training stability"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        nn.init.constant_(module.bias, 0.0)


class Actor(nn.Module):
    """
    Policy network that outputs action probabilities
    Uses shared parameters for all agents on the same team
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.action_head = nn.Linear(input_dim, action_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs, action_mask=None):
        """
        Forward pass
        
        Args:
            obs: (batch_size, obs_dim) observations
            action_mask: (batch_size, action_dim) binary mask for legal actions
            
        Returns:
            action_probs: (batch_size, action_dim) action probabilities
        """
        features = self.feature_extractor(obs)
        logits = self.action_head(features)
        
        # Apply action masking if provided
        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e8)
        
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def get_action(self, obs, action_mask=None, deterministic=False):
        """
        Sample action from policy
        
        Args:
            obs: (batch_size, obs_dim) or (obs_dim,) observations
            action_mask: optional action mask
            deterministic: if True, return argmax instead of sampling
            
        Returns:
            actions: sampled actions
            log_probs: log probabilities of actions
            entropy: policy entropy
        """
        # Handle single observation
        single_obs = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            single_obs = True
            if action_mask is not None:
                action_mask = action_mask.unsqueeze(0)
        
        action_probs = self.forward(obs, action_mask)
        dist = Categorical(action_probs)
        
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            actions = dist.sample()
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        if single_obs:
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)
            entropy = entropy.squeeze(0)
        
        return actions, log_probs, entropy
    
    def evaluate_actions(self, obs, actions, action_mask=None):
        """
        Evaluate log probs and entropy for given actions
        Used during PPO updates
        
        Args:
            obs: (batch_size, obs_dim)
            actions: (batch_size,)
            action_mask: optional mask
            
        Returns:
            log_probs: (batch_size,)
            entropy: (batch_size,)
        """
        action_probs = self.forward(obs, action_mask)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, entropy


class Critic(nn.Module):
    """
    Value network that estimates state value V(s)
    Larger network than actor for better value estimation
    """
    
    def __init__(self, obs_dim, hidden_dims=[512, 256]):
        super().__init__()
        
        layers = []
        input_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.Tanh())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.value_head = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs):
        """
        Forward pass
        
        Args:
            obs: (batch_size, obs_dim)
            
        Returns:
            values: (batch_size,) state values
        """
        features = self.feature_extractor(obs)
        values = self.value_head(features).squeeze(-1)
        return values


class PPOAgent:
    """
    Shared-parameter PPO agent for multi-agent coordination
    All agents on the same team share the same policy and value networks
    """
    
    def __init__(
        self,
        obs_dim,
        action_dim,
        device='cpu',
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        ppo_epochs=4,
        mini_batch_size=64
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        # Networks
        self.actor = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        # Training stats
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': []
        }
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(self.device)
            action, log_prob, entropy = self.actor.get_action(
                obs_tensor, deterministic=deterministic
            )
            value = self.critic(obs_tensor)
        
        return (
            action.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            entropy.cpu().numpy()
        )
    
    def update(self, rollout_buffer):
        """
        Perform PPO update using collected experience
        
        Args:
            rollout_buffer: Buffer containing trajectories
            
        Returns:
            Dictionary of training statistics
        """
        # Get data from buffer
        obs, actions, old_log_probs, returns, advantages, old_values = \
            rollout_buffer.get_training_data()
        
        # Convert to tensors
        obs = torch.FloatTensor(obs).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update epochs
        for epoch in range(self.ppo_epochs):
            # Mini-batch updates
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            for start in range(0, len(obs), self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]
                
                batch_obs = obs[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Evaluate actions
                new_log_probs, entropy = self.actor.evaluate_actions(
                    batch_obs, batch_actions
                )
                new_values = self.critic(batch_obs)
                
                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                value_pred_clipped = batch_old_values + torch.clamp(
                    new_values - batch_old_values,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_loss_unclipped = F.mse_loss(new_values, batch_returns)
                value_loss_clipped = F.mse_loss(value_pred_clipped, batch_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss +
                    self.value_loss_coef * value_loss +
                    self.entropy_coef * entropy_loss
                )
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track statistics
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                    )
                
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy'].append(-entropy_loss.item())
                self.training_stats['kl_divergence'].append(kl_div.item())
                self.training_stats['clip_fraction'].append(clip_fraction.item())
        
        # Return average statistics
        return {
            key: np.mean(values[-10:])  # Last 10 updates
            for key, values in self.training_stats.items()
        }
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])