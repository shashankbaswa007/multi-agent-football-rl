"""
Neural Network Models for Multi-Agent PPO
Includes Actor (policy) and Critic (value) networks with modern improvements:
- Layer Normalization for stable training
- Residual connections for better gradient flow
- Multi-head Self-Attention for capturing agent interactions
- Improved initialization and architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import math


class RunningMeanStd:
    """Tracks running mean and std of observations/rewards for normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


def init_weights(module):
    """Improved initialization for better training stability"""
    if isinstance(module, nn.Linear):
        # Use Xavier/Glorot initialization for layers before activation
        nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization for better gradient flow"""
    
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        return F.relu(x + residual)


class Actor(nn.Module):
    """
    Enhanced policy network with modern architecture improvements:
    - Layer normalization for training stability
    - Residual connections for better gradient flow
    - Larger hidden dimensions for better representation
    - Dropout for regularization (optional during training)
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dims=[512, 512], use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Input projection with layer norm
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU()
        )
        
        # Feature extraction layers with residual connections
        self.feature_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.feature_layers.append(ResidualBlock(hidden_dims[i]))
            else:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.ReLU()
                ))
        
        # Action head with separate advantage and value streams (Dueling architecture concept)
        final_dim = hidden_dims[-1]
        self.action_features = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU()
        )
        self.action_head = nn.Linear(final_dim // 2, action_dim)
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs, action_mask=None):
        """
        Forward pass through enhanced architecture
        
        Args:
            obs: (batch_size, obs_dim) observations
            action_mask: (batch_size, action_dim) binary mask for legal actions
            
        Returns:
            action_probs: (batch_size, action_dim) action probabilities
        """
        # Input projection
        x = self.input_proj(obs)
        
        # Feature extraction with residual connections
        for layer in self.feature_layers:
            x = layer(x)
        
        # Action head
        action_features = self.action_features(x)
        logits = self.action_head(action_features)
        
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
    Enhanced value network with modern architecture:
    - Deeper and wider for better value estimation
    - Layer normalization for stable training
    - Residual connections for gradient flow
    - Separate value head with proper scaling
    """
    
    def __init__(self, obs_dim, hidden_dims=[512, 512, 256], use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU()
        )
        
        # Feature extraction with residual connections
        self.feature_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.feature_layers.append(ResidualBlock(hidden_dims[i]))
            else:
                self.feature_layers.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.LayerNorm(hidden_dims[i+1]),
                    nn.ReLU()
                ))
        
        # Value head
        final_dim = hidden_dims[-1]
        self.value_head = nn.Sequential(
            nn.Linear(final_dim, final_dim // 2),
            nn.ReLU(),
            nn.Linear(final_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(init_weights)
    
    def forward(self, obs):
        """
        Forward pass through enhanced architecture
        
        Args:
            obs: (batch_size, obs_dim)
            
        Returns:
            values: (batch_size,) state values
        """
        # Input projection
        x = self.input_proj(obs)
        
        # Feature extraction
        for layer in self.feature_layers:
            x = layer(x)
        
        # Value prediction
        values = self.value_head(x).squeeze(-1)
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
        mini_batch_size=64,
        use_lr_scheduler=True,
        target_kl=0.015
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
        self.target_kl = target_kl  # Early stopping based on KL divergence
        
        # Enhanced networks with residual connections
        self.actor = Actor(obs_dim, action_dim, hidden_dims=[512, 512], use_residual=True).to(device)
        self.critic = Critic(obs_dim, hidden_dims=[512, 512, 256], use_residual=True).to(device)
        
        # Optimizers with weight decay for regularization
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            eps=1e-5
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), 
            lr=lr, 
            weight_decay=1e-5,
            eps=1e-5
        )
        
        # Learning rate schedulers for adaptive learning
        self.use_lr_scheduler = use_lr_scheduler
        if use_lr_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.actor_optimizer, T_0=1000, T_mult=2, eta_min=1e-6
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.critic_optimizer, T_0=1000, T_mult=2, eta_min=1e-6
            )
        
        # Training stats with more detailed tracking
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'kl_divergence': [],
            'clip_fraction': [],
            'explained_variance': [],
            'grad_norm_actor': [],
            'grad_norm_critic': []
        }
        
        # Update counter for learning rate scheduling
        self.update_count = 0
        
        # Normalization for observations and rewards
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=())
        self.normalize_obs = True
        self.normalize_reward = True
        self.clip_obs = 10.0
        self.clip_reward = 10.0
        
        # Epsilon-greedy exploration
        self.epsilon_greedy = 0.0
    
    def normalize_observation(self, obs):
        """Normalize observation using running statistics"""
        if not self.normalize_obs:
            return obs
        obs_array = np.array(obs) if not isinstance(obs, np.ndarray) else obs
        self.obs_rms.update(obs_array.reshape(1, -1) if obs_array.ndim == 1 else obs_array)
        normalized = (obs_array - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8)
        return np.clip(normalized, -self.clip_obs, self.clip_obs)
    
    def normalize_reward(self, reward):
        """Normalize reward using running statistics"""
        if not self.normalize_reward:
            return reward
        self.reward_rms.update(np.array([reward]))
        return np.clip(reward / np.sqrt(self.reward_rms.var + 1e-8), -self.clip_reward, self.clip_reward)
    
    def get_action(self, obs, deterministic=False):
        """Get action from policy with optional epsilon-greedy exploration"""
        # Normalize observation
        obs_normalized = self.normalize_observation(obs)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_normalized).to(self.device)
            
            # Epsilon-greedy: random action with probability epsilon
            if not deterministic and self.epsilon_greedy > 0 and np.random.random() < self.epsilon_greedy:
                action_dim = self.actor.action_head.out_features
                action = torch.tensor(np.random.randint(0, action_dim))
                # Still get log_prob and entropy from policy for logging
                _, log_prob, entropy = self.actor.get_action(obs_tensor, deterministic=False)
            else:
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
        
        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Calculate explained variance for monitoring
        with torch.no_grad():
            y_pred = old_values
            y_true = returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)
        
        # PPO update epochs with early stopping
        for epoch in range(self.ppo_epochs):
            # Mini-batch updates
            indices = np.arange(len(obs))
            np.random.shuffle(indices)
            
            epoch_kl_divs = []
            
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
                
                # Gradient clipping with norm tracking
                actor_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    self.max_grad_norm
                )
                critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Track statistics and check gradient health
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - new_log_probs).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.clip_epsilon).float().mean()
                    )
                    epoch_kl_divs.append(kl_div.item())
                    
                    # Check for gradient issues every 10 updates
                    if self.update_count % 10 == 0:
                        actor_grads = [p.grad.abs().mean().item() for p in self.actor.parameters() if p.grad is not None]
                        if actor_grads and np.mean(actor_grads) < 1e-6:
                            print(f"    ⚠️  WARNING: Vanishing gradients (avg: {np.mean(actor_grads):.2e})")
                
                self.training_stats['policy_loss'].append(policy_loss.item())
                self.training_stats['value_loss'].append(value_loss.item())
                self.training_stats['entropy'].append(-entropy_loss.item())
                self.training_stats['kl_divergence'].append(kl_div.item())
                self.training_stats['clip_fraction'].append(clip_fraction.item())
                self.training_stats['grad_norm_actor'].append(actor_grad_norm.item())
                self.training_stats['grad_norm_critic'].append(critic_grad_norm.item())
            
            # Early stopping if KL divergence is too high
            mean_kl = np.mean(epoch_kl_divs)
            if mean_kl > self.target_kl * 1.5:
                print(f"  Early stopping at epoch {epoch+1}/{self.ppo_epochs} due to high KL divergence: {mean_kl:.4f}")
                break
        
        # Update learning rate schedulers
        self.update_count += 1
        if self.use_lr_scheduler:
            self.actor_scheduler.step()
            self.critic_scheduler.step()
        
        # Add explained variance to stats
        self.training_stats['explained_variance'].append(explained_var.item())
        
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