"""
Experience Replay Buffer for PPO
Stores trajectories and computes advantages using GAE
"""

import numpy as np
import torch


class RolloutBuffer:
    """
    Buffer for storing and processing trajectories for PPO training
    Implements Generalized Advantage Estimation (GAE)
    """
    
    def __init__(self, buffer_size, obs_dim, gamma=0.99, gae_lambda=0.95):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        # Storage
        self.observations = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        
        # Computed during finalization
        self.returns = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        
        self.ptr = 0
        self.trajectory_start = 0
        self.full = False
    
    def add(self, obs, action, reward, value, log_prob, done):
        """Add a single transition to the buffer"""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        if self.ptr == 0:
            self.full = True
    
    def compute_returns_and_advantages(self, last_value=0):
        """
        Compute returns and advantages using GAE-Lambda
        Called at the end of each trajectory
        
        Args:
            last_value: Bootstrap value for the last state
        """
        # Get the trajectory we just finished
        if self.full:
            trajectory_slice = slice(self.trajectory_start, self.buffer_size)
        else:
            trajectory_slice = slice(self.trajectory_start, self.ptr)
        
        rewards = self.rewards[trajectory_slice]
        values = self.values[trajectory_slice]
        dones = self.dones[trajectory_slice]
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        last_gae_lambda = 0
        
        # Work backwards through the trajectory
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = last_value
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            
            # GAE
            advantages[t] = last_gae_lambda = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lambda
            )
        
        # Returns = advantages + values
        returns = advantages + values
        
        # Store in buffer
        self.advantages[trajectory_slice] = advantages
        self.returns[trajectory_slice] = returns
        
        # Update trajectory start pointer
        self.trajectory_start = self.ptr
    
    def get_training_data(self):
        """
        Get all data for training
        
        Returns:
            Tuple of (obs, actions, log_probs, returns, advantages, values)
        """
        if self.full:
            indices = slice(0, self.buffer_size)
        else:
            indices = slice(0, self.ptr)
        
        return (
            self.observations[indices],
            self.actions[indices],
            self.log_probs[indices],
            self.returns[indices],
            self.advantages[indices],
            self.values[indices]
        )
    
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.trajectory_start = 0
        self.full = False
    
    def __len__(self):
        return self.buffer_size if self.full else self.ptr


class MultiAgentBuffer:
    """
    Buffer manager for multiple agents
    Handles synchronization of experiences across agents
    """
    
    def __init__(self, num_agents, buffer_size, obs_dim, gamma=0.99, gae_lambda=0.95):
        self.num_agents = num_agents
        self.buffers = [
            RolloutBuffer(buffer_size, obs_dim, gamma, gae_lambda)
            for _ in range(num_agents)
        ]
    
    def add(self, agent_id, obs, action, reward, value, log_prob, done):
        """Add transition for a specific agent"""
        self.buffers[agent_id].add(obs, action, reward, value, log_prob, done)
    
    def compute_returns_and_advantages(self, last_values):
        """Compute returns for all agents"""
        for i, buffer in enumerate(self.buffers):
            buffer.compute_returns_and_advantages(last_values[i])
    
    def get_all_training_data(self):
        """
        Aggregate training data from all agents
        
        Returns:
            Combined RolloutBuffer data
        """
        all_obs = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        all_values = []
        
        for buffer in self.buffers:
            obs, actions, log_probs, returns, advantages, values = \
                buffer.get_training_data()
            
            all_obs.append(obs)
            all_actions.append(actions)
            all_log_probs.append(log_probs)
            all_returns.append(returns)
            all_advantages.append(advantages)
            all_values.append(values)
        
        # Concatenate
        combined_data = (
            np.concatenate(all_obs),
            np.concatenate(all_actions),
            np.concatenate(all_log_probs),
            np.concatenate(all_returns),
            np.concatenate(all_advantages),
            np.concatenate(all_values)
        )
        
        return combined_data
    
    def clear(self):
        """Clear all buffers"""
        for buffer in self.buffers:
            buffer.clear()
    
    def __len__(self):
        return sum(len(buffer) for buffer in self.buffers)


class DummyBuffer:
    """
    Simple wrapper that mimics RolloutBuffer interface
    For use with pre-computed GAE values
    """
    
    def __init__(self, obs, actions, log_probs, returns, advantages, values):
        self.obs = obs
        self.actions = actions
        self.log_probs = log_probs
        self.returns = returns
        self.advantages = advantages
        self.values = values
    
    def get_training_data(self):
        return (
            self.obs,
            self.actions,
            self.log_probs,
            self.returns,
            self.advantages,
            self.values
        )