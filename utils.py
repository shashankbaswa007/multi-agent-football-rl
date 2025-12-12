"""
Utility functions for training
"""

import numpy as np
from collections import deque


class LinearSchedule:
    """Linear interpolation schedule for hyperparameters"""
    
    def __init__(self, start_value, end_value, duration):
        self.start_value = start_value
        self.end_value = end_value
        self.duration = duration
    
    def value(self, step):
        """Get value at current step"""
        if step >= self.duration:
            return self.end_value
        
        fraction = step / self.duration
        return self.start_value + (self.end_value - self.start_value) * fraction


class ExponentialSchedule:
    """Exponential decay schedule"""
    
    def __init__(self, start_value, end_value, decay_rate):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate
    
    def value(self, step):
        """Get value at current step"""
        value = self.start_value * (self.decay_rate ** step)
        return max(value, self.end_value)


class EpisodeStats:
    """Track and compute episode statistics"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        
        self.rewards = deque(maxlen=window_size)
        self.lengths = deque(maxlen=window_size)
        self.wins = deque(maxlen=window_size)
        self.goals_scored = deque(maxlen=window_size)
        self.goals_conceded = deque(maxlen=window_size)
        self.passes = deque(maxlen=window_size)
        self.pass_success = deque(maxlen=window_size)
    
    def add_episode(self, reward, length, won, goals_scored, goals_conceded,
                   passes, successful_passes):
        """Add episode statistics"""
        self.rewards.append(reward)
        self.lengths.append(length)
        self.wins.append(1 if won else 0)
        self.goals_scored.append(goals_scored)
        self.goals_conceded.append(goals_conceded)
        self.passes.append(passes)
        
        if passes > 0:
            self.pass_success.append(successful_passes / passes)
        else:
            self.pass_success.append(0)
    
    def get_stats(self):
        """Get mean statistics over window"""
        if len(self.rewards) == 0:
            return None
        
        return {
            'mean_reward': np.mean(self.rewards),
            'mean_length': np.mean(self.lengths),
            'win_rate': np.mean(self.wins),
            'mean_goals_scored': np.mean(self.goals_scored),
            'mean_goals_conceded': np.mean(self.goals_conceded),
            'mean_passes': np.mean(self.passes),
            'pass_success_rate': np.mean(self.pass_success),
            'goal_difference': np.mean(self.goals_scored) - np.mean(self.goals_conceded)
        }


class RunningMeanStd:
    """
    Running mean and standard deviation
    Used for observation normalization
    """
    
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon
    
    def update(self, x):
        """Update statistics with new batch"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from precomputed moments"""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


def compute_pass_statistics(episode_data):
    """
    Compute detailed pass statistics from episode
    
    Args:
        episode_data: Dictionary with episode information
        
    Returns:
        Dictionary of pass statistics
    """
    passes = episode_data.get('passes', 0)
    successful = episode_data.get('successful_passes', 0)
    
    if passes == 0:
        return {
            'pass_success_rate': 0.0,
            'passes_per_minute': 0.0,
            'avg_pass_distance': 0.0
        }
    
    return {
        'pass_success_rate': successful / passes,
        'passes_per_minute': passes / (episode_data.get('episode_length', 1) / 60.0),
        'avg_pass_distance': episode_data.get('total_pass_distance', 0) / passes
    }


def compute_coordination_metrics(agent_positions_history):
    """
    Compute team coordination metrics
    
    Args:
        agent_positions_history: List of agent positions over time
        
    Returns:
        Dictionary of coordination metrics
    """
    if len(agent_positions_history) == 0:
        return {}
    
    positions = np.array(agent_positions_history)  # (T, num_agents, 2)
    
    # Spread (how dispersed the team is)
    centroids = np.mean(positions, axis=1)  # (T, 2)
    distances_to_centroid = np.linalg.norm(
        positions - centroids[:, np.newaxis, :], axis=2
    )  # (T, num_agents)
    mean_spread = np.mean(distances_to_centroid)
    
    # Formation stability (variance of spread over time)
    spread_over_time = np.mean(distances_to_centroid, axis=1)  # (T,)
    formation_stability = 1.0 / (1.0 + np.std(spread_over_time))
    
    # Synchronized movement (correlation of velocities)
    velocities = np.diff(positions, axis=0)  # (T-1, num_agents, 2)
    if len(velocities) > 0:
        vel_magnitudes = np.linalg.norm(velocities, axis=2)  # (T-1, num_agents)
        
        # Correlation between agent velocities
        correlations = []
        num_agents = velocities.shape[1]
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                corr = np.corrcoef(vel_magnitudes[:, i], vel_magnitudes[:, j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        movement_sync = np.mean(correlations) if correlations else 0.0
    else:
        movement_sync = 0.0
    
    return {
        'mean_spread': mean_spread,
        'formation_stability': formation_stability,
        'movement_synchronization': movement_sync
    }


def detect_emergent_strategies(episodes_data):
    """
    Detect emergent strategies from multiple episodes
    
    Args:
        episodes_data: List of episode dictionaries
        
    Returns:
        Dictionary describing detected strategies
    """
    strategies = {
        'passing_chains': 0,
        'direct_play': 0,
        'possession_play': 0,
        'counter_attacks': 0
    }
    
    for episode in episodes_data:
        # Passing chains: 3+ consecutive passes
        if episode.get('max_pass_chain', 0) >= 3:
            strategies['passing_chains'] += 1
        
        # Direct play: low pass count but high shot accuracy
        if episode.get('passes', 0) < 2 and episode.get('shots', 0) > 0:
            strategies['direct_play'] += 1
        
        # Possession play: high pass success rate and long sequences
        if episode.get('pass_success_rate', 0) > 0.7 and \
           episode.get('possession_time', 0) > 0.6:
            strategies['possession_play'] += 1
        
        # Counter attacks: fast transitions after gaining possession
        if episode.get('fast_breaks', 0) > 0:
            strategies['counter_attacks'] += 1
    
    # Normalize
    total = len(episodes_data)
    if total > 0:
        for key in strategies:
            strategies[key] = strategies[key] / total
    
    return strategies


def action_to_string(action):
    """Convert action index to human-readable string"""
    action_names = {
        0: "STAY",
        1: "MOVE_UP",
        2: "MOVE_DOWN",
        3: "MOVE_LEFT",
        4: "MOVE_RIGHT",
        5: "PASS",
        6: "SHOOT"
    }
    return action_names.get(action, f"UNKNOWN({action})")


def format_time(seconds):
    """Format seconds into readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def save_episode_replay(episode_states, filename):
    """Save episode states to file for replay"""
    import pickle
    
    with open(filename, 'wb') as f:
        pickle.dump(episode_states, f)
    
    print(f"Episode replay saved to {filename}")


def load_episode_replay(filename):
    """Load episode states from file"""
    import pickle
    
    with open(filename, 'rb') as f:
        episode_states = pickle.load(f)
    
    return episode_states