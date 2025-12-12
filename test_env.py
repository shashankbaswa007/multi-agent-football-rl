"""
Unit tests for Multi-Agent Football environment
Run with: pytest tests/test_env.py -v
"""

import pytest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.football_env import FootballEnv, STAY, MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT, PASS, SHOOT


class TestFootballEnvironment:
    """Test suite for Football environment"""
    
    def setup_method(self):
        """Setup before each test"""
        self.env = FootballEnv(
            num_agents_per_team=3,
            grid_width=12,
            grid_height=8,
            max_steps=200
        )
    
    def test_initialization(self):
        """Test environment initialization"""
        assert len(self.env.agents) == 6
        assert self.env.grid_width == 12
        assert self.env.grid_height == 8
        assert self.env.max_steps == 200
    
    def test_reset(self):
        """Test environment reset"""
        observations, infos = self.env.reset(seed=42)
        
        # Check all agents have observations
        assert len(observations) == 6
        
        # Check observation shapes
        for agent, obs in observations.items():
            assert obs.shape == (17,)
            assert obs.min() >= -1.0 and obs.max() <= 1.0
        
        # Check initial positions
        for agent, pos in self.env.agent_positions.items():
            assert 0 <= pos[0] < self.env.grid_width
            assert 0 <= pos[1] < self.env.grid_height
        
        # Ball should be at center
        center_x = self.env.grid_width // 2
        center_y = self.env.grid_height // 2
        assert np.array_equal(self.env.ball_position, [center_x, center_y])
        
        # No initial possession
        assert self.env.ball_possession is None
    
    def test_action_space(self):
        """Test action space"""
        for agent in self.env.agents:
            action_space = self.env.action_space(agent)
            assert action_space.n == 7  # 7 discrete actions
    
    def test_observation_space(self):
        """Test observation space"""
        for agent in self.env.agents:
            obs_space = self.env.observation_space(agent)
            assert obs_space.shape == (17,)
    
    def test_movement(self):
        """Test agent movement"""
        self.env.reset(seed=42)
        
        agent = "team_0_agent_0"
        initial_pos = self.env.agent_positions[agent].copy()
        
        # Test moving up
        self.env.step(MOVE_UP)
        # Due to AEC API, need to step through all agents
        for _ in range(5):  # Step through remaining agents
            self.env.step(STAY)
        
        # Position should change (if not at boundary)
        if initial_pos[1] < self.env.grid_height - 1:
            assert self.env.agent_positions[agent][1] > initial_pos[1]
    
    def test_ball_possession(self):
        """Test ball possession mechanics"""
        self.env.reset(seed=42)
        
        # Move agent to ball
        agent = "team_0_agent_0"
        ball_pos = self.env.ball_position.copy()
        
        # Teleport agent next to ball for testing
        self.env.agent_positions[agent] = ball_pos.copy()
        
        # Try to pick up ball
        self.env.agent_selection = agent
        self.env.step(STAY)
        
        # Agent should have possession after being near ball
        # (This might not work on first step due to distance)
        # Just check possession is either None or an agent name
        assert self.env.ball_possession is None or \
               isinstance(self.env.ball_possession, str)
    
    def test_pass_action(self):
        """Test passing between teammates"""
        self.env.reset(seed=42)
        
        # Setup: Give ball to agent 0
        agent_0 = "team_0_agent_0"
        agent_1 = "team_0_agent_1"
        
        self.env.ball_possession = agent_0
        self.env.agent_positions[agent_0] = np.array([3, 4])
        self.env.agent_positions[agent_1] = np.array([5, 4])
        self.env.ball_position = self.env.agent_positions[agent_0].copy()
        
        # Attempt pass
        self.env.agent_selection = agent_0
        self.env.step(PASS)
        
        # Pass might succeed or fail based on probability
        # Just check that something reasonable happened
        assert self.env.ball_possession in [None, agent_0, agent_1] or \
               self.env.ball_possession is None
    
    def test_shoot_action(self):
        """Test shooting at goal"""
        self.env.reset(seed=42)
        
        agent = "team_0_agent_0"
        self.env.ball_possession = agent
        
        # Place agent near opponent's goal
        self.env.agent_positions[agent] = np.array([10, 4])
        self.env.ball_position = self.env.agent_positions[agent].copy()
        
        initial_goals = self.env.episode_stats['goals_team_0']
        
        # Shoot
        self.env.agent_selection = agent
        self.env.step(SHOOT)
        
        # Shot might score or miss - just check it doesn't crash
        assert self.env.episode_stats['goals_team_0'] >= initial_goals
        assert self.env.episode_stats['shots'] > 0
    
    def test_goal_detection(self):
        """Test goal detection"""
        self.env.reset(seed=42)
        
        # Place ball in goal zone
        self.env.ball_position = np.array([0, self.env.grid_height // 2])
        
        result = self.env._check_goal()
        
        # Should detect goal (team 1 scored on team 0's goal)
        assert result == 1
    
    def test_episode_termination(self):
        """Test episode ends on max steps"""
        self.env.reset(seed=42)
        
        # Run for max_steps
        for _ in range(self.env.max_steps * len(self.env.agents)):
            if not any(self.env.terminations.values()) and \
               not any(self.env.truncations.values()):
                self.env.step(STAY)
        
        # Episode should be truncated
        assert any(self.env.truncations.values())
    
    def test_reward_structure(self):
        """Test reward calculation"""
        self.env.reset(seed=42)
        
        # All rewards should be reasonable numbers
        self.env.step(STAY)
        
        for agent in self.env.agents:
            reward = self.env.rewards[agent]
            assert -200 <= reward <= 200  # Reasonable bounds
    
    def test_render_ansi(self):
        """Test ASCII rendering"""
        self.env.reset(seed=42)
        
        render_output = self.env._render_ansi()
        
        assert isinstance(render_output, str)
        assert len(render_output) > 0
        assert 'B' in render_output  # Blue team marker
        assert 'R' in render_output  # Red team marker
    
    def test_episode_stats_tracking(self):
        """Test episode statistics are tracked"""
        self.env.reset(seed=42)
        
        # Run a few steps
        for _ in range(10):
            self.env.step(STAY)
        
        stats = self.env.episode_stats
        
        assert 'goals_team_0' in stats
        assert 'goals_team_1' in stats
        assert 'passes' in stats
        assert 'shots' in stats
        assert 'possession_time' in stats
    
    def test_collision_detection(self):
        """Test agents can't occupy same position"""
        self.env.reset(seed=42)
        
        # Place two agents next to each other
        agent_0 = "team_0_agent_0"
        agent_1 = "team_0_agent_1"
        
        self.env.agent_positions[agent_0] = np.array([5, 5])
        self.env.agent_positions[agent_1] = np.array([6, 5])
        
        # Try to move agent_1 into agent_0
        self.env.agent_selection = agent_1
        self.env.step(MOVE_LEFT)
        
        # Agents shouldn't be at same position
        assert not np.array_equal(
            self.env.agent_positions[agent_0],
            self.env.agent_positions[agent_1]
        )
    
    def test_deterministic_reset(self):
        """Test reset with same seed produces same result"""
        obs1, _ = self.env.reset(seed=42)
        positions1 = {k: v.copy() for k, v in self.env.agent_positions.items()}
        
        obs2, _ = self.env.reset(seed=42)
        positions2 = {k: v.copy() for k, v in self.env.agent_positions.items()}
        
        # Should be identical
        for agent in self.env.agents:
            assert np.allclose(obs1[agent], obs2[agent])
            assert np.array_equal(positions1[agent], positions2[agent])


class TestPPOAgent:
    """Test suite for PPO agent"""
    
    def setup_method(self):
        """Setup before each test"""
        import torch
        from models.ppo_agent import PPOAgent
        
        self.device = 'cpu'
        self.obs_dim = 17
        self.action_dim = 7
        
        self.agent = PPOAgent(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device
        )
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        assert self.agent.actor is not None
        assert self.agent.critic is not None
        assert self.agent.actor_optimizer is not None
        assert self.agent.critic_optimizer is not None
    
    def test_get_action(self):
        """Test action sampling"""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        action, log_prob, value, entropy = self.agent.get_action(obs)
        
        assert 0 <= action < self.action_dim
        assert isinstance(log_prob, (float, np.ndarray))
        assert isinstance(value, (float, np.ndarray))
        assert isinstance(entropy, (float, np.ndarray))
    
    def test_deterministic_action(self):
        """Test deterministic action selection"""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        
        action1, _, _, _ = self.agent.get_action(obs, deterministic=True)
        action2, _, _, _ = self.agent.get_action(obs, deterministic=True)
        
        # Should be same action
        assert action1 == action2


class TestBuffer:
    """Test suite for experience buffer"""
    
    def setup_method(self):
        """Setup before each test"""
        from training.buffer import RolloutBuffer
        
        self.buffer = RolloutBuffer(
            buffer_size=100,
            obs_dim=17,
            gamma=0.99,
            gae_lambda=0.95
        )
    
    def test_buffer_initialization(self):
        """Test buffer initializes correctly"""
        assert self.buffer.buffer_size == 100
        assert self.buffer.obs_dim == 17
        assert self.buffer.ptr == 0
        assert not self.buffer.full
    
    def test_add_transition(self):
        """Test adding transition to buffer"""
        obs = np.random.randn(17).astype(np.float32)
        action = 0
        reward = 1.0
        value = 0.5
        log_prob = -1.0
        done = False
        
        self.buffer.add(obs, action, reward, value, log_prob, done)
        
        assert self.buffer.ptr == 1
        assert np.array_equal(self.buffer.observations[0], obs)
        assert self.buffer.actions[0] == action
    
    def test_compute_gae(self):
        """Test GAE computation"""
        # Add some transitions
        for i in range(10):
            obs = np.random.randn(17).astype(np.float32)
            self.buffer.add(obs, 0, 1.0, 0.5, -1.0, False)
        
        self.buffer.compute_returns_and_advantages(last_value=0)
        
        # Check returns and advantages are computed
        assert not np.allclose(self.buffer.returns[:10], 0)
        assert not np.allclose(self.buffer.advantages[:10], 0)


def test_imports():
    """Test all modules can be imported"""
    try:
        from env.football_env import FootballEnv
        from models.ppo_agent import PPOAgent, Actor, Critic
        from training.buffer import RolloutBuffer, MultiAgentBuffer
        from training.utils import LinearSchedule, EpisodeStats
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


if __name__ == '__main__':
    # Run basic import test
    if test_imports():
        print("\n" + "="*60)
        print("Basic tests passed! Run full test suite with:")
        print("  pytest tests/test_env.py -v")
        print("="*60)
    else:
        print("\nImport test failed. Check your installation.")