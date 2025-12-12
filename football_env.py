"""
Multi-Agent Football Environment
Grid-based 3v3 football simulation using PettingZoo AEC API
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
import copy

# Action constants
STAY = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
PASS = 5
SHOOT = 6


class FootballEnv(AECEnv):
    """
    3v3 Football environment on a 12x8 grid
    
    State: Each agent observes (17-dim):
        - Self position (2)
        - Ball position (2) + possession flag (1)
        - Teammate positions (4)
        - Opponent positions (6)
        - Goal positions (2)
        
    Actions: 7 discrete actions (movement, pass, shoot)
    
    Rewards: Team-based with shaping for coordination
    """
    
    metadata = {
        'render_modes': ['human', 'ansi'],
        'name': 'football_v0'
    }
    
    def __init__(self, num_agents_per_team=3, grid_width=12, grid_height=8,
                 max_steps=200, render_mode=None):
        super().__init__()
        
        self.num_agents_per_team = num_agents_per_team
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Agent names
        self.agents = [f"team_0_agent_{i}" for i in range(num_agents_per_team)] + \
                     [f"team_1_agent_{i}" for i in range(num_agents_per_team)]
        self.possible_agents = self.agents[:]
        
        # Observation and action spaces
        obs_dim = 17  # As described above
        self._observation_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self._action_spaces = {
            agent: spaces.Discrete(7) for agent in self.agents
        }
        
        # State tracking
        self.step_count = 0
        self.episode_stats = {
            'goals_team_0': 0,
            'goals_team_1': 0,
            'passes': 0,
            'successful_passes': 0,
            'shots': 0,
            'possession_time': {0: 0, 1: 0}
        }
        
    def observation_space(self, agent):
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
        
        self.agents = self.possible_agents[:]
        self.step_count = 0
        
        # Initialize positions
        self.agent_positions = {}
        # Team 0 (left side)
        for i in range(self.num_agents_per_team):
            y = (i + 1) * self.grid_height // (self.num_agents_per_team + 1)
            self.agent_positions[f"team_0_agent_{i}"] = np.array([2, y])
        
        # Team 1 (right side)
        for i in range(self.num_agents_per_team):
            y = (i + 1) * self.grid_height // (self.num_agents_per_team + 1)
            self.agent_positions[f"team_1_agent_{i}"] = np.array([self.grid_width - 3, y])
        
        # Ball starts at center
        self.ball_position = np.array([self.grid_width // 2, self.grid_height // 2])
        self.ball_possession = None  # None or agent name
        
        # Goal positions
        self.goal_team_0 = np.array([0, self.grid_height // 2])
        self.goal_team_1 = np.array([self.grid_width - 1, self.grid_height // 2])
        
        # Rewards and terminations
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # Episode stats
        self.episode_stats = {
            'goals_team_0': 0,
            'goals_team_1': 0,
            'passes': 0,
            'successful_passes': 0,
            'shots': 0,
            'possession_time': {0: 0, 1: 0}
        }
        
        # Agent selector for AEC
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Store last actions for observations
        self.last_actions = {agent: 0 for agent in self.agents}
        
        return self._get_observations(), self.infos
    
    def step(self, action):
        """Execute one agent's action"""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        self.last_actions[agent] = action
        
        # Execute action
        reward = self._execute_action(agent, action)
        
        # Update rewards
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        
        # Check for episode end
        self.step_count += 1
        
        # Goal scored check
        goal_scored_team = self._check_goal()
        if goal_scored_team is not None:
            self._handle_goal(goal_scored_team)
            for a in self.agents:
                self.terminations[a] = True
        
        # Max steps reached
        if self.step_count >= self.max_steps:
            for a in self.agents:
                self.truncations[a] = True
        
        # Update possession time
        if self.ball_possession:
            team = 0 if 'team_0' in self.ball_possession else 1
            self.episode_stats['possession_time'][team] += 1
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()
        
        return self._get_observations(), self.rewards, self.terminations, self.truncations, self.infos
    
    def _execute_action(self, agent, action):
        """Execute agent action and return reward"""
        reward = -0.01  # Small time penalty
        pos = self.agent_positions[agent]
        
        if action == STAY:
            pass
        
        elif action in [MOVE_UP, MOVE_DOWN, MOVE_LEFT, MOVE_RIGHT]:
            # Movement
            new_pos = pos.copy()
            if action == MOVE_UP:
                new_pos[1] = min(self.grid_height - 1, new_pos[1] + 1)
            elif action == MOVE_DOWN:
                new_pos[1] = max(0, new_pos[1] - 1)
            elif action == MOVE_LEFT:
                new_pos[0] = max(0, new_pos[0] - 1)
            elif action == MOVE_RIGHT:
                new_pos[0] = min(self.grid_width - 1, new_pos[0] + 1)
            
            # Check collision with other agents
            collision = False
            for other_agent, other_pos in self.agent_positions.items():
                if other_agent != agent and np.array_equal(new_pos, other_pos):
                    collision = True
                    break
            
            if not collision:
                self.agent_positions[agent] = new_pos
                
                # Pick up ball if nearby
                if self.ball_possession is None:
                    dist_to_ball = np.linalg.norm(new_pos - self.ball_position)
                    if dist_to_ball <= 1.0:
                        self.ball_possession = agent
                        self.ball_position = new_pos.copy()
                        reward += 5  # Possession reward
                
                # Move ball with agent if possessing
                if self.ball_possession == agent:
                    self.ball_position = new_pos.copy()
                
                # Reward for moving toward ball if not possessing
                elif self.ball_possession is None:
                    old_dist = np.linalg.norm(pos - self.ball_position)
                    new_dist = np.linalg.norm(new_pos - self.ball_position)
                    if new_dist < old_dist:
                        reward += 2
            else:
                reward -= 1  # Collision penalty
        
        elif action == PASS:
            # Pass to nearest teammate
            if self.ball_possession == agent:
                team = 0 if 'team_0' in agent else 1
                teammates = [a for a in self.agents if f'team_{team}' in a and a != agent]
                
                if teammates:
                    # Find nearest teammate
                    nearest = None
                    min_dist = float('inf')
                    for teammate in teammates:
                        dist = np.linalg.norm(pos - self.agent_positions[teammate])
                        if dist < min_dist:
                            min_dist = dist
                            nearest = teammate
                    
                    # Pass mechanics
                    self.episode_stats['passes'] += 1
                    pass_success_prob = max(0.3, 1.0 - min_dist / 10.0)
                    
                    if np.random.random() < pass_success_prob:
                        self.ball_possession = nearest
                        self.ball_position = self.agent_positions[nearest].copy()
                        reward += 10  # Successful pass
                        self.episode_stats['successful_passes'] += 1
                    else:
                        # Failed pass, ball becomes loose
                        self.ball_possession = None
                        direction = self.agent_positions[nearest] - pos
                        direction = direction / (np.linalg.norm(direction) + 1e-6)
                        self.ball_position = pos + direction * min(min_dist / 2, 3)
                        self.ball_position = np.clip(self.ball_position, 
                                                     [0, 0], 
                                                     [self.grid_width - 1, self.grid_height - 1])
                        reward -= 5  # Failed pass penalty
            else:
                reward -= 1  # Invalid action
        
        elif action == SHOOT:
            # Shoot at goal
            if self.ball_possession == agent:
                team = 0 if 'team_0' in agent else 1
                target_goal = self.goal_team_1 if team == 0 else self.goal_team_0
                
                self.episode_stats['shots'] += 1
                dist_to_goal = np.linalg.norm(pos - target_goal)
                
                # Shot accuracy decreases with distance
                shot_accuracy = max(0.1, 1.0 - dist_to_goal / 15.0)
                
                if np.random.random() < shot_accuracy:
                    # Goal scored!
                    reward += 100
                    self.episode_stats[f'goals_team_{team}'] += 1
                else:
                    # Missed shot, ball goes loose
                    self.ball_possession = None
                    self.ball_position = target_goal + np.random.randn(2) * 2
                    self.ball_position = np.clip(self.ball_position,
                                                 [0, 0],
                                                 [self.grid_width - 1, self.grid_height - 1])
                    reward -= 5
            else:
                reward -= 1  # Invalid action
        
        return reward
    
    def _check_goal(self):
        """Check if goal was scored"""
        # Goal zone is x=0 or x=width-1, y around center
        if self.ball_position[0] <= 0:
            if abs(self.ball_position[1] - self.grid_height // 2) <= 1:
                return 1  # Team 1 scored on team 0's goal
        elif self.ball_position[0] >= self.grid_width - 1:
            if abs(self.ball_position[1] - self.grid_height // 2) <= 1:
                return 0  # Team 0 scored on team 1's goal
        return None
    
    def _handle_goal(self, scoring_team):
        """Distribute goal rewards to teams"""
        for agent in self.agents:
            agent_team = 0 if 'team_0' in agent else 1
            if agent_team == scoring_team:
                self.rewards[agent] = 100
            else:
                self.rewards[agent] = -100
    
    def _get_observations(self):
        """Get observations for all agents"""
        obs = {}
        for agent in self.agents:
            obs[agent] = self._get_obs(agent)
        return obs
    
    def _get_obs(self, agent):
        """Get observation for a single agent"""
        pos = self.agent_positions[agent]
        team = 0 if 'team_0' in agent else 1
        
        # Normalize positions to [-1, 1]
        norm_pos = (pos / [self.grid_width - 1, self.grid_height - 1]) * 2 - 1
        norm_ball_pos = (self.ball_position / [self.grid_width - 1, self.grid_height - 1]) * 2 - 1
        
        # Ball possession flag
        possession_flag = 1.0 if self.ball_possession == agent else 0.0
        
        # Teammate positions
        teammates = [a for a in self.agents if f'team_{team}' in a and a != agent]
        teammate_positions = []
        for tm in teammates:
            tm_pos = (self.agent_positions[tm] / [self.grid_width - 1, self.grid_height - 1]) * 2 - 1
            teammate_positions.extend(tm_pos)
        
        # Opponent positions
        opponents = [a for a in self.agents if f'team_{1-team}' in a]
        opponent_positions = []
        for opp in opponents:
            opp_pos = (self.agent_positions[opp] / [self.grid_width - 1, self.grid_height - 1]) * 2 - 1
            opponent_positions.extend(opp_pos)
        
        # Goal positions
        my_goal = self.goal_team_0 if team == 0 else self.goal_team_1
        enemy_goal = self.goal_team_1 if team == 0 else self.goal_team_0
        norm_my_goal = (my_goal / [self.grid_width - 1, self.grid_height - 1]) * 2 - 1
        
        # Concatenate observation
        obs = np.concatenate([
            norm_pos,                    # 2
            norm_ball_pos,               # 2
            [possession_flag],           # 1
            teammate_positions,          # 4
            opponent_positions,          # 6
            norm_my_goal,               # 2
        ]).astype(np.float32)
        
        return obs
    
    def render(self):
        """Render the environment"""
        if self.render_mode == 'ansi':
            return self._render_ansi()
        return None
    
    def _render_ansi(self):
        """ASCII rendering"""
        grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Place agents
        for agent, pos in self.agent_positions.items():
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                if 'team_0' in agent:
                    grid[y][x] = 'B'  # Blue team
                else:
                    grid[y][x] = 'R'  # Red team
        
        # Place ball
        bx, by = int(self.ball_position[0]), int(self.ball_position[1])
        if 0 <= bx < self.grid_width and 0 <= by < self.grid_height:
            if grid[by][bx] == ' ':
                grid[by][bx] = 'o'
            else:
                grid[by][bx] = grid[by][bx].lower()  # Possessed ball
        
        # Mark goals
        gy = self.grid_height // 2
        grid[gy][0] = 'G'
        grid[gy][self.grid_width - 1] = 'G'
        
        # Build string
        output = "+" + "-" * self.grid_width + "+\n"
        for row in reversed(grid):
            output += "|" + "".join(row) + "|\n"
        output += "+" + "-" * self.grid_width + "+\n"
        output += f"Step: {self.step_count}/{self.max_steps} | "
        output += f"Score: {self.episode_stats['goals_team_0']}-{self.episode_stats['goals_team_1']}\n"
        
        return output
    
    def close(self):
        pass


# Wrapper for parallel execution
def env_creator(config=None):
    """Create environment instance"""
    return FootballEnv(**config) if config else FootballEnv()