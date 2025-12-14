"""
Multi-Agent Football Environment
Grid-based 3v3 football simulation using PettingZoo AEC API
"""

import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
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
        # 🚨 FIX #2: Added goal direction vectors (attack_goal_dir + defend_goal_dir = +4 dims)
        # Observation: self_pos(2) + ball_pos(2) + possession(1) + teammates(2*(n-1)) + opponents(2*n) + 
        #              attack_goal_dir(2) + defend_goal_dir(2)
        obs_dim = 2 + 2 + 1 + 2*(num_agents_per_team-1) + 2*num_agents_per_team + 2 + 2
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
            # 🎯 CONTINUOUS PLAY: Reset ball to center instead of ending episode
            self._reset_ball_to_center()
            # Note: Episode continues, no terminations set
        
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
        """Execute agent action and return reward with dense shaping for learning"""
        # 🚨 FIX #1: Minimal time penalty (100x reduction) - prevents drowning sparse rewards
        reward = -0.001  # Was -0.005, now negligible so goals dominate
        pos = self.agent_positions[agent]
        team = 0 if 'team_0' in agent else 1
        
        # CORRECTED: Team 0 attacks RIGHT (high x), Team 1 attacks LEFT (low x)
        target_goal = np.array([self.grid_width - 1, self.grid_height // 2]) if team == 0 \
                      else np.array([0, self.grid_height // 2])
        
        # Dense reward: Possession bonus (encourages keeping ball)
        if self.ball_possession == agent:
            reward += 0.05  # Continuous possession reward (balanced)
        
        if action == STAY:
            # CRITICAL: Heavy penalty for staying still to prevent do-nothing policy
            if self.ball_possession == agent:
                reward -= 2.0  # Very strong penalty for inaction with ball
            else:
                reward -= 0.5  # Penalty for staying still without ball
        
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
                
                # Pick up ball if nearby (interception or pickup)
                if self.ball_possession is None:
                    dist_to_ball = np.linalg.norm(new_pos - self.ball_position)
                    if dist_to_ball <= 1.0:
                        self.ball_possession = agent
                        self.ball_position = new_pos.copy()
                        reward += 10.0  # Strong signal for gaining possession
                
                # 🛡️ DEFENSIVE REWARD: Intercept opponent's ball
                elif self.ball_possession is not None and f'team_{1-team}' in self.ball_possession:
                    # Check if we can steal ball from opponent (increased range)
                    opp_pos = self.agent_positions[self.ball_possession]
                    steal_distance = np.linalg.norm(new_pos - opp_pos)
                    
                    if steal_distance <= 1.5:  # Increased from 1.0
                        # Successful interception!
                        self.ball_possession = agent
                        self.ball_position = new_pos.copy()
                        reward += 25.0  # HUGE reward for stealing ball! (increased from 15)
                    elif steal_distance < 3.0:
                        # Reward for pressuring opponent with ball
                        reward += 0.5 * (3.0 - steal_distance)  # Closer = more reward
                
                # Move ball with agent if possessing (DRIBBLING)
                if self.ball_possession == agent:
                    # 🏃‍♂️ REALISTIC DRIBBLING: Check for opponent pressure
                    opponents = [a for a in self.agents if f'team_{1-team}' in a]
                    nearest_opponent_dist = float('inf')
                    for opp in opponents:
                        opp_dist = np.linalg.norm(new_pos - self.agent_positions[opp])
                        nearest_opponent_dist = min(nearest_opponent_dist, opp_dist)
                    
                    # If opponent very close, might lose ball (realistic pressure)
                    if nearest_opponent_dist < 1.2:
                        # Under pressure! Small chance to lose possession
                        if np.random.random() < 0.15:  # 15% chance when pressured
                            # Lost possession under pressure!
                            self.ball_possession = None
                            self.ball_position = new_pos + np.random.randn(2) * 0.5
                            self.ball_position = np.clip(self.ball_position, [0, 0], 
                                                       [self.grid_width - 1, self.grid_height - 1])
                            reward -= 3.0  # Penalty for losing ball
                            return reward
                    
                    # Successfully dribbled - move ball with agent
                    self.ball_position = new_pos.copy()
                    
                    # 🎯 REALISTIC: Reward for dribbling toward goal (progressive football)
                    old_dist_to_goal = np.linalg.norm(pos - target_goal)
                    new_dist_to_goal = np.linalg.norm(new_pos - target_goal)
                    progress = old_dist_to_goal - new_dist_to_goal
                    
                    if progress > 0:
                        # Reward for moving closer to goal (but not excessive)
                        # Encourage gradual advancement like real football
                        base_reward = 1.0 + progress * 3.0  # Reduced from 5.0
                        
                        # Bonus for getting into dangerous positions (penalty area)
                        if new_dist_to_goal < 3.0:
                            base_reward += 2.0  # In shooting range!
                        
                        reward += base_reward
                    else:
                        reward -= 0.5  # Small penalty for moving away
                
                # Reward for moving toward ball if not possessing
                elif self.ball_possession is None:
                    old_dist = np.linalg.norm(pos - self.ball_position)
                    new_dist = np.linalg.norm(new_pos - self.ball_position)
                    progress = old_dist - new_dist
                    if progress > 0:
                        reward += 0.3 + progress * 2.0  # Reward approaching ball
                
                # 🛡️ DEFENSIVE: When opponent has ball - pressure and defend
                elif f'team_{1-team}' in self.ball_possession:
                    # Primary: Chase the opponent with ball!
                    opp_pos = self.agent_positions[self.ball_possession]
                    old_chase_dist = np.linalg.norm(pos - opp_pos)
                    new_chase_dist = np.linalg.norm(new_pos - opp_pos)
                    
                    if new_chase_dist < old_chase_dist:
                        # Reward for chasing opponent
                        reward += 0.8 + (old_chase_dist - new_chase_dist) * 2.0
                    
                    # Secondary: Defensive positioning
                    defend_goal = np.array([0, self.grid_height // 2]) if team == 0 \
                                  else np.array([self.grid_width - 1, self.grid_height // 2])
                    
                    ball_to_goal_dist = np.linalg.norm(self.ball_position - defend_goal)
                    agent_to_goal_dist = np.linalg.norm(new_pos - defend_goal)
                    
                    # Reward if positioned between ball and goal
                    if agent_to_goal_dist < ball_to_goal_dist:
                        reward += 0.15  # Defensive positioning
                    
                # Reward for positioning (moving toward strategic positions)
                else:
                    # PRIMARY: General attacking positioning - move toward opponent half!
                    field_center_x = self.grid_width / 2
                    if team == 0:
                        # Team 0 attacks right, reward for being in right half
                        if new_pos[0] > field_center_x:
                            reward += 0.3  # Good attacking position
                            if new_pos[0] > field_center_x + 2:
                                reward += 0.2  # Excellent deep position
                    else:
                        # Team 1 attacks left, reward for being in left half
                        if new_pos[0] < field_center_x:
                            reward += 0.3
                            if new_pos[0] < field_center_x - 2:
                                reward += 0.2
                    
                    # SECONDARY: Spacing for passing lanes
                    teammates = [a for a in self.agents if f'team_{team}' in a and a != agent]
                    if teammates:
                        avg_teammate_dist = np.mean([
                            np.linalg.norm(new_pos - self.agent_positions[tm]) 
                            for tm in teammates
                        ])
                        if avg_teammate_dist > 2.0:
                            reward += 0.05
            else:
                reward -= 0.3  # Small collision penalty (was -1, too harsh)
        
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
                    
                    # 🏃‍♂️ REALISTIC PASSING: Distance and pressure affect success
                    MAX_PASS_DISTANCE = 5.0  # Can't pass across entire field
                    
                    if min_dist > MAX_PASS_DISTANCE:
                        # Pass too long! Intercept more likely
                        reward -= 2.0
                        self.ball_possession = None
                        # Ball goes loose between passer and target
                        self.ball_position = (pos + self.agent_positions[nearest]) / 2
                        self.ball_position += np.random.randn(2) * 1.0
                        self.ball_position = np.clip(self.ball_position, [0, 0],
                                                   [self.grid_width - 1, self.grid_height - 1])
                        return reward
                    
                    # Check for opponent interception (realistic defending)
                    opponents = [a for a in self.agents if f'team_{1-team}' in a]
                    pass_intercepted = False
                    for opp in opponents:
                        opp_pos = self.agent_positions[opp]
                        # Check if opponent is on the passing lane
                        dist_to_line = np.abs(np.cross(self.agent_positions[nearest] - pos, 
                                                      opp_pos - pos)) / min_dist
                        if dist_to_line < 1.0 and np.linalg.norm(opp_pos - pos) < min_dist:
                            # Opponent can intercept!
                            if np.random.random() < 0.4:  # 40% interception chance
                                pass_intercepted = True
                                break
                    
                    if pass_intercepted:
                        # Pass intercepted!
                        self.ball_possession = None
                        self.ball_position = (pos + self.agent_positions[nearest]) / 2
                        self.ball_position += np.random.randn(2) * 0.5
                        self.ball_position = np.clip(self.ball_position, [0, 0],
                                                   [self.grid_width - 1, self.grid_height - 1])
                        reward -= 4.0  # Strong penalty for intercepted pass
                        return reward
                    
                    self.episode_stats['passes'] += 1
                    
                    # Pass success probability (distance and pressure based)
                    base_success = 0.7  # Base success rate
                    distance_penalty = min_dist / MAX_PASS_DISTANCE * 0.3
                    pass_success_prob = base_success - distance_penalty
                    
                    if np.random.random() < pass_success_prob:
                        self.ball_possession = nearest
                        self.ball_position = self.agent_positions[nearest].copy()
                        
                        # Base reward for successful pass
                        base_pass_reward = 2.0
                        
                        # CRITICAL: Bonus for forward passes (toward goal)
                        teammate_dist_to_goal = np.linalg.norm(
                            self.agent_positions[nearest] - target_goal
                        )
                        passer_dist_to_goal = np.linalg.norm(pos - target_goal)
                        
                        if teammate_dist_to_goal < passer_dist_to_goal:
                            # Reward proportional to forward progress
                            forward_progress = passer_dist_to_goal - teammate_dist_to_goal
                            base_pass_reward += forward_progress * 2.0
                        
                        reward += base_pass_reward
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
                        reward -= 1.0  # Small penalty (was -8, too harsh)
            else:
                reward -= 1  # Invalid action
        
        elif action == SHOOT:
            # Shoot at goal - REALISTIC: Must be very close!
            if self.ball_possession == agent:
                team = 0 if 'team_0' in agent else 1
                target_goal = self.goal_team_1 if team == 0 else self.goal_team_0
                
                dist_to_goal = np.linalg.norm(pos - target_goal)
                
                # 🏃‍♂️ REALISTIC: Can only shoot when in shooting range (like real football)
                MAX_SHOOTING_DISTANCE = 4.0  # Must be within ~4 units to shoot
                
                if dist_to_goal > MAX_SHOOTING_DISTANCE:
                    # Too far to shoot! Penalize and encourage dribbling closer
                    reward -= 5.0  # Strong penalty for unrealistic long shots
                    # Don't count as a shot
                    return reward
                
                # Within shooting range - count the shot
                self.episode_stats['shots'] += 1
                
                # 🎯 ENCOURAGE SHOOTING when very close!
                if dist_to_goal < 2.5:
                    reward += 5.0  # Big bonus for close shot attempt
                
                # Shot accuracy model (MUCH steeper - realistic football)
                # Close shots have high success, drops off quickly
                if dist_to_goal < 1.5:
                    shot_accuracy = 0.85  # Very high from point-blank
                elif dist_to_goal < 2.5:
                    shot_accuracy = 0.60  # Good from close
                elif dist_to_goal < 3.5:
                    shot_accuracy = 0.35  # Moderate from medium
                else:
                    shot_accuracy = 0.15  # Low from max range
                
                if np.random.random() < shot_accuracy:
                    # 🏆 GOAL SCORED! Big reward but not overwhelming
                    reward += 100.0  # Reduced from 200 - still significant but balanced
                    self.episode_stats[f'goals_team_{team}'] += 1
                else:
                    # Missed shot - lose possession (realistic)
                    if dist_to_goal < 2.5:
                        reward += 0.5  # Small reward for close attempt
                    else:
                        reward -= 2.0  # Penalty for missing from far
                    
                    # Missed shot, ball goes loose near goal (realistic rebound)
                    self.ball_possession = None
                    self.ball_position = target_goal + np.random.randn(2) * 1.5
                    self.ball_position = np.clip(self.ball_position,
                                                 [0, 0],
                                                 [self.grid_width - 1, self.grid_height - 1])
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
    
    def _reset_ball_to_center(self):
        """Reset ball to center of field after a goal - continuous play!"""
        # Place ball at center of field
        self.ball_position = np.array([
            self.grid_width / 2.0,
            self.grid_height / 2.0
        ], dtype=np.float32)
        
        # Clear possession - ball is free for anyone to grab
        self.ball_possession = None
        
        # Optional: Reset agent positions to their starting sides
        # This creates a "kickoff" scenario after each goal
        for agent in self.agents:
            team = 0 if 'team_0' in agent else 1
            agent_num = int(agent.split('_')[-1])
            
            # Team 0 starts on left side, Team 1 on right side
            if team == 0:
                x_pos = self.grid_width * 0.3  # Left third of field
            else:
                x_pos = self.grid_width * 0.7  # Right third of field
            
            # Spread agents vertically
            y_spacing = self.grid_height / (self.num_agents_per_team + 1)
            y_pos = y_spacing * (agent_num + 1)
            
            self.agent_positions[agent] = np.array([x_pos, y_pos], dtype=np.float32)
    
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
        
        # 🚨 FIX #2: Direction vectors to goals (normalized unit vectors)
        # Attack direction: points toward enemy goal
        attack_goal_vec = enemy_goal - pos
        attack_goal_dir = attack_goal_vec / (np.linalg.norm(attack_goal_vec) + 1e-6)
        
        # Defend direction: points toward own goal
        defend_goal_vec = my_goal - pos
        defend_goal_dir = defend_goal_vec / (np.linalg.norm(defend_goal_vec) + 1e-6)
        
        # Concatenate observation
        obs = np.concatenate([
            norm_pos,                    # 2
            norm_ball_pos,               # 2
            [possession_flag],           # 1
            teammate_positions,          # 2*(n-1)
            opponent_positions,          # 2*n
            attack_goal_dir,            # 2 - NEW: direction to attack
            defend_goal_dir,            # 2 - NEW: direction to defend
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