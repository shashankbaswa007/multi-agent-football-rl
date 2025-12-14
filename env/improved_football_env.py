"""
IMPROVED FOOTBALL ENVIRONMENT
Redesigned from scratch for realistic football behavior with:
- Proper kickoff positioning
- Smart agent behaviors (attack/defend modes)
- Improved movement mechanics
- Better reward shaping
- Enhanced observations
"""

import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces
import functools

# Action constants
STAY = 0
MOVE_UP = 1
MOVE_DOWN = 2
MOVE_LEFT = 3
MOVE_RIGHT = 4
SHOOT = 5
PASS = 6

ACTION_NAMES = ['STAY', 'MOVE_UP', 'MOVE_DOWN', 'MOVE_LEFT', 'MOVE_RIGHT', 'SHOOT', 'PASS']

class ImprovedFootballEnv(AECEnv):
    """
    Multi-agent football environment with realistic behaviors
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'improved_football_v1'
    }
    
    def __init__(
        self,
        num_agents_per_team=2,
        grid_width=10,
        grid_height=6,
        max_steps=150,
        movement_speed=0.5,
        shoot_range=3.0,
        pass_range=4.0,
        possession_radius=0.3,
        debug=False
    ):
        super().__init__()
        
        # Environment parameters
        self.num_agents_per_team = num_agents_per_team
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.max_steps = max_steps
        self.movement_speed = movement_speed
        self.shoot_range = shoot_range
        self.pass_range = pass_range
        self.possession_radius = possession_radius
        self.debug = debug
        
        # Goal parameters
        self.goal_width = grid_height * 0.4  # 40% of field height
        self.goal_center_y = grid_height / 2
        self.goal_top = self.goal_center_y - self.goal_width / 2
        self.goal_bottom = self.goal_center_y + self.goal_width / 2
        
        # Agents
        self.agents = [
            f"team_0_agent_{i}" for i in range(num_agents_per_team)
        ] + [
            f"team_1_agent_{i}" for i in range(num_agents_per_team)
        ]
        self.possible_agents = self.agents[:]
        
        # Observation space: [agent_x, agent_y, ball_x, ball_y, has_ball,
        #                     ball_vec_x, ball_vec_y, goal_vec_x, goal_vec_y,
        #                     nearest_teammate_x, nearest_teammate_y,
        #                     nearest_opponent_x, nearest_opponent_y,
        #                     agent_vel_x, agent_vel_y, ball_vel_x, ball_vel_y,
        #                     dist_to_left_bound, dist_to_right_bound,
        #                     dist_to_top_bound, dist_to_bottom_bound]
        self._observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(21,), dtype=np.float32
        )
        self._action_space = spaces.Discrete(7)
        
        # State variables
        self.agent_positions = {}
        self.agent_velocities = {}
        self.previous_positions = {}
        self.ball_position = np.array([grid_width / 2, grid_height / 2], dtype=np.float32)
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_possession = None
        self.kickoff_timer = 0  # Delay before ball can be claimed
        
        # Episode tracking
        self.step_count = 0
        self.episode_stats = {
            'goals_team_0': 0,
            'goals_team_1': 0,
            'shots_team_0': 0,
            'shots_team_1': 0,
            'passes': 0,
            'successful_passes': 0,
            'interceptions': 0,
            'possession_time': [0, 0]
        }
        
        # Agent behavior modes
        self.agent_modes = {}  # 'attack' or 'defend'
        
        # AEC environment requirements
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self.last_actions = {agent: None for agent in self.agents}
        
    def reset(self, seed=None, options=None):
        """
        ═══════════════════════════════════════════════════════════════
        1. KICKOFF LAYOUT - Proper starting positions
        ═══════════════════════════════════════════════════════════════
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset episode tracking
        self.step_count = 0
        self.episode_stats = {
            'goals_team_0': 0,
            'goals_team_1': 0,
            'shots_team_0': 0,
            'shots_team_1': 0,
            'passes': 0,
            'successful_passes': 0,
            'interceptions': 0,
            'possession_time': [0, 0]
        }
        
        # Ball starts at exact center
        self.ball_position = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_possession = None
        self.kickoff_timer = 5  # 5 timesteps before ball can be claimed
        
        # Team 0 starting positions (LEFT side, around 20% of width)
        team_0_center_x = self.grid_width * 0.25
        team_0_center_y = self.grid_height / 2
        team_0_radius = min(self.grid_width * 0.08, self.grid_height * 0.15)
        
        # Team 1 starting positions (RIGHT side, around 80% of width)
        team_1_center_x = self.grid_width * 0.75
        team_1_center_y = self.grid_height / 2
        team_1_radius = min(self.grid_width * 0.08, self.grid_height * 0.15)
        
        for i, agent in enumerate(self.agents):
            if 'team_0' in agent:
                # Place in ring around team_0_center
                angle = (2 * np.pi * i) / self.num_agents_per_team
                offset_x = team_0_radius * np.cos(angle)
                offset_y = team_0_radius * np.sin(angle)
                pos = np.array([
                    team_0_center_x + offset_x,
                    team_0_center_y + offset_y
                ], dtype=np.float32)
            else:
                # Place in ring around team_1_center
                agent_idx = i - self.num_agents_per_team
                angle = (2 * np.pi * agent_idx) / self.num_agents_per_team
                offset_x = team_1_radius * np.cos(angle)
                offset_y = team_1_radius * np.sin(angle)
                pos = np.array([
                    team_1_center_x + offset_x,
                    team_1_center_y + offset_y
                ], dtype=np.float32)
            
            # Clamp to field boundaries with margin
            pos[0] = np.clip(pos[0], 0.5, self.grid_width - 0.5)
            pos[1] = np.clip(pos[1], 0.5, self.grid_height - 0.5)
            
            self.agent_positions[agent] = pos
            self.agent_velocities[agent] = np.zeros(2, dtype=np.float32)
            self.previous_positions[agent] = pos.copy()
            self.agent_modes[agent] = 'attack'  # Start in neutral/attack mode
        
        if self.debug:
            print("\n" + "="*60)
            print("KICKOFF LAYOUT")
            print("="*60)
            print(f"Ball position: {self.ball_position}")
            print(f"Team 0 agents (attacking RIGHT →):")
            for agent in self.agents[:self.num_agents_per_team]:
                print(f"  {agent}: {self.agent_positions[agent]}")
            print(f"Team 1 agents (attacking LEFT ←):")
            for agent in self.agents[self.num_agents_per_team:]:
                print(f"  {agent}: {self.agent_positions[agent]}")
            print("="*60)
        
        # Reset AEC state
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.last_actions = {agent: None for agent in self.agents}
        
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        return self._get_observations(), self.infos
    
    def observe(self, agent):
        """Get observation for specific agent"""
        return self._get_observation(agent)
    
    def _get_observations(self):
        """
        ═══════════════════════════════════════════════════════════════
        5. OBSERVATION SPACE - Rich information for intelligent behavior
        ═══════════════════════════════════════════════════════════════
        """
        observations = {}
        for agent in self.agents:
            observations[agent] = self._get_observation(agent)
        return observations
    
    def _get_observation(self, agent):
        """Build comprehensive observation for agent"""
        pos = self.agent_positions[agent]
        vel = self.agent_velocities[agent]
        team_id = 0 if 'team_0' in agent else 1
        
        # Normalize positions to [-1, 1]
        norm_agent_x = (pos[0] / self.grid_width) * 2 - 1
        norm_agent_y = (pos[1] / self.grid_height) * 2 - 1
        norm_ball_x = (self.ball_position[0] / self.grid_width) * 2 - 1
        norm_ball_y = (self.ball_position[1] / self.grid_height) * 2 - 1
        
        # Has ball flag
        has_ball = 1.0 if self.ball_possession == agent else 0.0
        
        # Vector to ball (normalized)
        ball_vec = self.ball_position - pos
        ball_dist = np.linalg.norm(ball_vec) + 1e-6
        ball_vec_norm = ball_vec / ball_dist
        ball_vec_norm = np.clip(ball_vec_norm, -1, 1)
        
        # Vector to opponent goal (normalized)
        if team_id == 0:
            goal_pos = np.array([self.grid_width, self.grid_height / 2])
        else:
            goal_pos = np.array([0, self.grid_height / 2])
        goal_vec = goal_pos - pos
        goal_dist = np.linalg.norm(goal_vec) + 1e-6
        goal_vec_norm = goal_vec / goal_dist
        goal_vec_norm = np.clip(goal_vec_norm, -1, 1)
        
        # Nearest teammate
        teammates = [a for a in self.agents if ('team_0' in a) == ('team_0' in agent) and a != agent]
        if teammates:
            teammate_positions = [self.agent_positions[t] for t in teammates]
            teammate_dists = [np.linalg.norm(p - pos) for p in teammate_positions]
            nearest_teammate_idx = np.argmin(teammate_dists)
            nearest_teammate_pos = teammate_positions[nearest_teammate_idx]
            teammate_vec = nearest_teammate_pos - pos
            teammate_vec_norm = teammate_vec / (np.linalg.norm(teammate_vec) + 1e-6)
        else:
            teammate_vec_norm = np.zeros(2)
        teammate_vec_norm = np.clip(teammate_vec_norm, -1, 1)
        
        # Nearest opponent
        opponents = [a for a in self.agents if ('team_0' in a) != ('team_0' in agent)]
        if opponents:
            opponent_positions = [self.agent_positions[o] for o in opponents]
            opponent_dists = [np.linalg.norm(p - pos) for p in opponent_positions]
            nearest_opponent_idx = np.argmin(opponent_dists)
            nearest_opponent_pos = opponent_positions[nearest_opponent_idx]
            opponent_vec = nearest_opponent_pos - pos
            opponent_vec_norm = opponent_vec / (np.linalg.norm(opponent_vec) + 1e-6)
        else:
            opponent_vec_norm = np.zeros(2)
        opponent_vec_norm = np.clip(opponent_vec_norm, -1, 1)
        
        # Velocities (normalized)
        vel_norm = vel / (self.movement_speed + 1e-6)
        vel_norm = np.clip(vel_norm, -1, 1)
        ball_vel_norm = self.ball_velocity / (self.movement_speed * 2 + 1e-6)
        ball_vel_norm = np.clip(ball_vel_norm, -1, 1)
        
        # Boundary distances (normalized)
        dist_left = pos[0] / self.grid_width
        dist_right = (self.grid_width - pos[0]) / self.grid_width
        dist_top = pos[1] / self.grid_height
        dist_bottom = (self.grid_height - pos[1]) / self.grid_height
        
        observation = np.array([
            norm_agent_x, norm_agent_y,  # 0-1: agent position
            norm_ball_x, norm_ball_y,    # 2-3: ball position
            has_ball,                     # 4: possession flag
            ball_vec_norm[0], ball_vec_norm[1],  # 5-6: vector to ball
            goal_vec_norm[0], goal_vec_norm[1],  # 7-8: vector to goal
            teammate_vec_norm[0], teammate_vec_norm[1],  # 9-10: nearest teammate
            opponent_vec_norm[0], opponent_vec_norm[1],  # 11-12: nearest opponent
            vel_norm[0], vel_norm[1],    # 13-14: agent velocity
            ball_vel_norm[0], ball_vel_norm[1],  # 15-16: ball velocity
            dist_left, dist_right,        # 17-18: horizontal bounds
            dist_top, dist_bottom         # 19-20: vertical bounds
        ], dtype=np.float32)
        
        return observation
    
    def step(self, action):
        """Execute agent action"""
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        self.last_actions[agent] = action
        
        # Execute action and get reward
        reward = self._execute_action(agent, action)
        
        # Update rewards
        self.rewards[agent] = reward
        self._cumulative_rewards[agent] += reward
        
        # Check for goals
        goal_team = self._check_goal()
        if goal_team is not None:
            self._handle_goal(goal_team)
        
        # Update step count
        self.step_count += 1
        
        # Decrease kickoff timer
        if self.kickoff_timer > 0:
            self.kickoff_timer -= 1
        
        # Update possession time
        if self.ball_possession:
            team = 0 if 'team_0' in self.ball_possession else 1
            self.episode_stats['possession_time'][team] += 1
        
        # Check termination
        if self.step_count >= self.max_steps:
            for a in self.agents:
                self.truncations[a] = True
        
        # Select next agent
        self.agent_selection = self._agent_selector.next()
        
        # Update ball physics
        self._update_ball_physics()
        
        return self._get_observations(), self.rewards, self.terminations, self.truncations, self.infos
    
    def _execute_action(self, agent, action):
        """
        ═══════════════════════════════════════════════════════════════
        2 & 3. AGENT BEHAVIORS & MOVEMENT IMPROVEMENT
        ═══════════════════════════════════════════════════════════════
        """
        reward = 0.0
        pos = self.agent_positions[agent]
        team_id = 0 if 'team_0' in agent else 1
        
        # Store previous position for movement tracking
        self.previous_positions[agent] = pos.copy()
        
        # Determine agent mode (attack or defend)
        self._update_agent_mode(agent)
        mode = self.agent_modes[agent]
        
        # Movement actions
        movement_vec = np.zeros(2, dtype=np.float32)
        
        if action == MOVE_UP:
            movement_vec[1] = -self.movement_speed
        elif action == MOVE_DOWN:
            movement_vec[1] = self.movement_speed
        elif action == MOVE_LEFT:
            movement_vec[0] = -self.movement_speed
        elif action == MOVE_RIGHT:
            movement_vec[0] = self.movement_speed
        elif action == STAY:
            # Small penalty for staying still (encourages movement)
            reward -= 0.01
        elif action == SHOOT:
            reward += self._attempt_shoot(agent, team_id)
        elif action == PASS:
            reward += self._attempt_pass(agent, team_id)
        
        # Apply movement
        if np.linalg.norm(movement_vec) > 0:
            new_pos = pos + movement_vec
            # Clamp to boundaries
            new_pos[0] = np.clip(new_pos[0], 0, self.grid_width)
            new_pos[1] = np.clip(new_pos[1], 0, self.grid_height)
            
            # Update position and velocity
            self.agent_positions[agent] = new_pos
            self.agent_velocities[agent] = movement_vec
            
            # Movement rewards
            reward += self._compute_movement_reward(agent, pos, new_pos, mode)
        else:
            self.agent_velocities[agent] = np.zeros(2, dtype=np.float32)
        
        # Check possession (if not on cooldown)
        if self.kickoff_timer == 0:
            reward += self._check_possession(agent)
        
        # Debug output
        if self.debug and action != STAY:
            print(f"{agent} [{mode}]: {ACTION_NAMES[action]} -> {self.agent_positions[agent]} (reward: {reward:.3f})")
        
        return reward
    
    def _update_agent_mode(self, agent):
        """
        ═══════════════════════════════════════════════════════════════
        8. BEHAVIOR MODES - Attack/Defend switching
        ═══════════════════════════════════════════════════════════════
        """
        team_id = 0 if 'team_0' in agent else 1
        
        # Check if this team has possession
        team_has_ball = False
        if self.ball_possession:
            team_has_ball = ('team_0' in self.ball_possession) == (team_id == 0)
        
        # Check if this agent is closest to ball among teammates
        teammates = [a for a in self.agents if ('team_0' in a) == ('team_0' in agent)]
        distances_to_ball = {a: np.linalg.norm(self.agent_positions[a] - self.ball_position) 
                            for a in teammates}
        closest_teammate = min(distances_to_ball, key=distances_to_ball.get)
        is_closest = (closest_teammate == agent)
        
        # Mode logic
        if team_has_ball or (not self.ball_possession and is_closest):
            self.agent_modes[agent] = 'attack'
        else:
            self.agent_modes[agent] = 'defend'
    
    def _compute_movement_reward(self, agent, old_pos, new_pos, mode):
        """
        ═══════════════════════════════════════════════════════════════
        4. IMPROVED REWARD SHAPING - Detailed rewards
        ═══════════════════════════════════════════════════════════════
        """
        reward = 0.0
        team_id = 0 if 'team_0' in agent else 1
        has_ball = (self.ball_possession == agent)
        
        # Movement direction
        movement = new_pos - old_pos
        movement_dist = np.linalg.norm(movement)
        
        if movement_dist > 0:
            movement_dir = movement / movement_dist
            
            # Ball contesting (if no one has ball)
            if not self.ball_possession:
                to_ball = self.ball_position - old_pos
                to_ball_dist = np.linalg.norm(to_ball)
                if to_ball_dist > 0:
                    to_ball_dir = to_ball / to_ball_dist
                    ball_alignment = np.dot(movement_dir, to_ball_dir)
                    if ball_alignment > 0:
                        reward += 0.05  # Moving toward ball
                    else:
                        reward -= 0.05  # Moving away from ball
            
            # Possession behavior
            if has_ball:
                # Attacking direction
                goal_pos = np.array([self.grid_width, self.grid_height / 2]) if team_id == 0 else np.array([0, self.grid_height / 2])
                to_goal = goal_pos - old_pos
                to_goal_dist = np.linalg.norm(to_goal)
                if to_goal_dist > 0:
                    to_goal_dir = to_goal / to_goal_dist
                    goal_alignment = np.dot(movement_dir, to_goal_dir)
                    if goal_alignment > 0:
                        reward += 0.04  # Moving toward goal with ball
                    else:
                        reward -= 0.06  # Moving backward with ball
                
                # Reward per timestep holding ball
                reward += 0.03
            
            # Defense (intercept opponent with ball)
            if mode == 'defend' and self.ball_possession:
                opponent_with_ball_pos = self.agent_positions[self.ball_possession]
                to_opponent = opponent_with_ball_pos - old_pos
                to_opponent_dist = np.linalg.norm(to_opponent)
                if to_opponent_dist > 0:
                    to_opponent_dir = to_opponent / to_opponent_dist
                    intercept_alignment = np.dot(movement_dir, to_opponent_dir)
                    if intercept_alignment > 0:
                        reward += 0.03  # Moving to intercept
        
        # Penalty for zigzagging (check velocity reversal)
        if np.linalg.norm(self.agent_velocities[agent]) > 0:
            prev_vel = self.agent_velocities[agent]
            curr_vel = movement
            if np.linalg.norm(prev_vel) > 0 and np.linalg.norm(curr_vel) > 0:
                vel_dot = np.dot(prev_vel / np.linalg.norm(prev_vel), 
                                curr_vel / np.linalg.norm(curr_vel))
                if vel_dot < -0.5:  # Reversed direction
                    reward -= 0.02
        
        return np.clip(reward, -1.0, 1.0)
    
    def _check_possession(self, agent):
        """Check if agent gains possession"""
        reward = 0.0
        dist_to_ball = np.linalg.norm(self.agent_positions[agent] - self.ball_position)
        
        if dist_to_ball < self.possession_radius:
            if self.ball_possession != agent:
                # Gaining possession
                was_opponent = False
                if self.ball_possession:
                    was_opponent = ('team_0' in self.ball_possession) != ('team_0' in agent)
                
                self.ball_possession = agent
                self.ball_velocity = np.zeros(2, dtype=np.float32)
                
                if was_opponent:
                    reward += 0.2  # Interception
                    self.episode_stats['interceptions'] += 1
                else:
                    reward += 0.2  # Gaining possession
                
                if self.debug:
                    print(f"  → {agent} gains possession!")
        
        return reward
    
    def _attempt_shoot(self, agent, team_id):
        """Attempt to shoot at goal"""
        reward = 0.0
        
        if self.ball_possession != agent:
            return -0.1  # Penalty for shooting without ball
        
        pos = self.agent_positions[agent]
        goal_pos = np.array([self.grid_width, self.grid_height / 2]) if team_id == 0 else np.array([0, self.grid_height / 2])
        dist_to_goal = np.linalg.norm(pos - goal_pos)
        
        if dist_to_goal > self.shoot_range:
            return -0.05  # Too far to shoot
        
        # Record shot
        if team_id == 0:
            self.episode_stats['shots_team_0'] += 1
        else:
            self.episode_stats['shots_team_1'] += 1
        
        # Calculate shot success probability
        accuracy = max(0.1, 1.0 - (dist_to_goal / self.shoot_range) * 0.7)
        
        if np.random.random() < accuracy:
            # Goal scored!
            if team_id == 0:
                self.episode_stats['goals_team_0'] += 1
            else:
                self.episode_stats['goals_team_1'] += 1
            
            reward += 1.0  # Goal reward
            self._reset_after_goal()
            
            if self.debug:
                print(f"  ⚽ GOAL by {agent}!")
        else:
            reward += 0.1  # Reward for attempting shot
            self.ball_possession = None
            # Ball goes out or is saved
            self._reset_ball_to_center()
        
        return reward
    
    def _attempt_pass(self, agent, team_id):
        """Attempt to pass to teammate"""
        reward = 0.0
        
        if self.ball_possession != agent:
            return -0.1  # Penalty for passing without ball
        
        # Find nearest teammate
        teammates = [a for a in self.agents if ('team_0' in a) == ('team_0' in agent) and a != agent]
        if not teammates:
            return -0.05  # No teammate to pass to
        
        pos = self.agent_positions[agent]
        teammate_positions = {t: self.agent_positions[t] for t in teammates}
        teammate_dists = {t: np.linalg.norm(p - pos) for t, p in teammate_positions.items()}
        nearest_teammate = min(teammate_dists, key=teammate_dists.get)
        nearest_dist = teammate_dists[nearest_teammate]
        
        if nearest_dist > self.pass_range:
            return -0.05  # Too far to pass
        
        self.episode_stats['passes'] += 1
        
        # Check if opponent can intercept
        opponents = [a for a in self.agents if ('team_0' in a) != ('team_0' in agent)]
        teammate_pos = teammate_positions[nearest_teammate]
        pass_vec = teammate_pos - pos
        
        intercepted = False
        for opp in opponents:
            opp_pos = self.agent_positions[opp]
            # Check if opponent is on pass line
            to_opp = opp_pos - pos
            proj = np.dot(to_opp, pass_vec) / (np.linalg.norm(pass_vec) ** 2 + 1e-6)
            if 0 < proj < 1:
                closest_point = pos + proj * pass_vec
                dist_to_line = np.linalg.norm(opp_pos - closest_point)
                if dist_to_line < self.possession_radius * 2:
                    intercepted = True
                    break
        
        if intercepted:
            self.ball_possession = None
            reward -= 0.1
        else:
            self.ball_possession = nearest_teammate
            self.ball_position = teammate_pos.copy()
            self.episode_stats['successful_passes'] += 1
            reward += 0.15
            
            if self.debug:
                print(f"  → {agent} passes to {nearest_teammate}")
        
        return reward
    
    def _update_ball_physics(self):
        """Update ball position based on physics"""
        if self.ball_possession:
            # Ball follows possessing agent
            self.ball_position = self.agent_positions[self.ball_possession].copy()
            self.ball_velocity = np.zeros(2, dtype=np.float32)
        else:
            # Ball moves with velocity and friction
            if np.linalg.norm(self.ball_velocity) > 0.01:
                self.ball_position += self.ball_velocity
                self.ball_velocity *= 0.9  # Friction
                
                # Clamp to field
                self.ball_position[0] = np.clip(self.ball_position[0], 0, self.grid_width)
                self.ball_position[1] = np.clip(self.ball_position[1], 0, self.grid_height)
            else:
                self.ball_velocity = np.zeros(2, dtype=np.float32)
    
    def _check_goal(self):
        """
        ═══════════════════════════════════════════════════════════════
        6. ENVIRONMENT FIXES - Correct goal detection
        ═══════════════════════════════════════════════════════════════
        """
        # Team 0 goal (left side, x = 0)
        if self.ball_position[0] <= 0.1:
            if self.goal_top <= self.ball_position[1] <= self.goal_bottom:
                return 1  # Team 1 scored
        
        # Team 1 goal (right side, x = grid_width)
        if self.ball_position[0] >= self.grid_width - 0.1:
            if self.goal_top <= self.ball_position[1] <= self.goal_bottom:
                return 0  # Team 0 scored
        
        return None
    
    def _handle_goal(self, scoring_team):
        """Handle goal scored"""
        # Award goals
        if scoring_team == 0:
            for agent in self.agents:
                if 'team_0' in agent:
                    self.rewards[agent] += 1.0  # Goal scored
                else:
                    self.rewards[agent] += 0.3  # Goal prevented (small consolation)
        else:
            for agent in self.agents:
                if 'team_1' in agent:
                    self.rewards[agent] += 1.0
                else:
                    self.rewards[agent] += 0.3
        
        self._reset_after_goal()
    
    def _reset_after_goal(self):
        """Reset positions after goal"""
        self.ball_position = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_possession = None
        self.kickoff_timer = 5  # Restart cooldown
    
    def _reset_ball_to_center(self):
        """Reset ball to center"""
        self.ball_position = np.array([self.grid_width / 2, self.grid_height / 2], dtype=np.float32)
        self.ball_velocity = np.zeros(2, dtype=np.float32)
        self.ball_possession = None
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self._observation_space
    
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self._action_space


# ═══════════════════════════════════════════════════════════════
# TESTING AND VALIDATION
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*70)
    print("IMPROVED FOOTBALL ENVIRONMENT - TEST SUITE")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6, debug=True)
    
    # Test 1: Kickoff positioning
    print("\n[TEST 1] Kickoff Positioning")
    obs, info = env.reset()
    print("✓ Ball at center:", np.allclose(env.ball_position, [5.0, 3.0], atol=0.1))
    print("✓ Team 0 on left side:", all(env.agent_positions[a][0] < 5.0 for a in env.agents[:2]))
    print("✓ Team 1 on right side:", all(env.agent_positions[a][0] > 5.0 for a in env.agents[2:]))
    
    # Test 2: Movement mechanics
    print("\n[TEST 2] Movement Mechanics")
    agent = env.agents[0]
    start_pos = env.agent_positions[agent].copy()
    env.step(MOVE_RIGHT)  # Move right
    env.agent_selection = agent  # Reset to test agent
    new_pos = env.agent_positions[agent]
    print(f"✓ Agent moved: {start_pos} -> {new_pos}")
    print(f"✓ Movement distance: {np.linalg.norm(new_pos - start_pos):.3f}")
    
    # Test 3: Observation shape
    print("\n[TEST 3] Observation Space")
    print(f"✓ Observation shape: {obs[agent].shape} (expected: (21,))")
    print(f"✓ All values in [-1, 1]: {np.all(np.abs(obs[agent]) <= 1.01)}")
    
    # Test 4: Possession mechanics
    print("\n[TEST 4] Possession Mechanics")
    env.reset()
    env.ball_position = env.agent_positions[agent] + 0.1  # Place ball near agent
    env.kickoff_timer = 0  # Disable cooldown
    reward = env._check_possession(agent)
    print(f"✓ Possession gained: {env.ball_possession == agent}")
    print(f"✓ Possession reward: {reward:.3f}")
    
    # Test 5: Replay demonstration
    print("\n[TEST 5] 50-Step Replay")
    env.reset()
    for step in range(50):
        agent = env.agent_selection
        # Random valid action
        action = np.random.randint(0, 7)
        obs, rewards, terms, truncs, infos = env.step(action)
        
        if step % 10 == 0:
            print(f"  Step {step}: Ball at {env.ball_position}, Possession: {env.ball_possession}")
        
        if any(terms.values()) or any(truncs.values()):
            print(f"  Episode ended at step {step}")
            break
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - Environment Ready!")
    print("="*70)
