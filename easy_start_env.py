#!/usr/bin/env python3
"""
Simplified training wrapper that gives agents easier scenarios to learn from
"""
import sys
import os
sys.path.insert(0, '/Users/shashi/reinforcement_learning')

from env.football_env import FootballEnv
import numpy as np
from collections import defaultdict

class EasyStartEnv(FootballEnv):
    """Modified environment that starts agents in advantageous positions"""
    
    def __init__(self, *args, difficulty='easy', **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty = difficulty
        self.episode_count = 0
        
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        
        # Curriculum: Start easier and gradually increase difficulty
        # REALISTIC: Start in opponent half, not at goal (like real attacking scenario)
        if self.difficulty == 'easy' or self.episode_count < 100:
            # EASY: Agent starts in opponent half with ball (realistic attacking position)
            # Start at midfield to final third (40-60% of field) - needs to dribble forward
            for agent in self.agents:
                team = 0 if 'team_0' in agent else 1
                if team == 0:
                    # Team 0: Start in midfield/opponent half (40-60% of field)
                    # This gives space to practice dribbling and shooting
                    self.agent_positions[agent] = np.array([
                        self.grid_width * 0.4 + np.random.rand() * self.grid_width * 0.2,
                        self.grid_height / 2 + (np.random.rand() - 0.5) * 2
                    ])
                else:
                    # Team 1: Start on defensive side
                    self.agent_positions[agent] = np.array([
                        self.grid_width * 0.3 + np.random.rand() * self.grid_width * 0.1,
                        self.grid_height / 2 + (np.random.rand() - 0.5) * 2
                    ])
            
            # Give ball to team 0 first agent (in opponent half, need to dribble)
            self.ball_possession = 'team_0_agent_0'
            self.ball_position = self.agent_positions['team_0_agent_0'].copy()
            
        elif self.difficulty == 'medium' or self.episode_count < 300:
            # MEDIUM: Start in opponent half
            for agent in self.agents:
                team = 0 if 'team_0' in agent else 1
                if team == 0:
                    self.agent_positions[agent] = np.array([
                        self.grid_width * 0.6 + np.random.rand() * self.grid_width * 0.2,
                        np.random.rand() * self.grid_height
                    ])
                else:
                    self.agent_positions[agent] = np.array([
                        self.grid_width * 0.2 + np.random.rand() * self.grid_width * 0.2,
                        np.random.rand() * self.grid_height
                    ])
        
        self.episode_count += 1
        return self._get_observations(), info

if __name__ == "__main__":
    # Quick test of easy start environment
    env = EasyStartEnv(num_agents_per_team=1, difficulty='easy')
    
    print("Testing Easy Start Environment")
    print("=" * 60)
    
    for episode in range(5):
        obs, info = env.reset()
        print(f"\nEpisode {episode + 1}:")
        
        agent = 'team_0_agent_0'
        pos = env.agent_positions[agent]
        goal = np.array([env.grid_width - 1, env.grid_height // 2])
        dist = np.linalg.norm(pos - goal)
        
        print(f"  Agent position: {pos}")
        print(f"  Goal position: {goal}")
        print(f"  Distance to goal: {dist:.2f}")
        print(f"  Has ball: {env.ball_possession == agent}")
        
        # Try shooting
        if env.ball_possession == agent:
            reward = env._execute_action(agent, 6)  # SHOOT
            print(f"  Shot reward: {reward:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Easy start environment working!")
