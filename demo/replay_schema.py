"""
Replay JSON Schema and Utilities for Multi-Agent RL Football Demo
=================================================================

This module defines the replay format, writer, and reader for recording
and playing back multi-agent RL episodes.
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid


@dataclass
class AgentState:
    """State of a single agent at a timestep."""
    agent_id: str
    team: int
    position: List[float]  # [x, y]
    action: int
    action_name: str
    reward: float
    has_ball: bool
    

@dataclass
class Timestep:
    """Complete state at a single timestep."""
    step: int
    agents: List[Dict[str, Any]]  # List of AgentState dicts
    ball_position: List[float]
    score: List[int]  # [team0_score, team1_score]
    episode_done: bool
    reward_breakdown: Dict[str, float]  # Per-team reward components


@dataclass
class ReplayMetadata:
    """Metadata about the replay."""
    replay_id: str
    timestamp: str
    scenario: str  # "1v1", "2v2", "3v3"
    num_agents: int
    teams: List[int]  # List of team IDs
    agent_names: List[str]
    seed: int
    total_steps: int
    final_score: List[int]
    winner: Optional[int]


class ReplayWriter:
    """Records episode data and writes to JSON."""
    
    def __init__(self, scenario: str = "3v3", seed: int = 42):
        self.replay_id = str(uuid.uuid4())[:8]
        self.scenario = scenario
        self.seed = seed
        self.timesteps: List[Timestep] = []
        self.agent_names: List[str] = []
        self.teams: List[int] = []
        
    def add_timestep(
        self,
        step: int,
        agents_data: List[Dict[str, Any]],
        ball_position: List[float],
        score: List[int],
        episode_done: bool,
        reward_breakdown: Dict[str, float]
    ):
        """Add a timestep to the replay."""
        timestep = Timestep(
            step=step,
            agents=agents_data,
            ball_position=ball_position,
            score=score,
            episode_done=episode_done,
            reward_breakdown=reward_breakdown
        )
        self.timesteps.append(timestep)
        
    def save(self, filepath: str) -> str:
        """Save replay to JSON file."""
        if not self.timesteps:
            raise ValueError("No timesteps recorded")
            
        # Extract metadata from timesteps
        final_step = self.timesteps[-1]
        final_score = final_step.score
        winner = 0 if final_score[0] > final_score[1] else (1 if final_score[1] > final_score[0] else None)
        
        # Get unique agent names and teams
        if self.timesteps:
            first_agents = self.timesteps[0].agents
            self.agent_names = [a['agent_id'] for a in first_agents]
            self.teams = sorted(list(set(a['team'] for a in first_agents)))
        
        metadata = ReplayMetadata(
            replay_id=self.replay_id,
            timestamp=datetime.now().isoformat(),
            scenario=self.scenario,
            num_agents=len(self.agent_names),
            teams=self.teams,
            agent_names=self.agent_names,
            seed=self.seed,
            total_steps=len(self.timesteps),
            final_score=final_score,
            winner=winner
        )
        
        replay_data = {
            "metadata": asdict(metadata),
            "timesteps": [asdict(ts) for ts in self.timesteps]
        }
        
        with open(filepath, 'w') as f:
            json.dump(replay_data, f, indent=2)
            
        return filepath


class ReplayReader:
    """Reads and parses replay JSON files."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.data = json.load(f)
            
        self.metadata = self.data['metadata']
        self.timesteps = self.data['timesteps']
        
    def get_metadata(self) -> Dict[str, Any]:
        """Get replay metadata."""
        return self.metadata
        
    def get_timestep(self, step: int) -> Dict[str, Any]:
        """Get data for a specific timestep."""
        if 0 <= step < len(self.timesteps):
            return self.timesteps[step]
        raise IndexError(f"Timestep {step} out of range [0, {len(self.timesteps)-1}]")
        
    def get_num_timesteps(self) -> int:
        """Get total number of timesteps."""
        return len(self.timesteps)
        
    def iter_timesteps(self):
        """Iterate through all timesteps."""
        for ts in self.timesteps:
            yield ts


def generate_example_replay(scenario: str = "3v3", num_steps: int = 50, seed: int = 42) -> Dict[str, Any]:
    """
    Generate a synthetic example replay with realistic football behavior.
    
    This creates a replay JSON that can be used for demos without a trained model.
    Agents move semi-randomly but with some football-like patterns.
    """
    np.random.seed(seed)
    
    # Parse scenario
    agents_per_team = int(scenario[0])
    total_agents = agents_per_team * 2
    
    # Field dimensions
    field_width = 12
    field_height = 8
    
    # Initialize agents
    agents = []
    for team in [0, 1]:
        for i in range(agents_per_team):
            agent = {
                'agent_id': f'team{team}_agent{i}',
                'team': team,
                'position': [
                    field_width * (0.25 if team == 0 else 0.75),
                    field_height * (i + 1) / (agents_per_team + 1)
                ]
            }
            agents.append(agent)
    
    # Initialize ball
    ball_pos = [field_width / 2, field_height / 2]
    ball_holder = 0  # Index of agent with ball
    
    # Score
    score = [0, 0]
    
    # Action names
    action_names = ['hold', 'up', 'down', 'left', 'right', 'pass', 'shoot']
    
    writer = ReplayWriter(scenario=scenario, seed=seed)
    
    for step in range(num_steps):
        agents_data = []
        total_reward_team0 = 0
        total_reward_team1 = 0
        
        # Update each agent
        for idx, agent in enumerate(agents):
            # Determine action (semi-realistic)
            has_ball = (idx == ball_holder)
            
            if has_ball:
                # Agent with ball: more likely to pass or shoot
                if np.random.rand() < 0.2 and step > 10:
                    action = 6  # shoot
                elif np.random.rand() < 0.3:
                    action = 5  # pass
                else:
                    action = np.random.choice([1, 2, 3, 4])  # move
            else:
                # Agent without ball: move towards ball or goal
                action = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.225, 0.225, 0.225, 0.225])
            
            action_name = action_names[action]
            
            # Update position based on action
            new_pos = agent['position'].copy()
            move_speed = 0.5
            
            if action == 1:  # up
                new_pos[1] = max(0, new_pos[1] - move_speed)
            elif action == 2:  # down
                new_pos[1] = min(field_height, new_pos[1] + move_speed)
            elif action == 3:  # left
                new_pos[0] = max(0, new_pos[0] - move_speed)
            elif action == 4:  # right
                new_pos[0] = min(field_width, new_pos[0] + move_speed)
            
            agent['position'] = new_pos
            
            # Calculate reward (simplified)
            reward = -0.01  # time penalty
            
            # Reward for ball possession
            if has_ball:
                reward += 0.1
                
            # Update ball position
            if has_ball:
                ball_pos = new_pos.copy()
                
                # Handle shooting
                if action == 6:
                    goal_x = field_width if agent['team'] == 0 else 0
                    if abs(new_pos[0] - goal_x) < 2.0:
                        # Goal scored!
                        score[agent['team']] += 1
                        reward += 100
                        # Reset ball
                        ball_pos = [field_width / 2, field_height / 2]
                        ball_holder = (ball_holder + agents_per_team) % total_agents
                        
                # Handle passing
                elif action == 5:
                    # Find teammate
                    teammates = [i for i, a in enumerate(agents) if a['team'] == agent['team'] and i != idx]
                    if teammates:
                        ball_holder = np.random.choice(teammates)
                        reward += 1.0
            
            agents_data.append({
                'agent_id': agent['agent_id'],
                'team': agent['team'],
                'position': agent['position'],
                'action': action,
                'action_name': action_name,
                'reward': reward,
                'has_ball': has_ball
            })
            
            if agent['team'] == 0:
                total_reward_team0 += reward
            else:
                total_reward_team1 += reward
        
        # Reward breakdown
        reward_breakdown = {
            'team0_total': total_reward_team0,
            'team1_total': total_reward_team1,
            'team0_goal': score[0] * 100,
            'team1_goal': score[1] * 100
        }
        
        episode_done = (step == num_steps - 1) or (score[0] >= 3) or (score[1] >= 3)
        
        writer.add_timestep(
            step=step,
            agents_data=agents_data,
            ball_position=ball_pos,
            score=score.copy(),
            episode_done=episode_done,
            reward_breakdown=reward_breakdown
        )
        
        if episode_done:
            break
    
    # Return as dict instead of saving
    metadata = ReplayMetadata(
        replay_id=writer.replay_id,
        timestamp=datetime.now().isoformat(),
        scenario=scenario,
        num_agents=total_agents,
        teams=[0, 1],
        agent_names=[a['agent_id'] for a in agents],
        seed=seed,
        total_steps=len(writer.timesteps),
        final_score=score,
        winner=0 if score[0] > score[1] else (1 if score[1] > score[0] else None)
    )
    
    return {
        "metadata": asdict(metadata),
        "timesteps": [asdict(ts) for ts in writer.timesteps]
    }


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


if __name__ == "__main__":
    # Generate and save example replays
    for scenario in ["1v1", "2v2", "3v3"]:
        replay = generate_example_replay(scenario=scenario, num_steps=100, seed=42)
        replay = convert_numpy_types(replay)  # Convert numpy types
        filepath = f"replays/example_{scenario}.json"
        with open(filepath, 'w') as f:
            json.dump(replay, f, indent=2)
        print(f"Generated {filepath}")
