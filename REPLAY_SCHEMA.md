# Replay Data Schema

This document describes the JSON schema for replay data used by the visualization suite.

## Top-Level Structure

```json
{
  "field_width": 10,
  "field_height": 6,
  "num_agents_per_team": 2,
  "metadata": {
    "episode_id": "optional_id",
    "timestamp": "2024-12-14T10:00:00",
    "model_checkpoint": "runs/run_20241214/episode_500.pt"
  },
  "frames": [
    // Array of frame objects (see below)
  ]
}
```

## Frame Structure

Each frame represents one timestep of the episode:

```json
{
  "frame_idx": 0,
  "agent_positions": {
    "team_0_agent_0": [3.5, 3.0],
    "team_0_agent_1": [2.5, 3.0],
    "team_1_agent_0": [7.5, 3.0],
    "team_1_agent_1": [8.5, 3.0]
  },
  "ball_position": [5.0, 3.0],
  "ball_possession": "team_0_agent_0",
  "stats": {
    "goals_team_0": 0,
    "goals_team_1": 0,
    "shots_team_0": 0,
    "shots_team_1": 0,
    "passes": 0,
    "successful_passes": 0
  },
  "pass_from": "team_0_agent_0",
  "pass_to": "team_0_agent_1",
  "goal_scored": null
}
```

## Field Descriptions

### Top Level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `field_width` | float | Yes | Width of the playing field |
| `field_height` | float | Yes | Height of the playing field |
| `num_agents_per_team` | int | Yes | Number of agents per team |
| `metadata` | object | No | Optional metadata about the episode |
| `frames` | array | Yes | Array of frame objects |

### Frame Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `frame_idx` | int | Yes | Frame index (0-based) |
| `agent_positions` | object | Yes | Dict mapping agent names to [x, y] positions |
| `ball_position` | array | Yes | [x, y] position of the ball |
| `ball_possession` | string or null | No | Name of agent with possession, or null |
| `stats` | object | No | Episode statistics up to this frame |
| `pass_from` | string | No | Agent who made a pass (only in pass frames) |
| `pass_to` | string | No | Agent who received pass (only in pass frames) |
| `goal_scored` | string or null | No | "team_0" or "team_1" if goal scored this frame |

### Agent Positions

Keys are agent names in format: `team_{0|1}_agent_{n}`

Values are `[x, y]` coordinates where:
- `x`: horizontal position (0 = left edge, field_width = right edge)
- `y`: vertical position (0 = top edge, field_height = bottom edge)

### Stats Object

| Field | Type | Description |
|-------|------|-------------|
| `goals_team_0` | int | Goals scored by team 0 |
| `goals_team_1` | int | Goals scored by team 1 |
| `shots_team_0` | int | Shots attempted by team 0 |
| `shots_team_1` | int | Shots attempted by team 1 |
| `passes` | int | Total pass attempts |
| `successful_passes` | int | Successful passes |

## Example Complete Replay

```json
{
  "field_width": 10,
  "field_height": 6,
  "num_agents_per_team": 2,
  "metadata": {
    "episode_id": "episode_001",
    "timestamp": "2024-12-14T10:00:00"
  },
  "frames": [
    {
      "frame_idx": 0,
      "agent_positions": {
        "team_0_agent_0": [2.5, 3.0],
        "team_0_agent_1": [1.5, 2.5],
        "team_1_agent_0": [7.5, 3.5],
        "team_1_agent_1": [8.5, 3.0]
      },
      "ball_position": [5.0, 3.0],
      "ball_possession": null,
      "stats": {
        "goals_team_0": 0,
        "goals_team_1": 0
      }
    },
    {
      "frame_idx": 1,
      "agent_positions": {
        "team_0_agent_0": [2.8, 3.0],
        "team_0_agent_1": [1.7, 2.5],
        "team_1_agent_0": [7.2, 3.5],
        "team_1_agent_1": [8.3, 3.0]
      },
      "ball_position": [5.0, 3.0],
      "ball_possession": null,
      "stats": {
        "goals_team_0": 0,
        "goals_team_1": 0
      }
    },
    {
      "frame_idx": 2,
      "agent_positions": {
        "team_0_agent_0": [3.2, 3.0],
        "team_0_agent_1": [2.0, 2.5],
        "team_1_agent_0": [6.8, 3.5],
        "team_1_agent_1": [8.0, 3.0]
      },
      "ball_position": [5.0, 3.0],
      "ball_possession": "team_0_agent_0",
      "stats": {
        "goals_team_0": 0,
        "goals_team_1": 0
      }
    },
    {
      "frame_idx": 3,
      "agent_positions": {
        "team_0_agent_0": [3.6, 3.2],
        "team_0_agent_1": [2.4, 2.7],
        "team_1_agent_0": [6.5, 3.4],
        "team_1_agent_1": [7.8, 3.1]
      },
      "ball_position": [3.6, 3.2],
      "ball_possession": "team_0_agent_0",
      "pass_from": "team_0_agent_0",
      "pass_to": "team_0_agent_1",
      "stats": {
        "goals_team_0": 0,
        "goals_team_1": 0,
        "passes": 1,
        "successful_passes": 1
      }
    }
  ]
}
```

## Creating Replay Data from Environment

Here's how to collect replay data during environment execution:

```python
from env.improved_football_env import ImprovedFootballEnv
import json

def collect_replay(env, num_steps=100):
    """Collect replay data from environment"""
    replay = {
        'field_width': env.grid_width,
        'field_height': env.grid_height,
        'num_agents_per_team': env.num_agents_per_team,
        'frames': []
    }
    
    obs, info = env.reset()
    
    for step in range(num_steps):
        # Get current state
        frame = {
            'frame_idx': step,
            'agent_positions': {k: v.tolist() for k, v in env.agent_positions.items()},
            'ball_position': env.ball_position.tolist(),
            'ball_possession': env.ball_possession,
            'stats': env.episode_stats.copy()
        }
        
        replay['frames'].append(frame)
        
        # Take random action
        agent = env.agent_selection
        action = env.action_space(agent).sample()
        obs, rewards, terms, truncs, infos = env.step(action)
        
        if any(terms.values()) or any(truncs.values()):
            break
    
    return replay

# Example usage
env = ImprovedFootballEnv()
replay = collect_replay(env)

# Save to file
with open('replay.json', 'w') as f:
    json.dump(replay, f, indent=2)
```

## Notes

- All positions are in continuous coordinates (floats)
- Coordinate system: (0, 0) is top-left, (field_width, field_height) is bottom-right
- Agent names must follow format: `team_{team_id}_agent_{agent_id}`
- Pass events are indicated by `pass_from` and `pass_to` fields in the frame
- Goal events can be indicated by `goal_scored` field or by checking stats difference
- Frames should be ordered chronologically by `frame_idx`
