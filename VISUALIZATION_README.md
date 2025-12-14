# Visualization Suite for Multi-Agent Football RL

## 📊 Overview

Complete visualization system with 6 high-impact visualizations for analyzing multi-agent football reinforcement learning experiments.

## 🎨 Visualizations

### 1. Movement Trails
**Function**: `plot_movement_trails(replay, save_path)`

Shows agent trajectories over entire episode with directional arrows.

**Features**:
- Color-coded by team (blue/red)
- Direction arrows every N steps
- Start/end markers
- Ball trail overlay

**Use cases**:
- Analyze movement patterns
- Identify coordination strategies
- Debug navigation issues

---

### 2. Pass Network Overlay
**Function**: `plot_pass_network(replay, save_path)`

Network diagram showing pass connections between agents.

**Features**:
- Nodes = agents at average positions
- Directed edges = passes
- Arrow thickness = pass frequency
- Pass count labels

**Use cases**:
- Evaluate team coordination
- Identify key playmakers
- Analyze passing strategies

---

### 3. Positional Heatmaps
**Function**: `plot_heatmaps(replay, save_dir)`

2D heatmaps showing spatial distribution.

**Generates**:
- Team 0 agent heatmap
- Team 1 agent heatmap
- Ball heatmap

**Use cases**:
- Identify territorial control
- Analyze positioning strategies
- Find hot zones

---

### 4. Episode Replay GIF
**Function**: `make_replay_gif(replay, save_path)`

Animated replay of entire episode.

**Features**:
- Frame-by-frame animation
- Movement trails (optional)
- Score display
- Customizable FPS

**Use cases**:
- Visualize complete episodes
- Create demos for presentations
- Debug specific game situations

---

### 5. Before/After Comparison GIF
**Function**: `make_before_after_gif(replay_before, replay_after, save_path)`

Side-by-side comparison of two episodes.

**Features**:
- Synchronized playback
- Dual-panel layout
- Independent statistics
- Highlight differences

**Use cases**:
- Compare training progress
- Evaluate model improvements
- Demonstrate learning outcomes

---

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install matplotlib numpy imageio seaborn

# Optional: for better pass network visualization
pip install networkx
```

### Basic Usage

```python
from visualization_utils import (
    plot_movement_trails,
    plot_pass_network,
    plot_heatmaps,
    make_replay_gif,
    make_before_after_gif,
    load_replay_from_json
)

# Load replay data
replay = load_replay_from_json('replay.json')

# Generate visualizations
plot_movement_trails(replay, 'trails.png')
plot_pass_network(replay, 'network.png')
plot_heatmaps(replay, 'heatmaps/')
make_replay_gif(replay, 'replay.gif', fps=10)
```

### Run Example

```bash
# Generate all visualizations with synthetic data
python example_usage.py
```

This creates:
- `visualization_outputs/movement_trails.png`
- `visualization_outputs/pass_network.png`
- `visualization_outputs/heatmaps/all_heatmaps.png`
- `visualization_outputs/replay.gif`
- `visualization_outputs/comparison.gif`

---

## 📋 Replay Data Format

Visualizations expect replay data in JSON format:

```json
{
  "field_width": 10,
  "field_height": 6,
  "num_agents_per_team": 2,
  "frames": [
    {
      "frame_idx": 0,
      "agent_positions": {
        "team_0_agent_0": [3.5, 3.0],
        "team_1_agent_0": [7.5, 3.0]
      },
      "ball_position": [5.0, 3.0],
      "ball_possession": "team_0_agent_0",
      "stats": {
        "goals_team_0": 0,
        "goals_team_1": 0
      },
      "pass_from": "team_0_agent_0",
      "pass_to": "team_0_agent_1"
    }
  ]
}
```

See `REPLAY_SCHEMA.md` for complete documentation.

---

## 🔧 Collecting Replay Data

### From Environment

```python
from env.improved_football_env import ImprovedFootballEnv
import json

def collect_replay(env, num_steps=100):
    """Collect replay data during episode"""
    replay = {
        'field_width': env.grid_width,
        'field_height': env.grid_height,
        'num_agents_per_team': env.num_agents_per_team,
        'frames': []
    }
    
    obs, info = env.reset()
    
    for step in range(num_steps):
        frame = {
            'frame_idx': step,
            'agent_positions': {
                k: v.tolist() for k, v in env.agent_positions.items()
            },
            'ball_position': env.ball_position.tolist(),
            'ball_possession': env.ball_possession,
            'stats': env.episode_stats.copy()
        }
        
        replay['frames'].append(frame)
        
        # Execute action
        agent = env.agent_selection
        action = get_action(obs[agent])  # Your policy
        obs, rewards, terms, truncs, infos = env.step(action)
        
        if any(terms.values()) or any(truncs.values()):
            break
    
    return replay

# Save to file
env = ImprovedFootballEnv()
replay = collect_replay(env)
with open('replay.json', 'w') as f:
    json.dump(replay, f, indent=2)
```

### From Training Loop

Add replay collection to your training script:

```python
def train_with_replay_capture(trainer, capture_every=50):
    """Train and capture replays periodically"""
    
    for episode in range(num_episodes):
        # Regular training
        episode_data = trainer.collect_episode()
        
        # Capture replay periodically
        if episode % capture_every == 0:
            replay = collect_replay(trainer.env)
            save_replay_to_json(
                replay, 
                f'replays/episode_{episode}.json'
            )
```

---

## 📊 Advanced Usage

### Customization Options

#### Movement Trails

```python
plot_movement_trails(
    replay,
    save_path='trails.png',
    arrow_interval=15,      # Arrows every 15 steps
    figsize=(14, 10)        # Larger figure
)
```

#### Pass Network

```python
plot_pass_network(
    replay,
    save_path='network.png',
    figsize=(16, 10)        # Wide layout
)
```

#### Heatmaps

```python
plot_heatmaps(
    replay,
    save_dir='heatmaps/',
    figsize=(20, 6)         # Extra wide
)
```

#### Replay GIF

```python
make_replay_gif(
    replay,
    save_path='replay.gif',
    fps=15,                 # Faster playback
    show_trails=True,       # Show recent trails
    trail_length=30         # Longer trails
)
```

#### Comparison GIF

```python
make_before_after_gif(
    replay_before,
    replay_after,
    save_path='comparison.gif',
    fps=12,
    show_trails=False       # Cleaner view
)
```

---

## 🎬 Example Workflows

### 1. Training Progress Analysis

```python
# Collect replays at different training stages
replays = {
    'untrained': collect_replay(env, model_episode_0),
    'early': collect_replay(env, model_episode_100),
    'mid': collect_replay(env, model_episode_500),
    'final': collect_replay(env, model_episode_1000)
}

# Compare early vs final
make_before_after_gif(
    replays['early'],
    replays['final'],
    'training_progress.gif'
)

# Analyze movement evolution
for stage, replay in replays.items():
    plot_movement_trails(replay, f'trails_{stage}.png')
```

### 2. Strategy Comparison

```python
# Compare different training approaches
replay_ppo = load_replay_from_json('ppo_model.json')
replay_dqn = load_replay_from_json('dqn_model.json')

# Visual comparison
make_before_after_gif(replay_ppo, replay_dqn, 'algorithm_comparison.gif')

# Heatmap comparison
plot_heatmaps(replay_ppo, 'heatmaps_ppo/')
plot_heatmaps(replay_dqn, 'heatmaps_dqn/')
```

### 3. Behavior Analysis

```python
# Analyze specific behavior
replay = load_replay_from_json('interesting_episode.json')

# Movement patterns
plot_movement_trails(replay, 'movement_analysis.png', arrow_interval=5)

# Positioning strategy
plot_heatmaps(replay, 'positioning_analysis/')

# Team coordination
plot_pass_network(replay, 'coordination_analysis.png')
```

---

## 🎯 Tips & Best Practices

### Performance

- **GIF generation**: Takes ~1-2 seconds per frame. For 100-frame episodes:
  - Single replay GIF: ~2 minutes
  - Comparison GIF: ~3-4 minutes

- **Memory**: Each frame stores full state. For long episodes (>500 frames), consider:
  - Sampling every Nth frame
  - Splitting into multiple shorter replays
  - Compressing JSON files

### Quality

- **Figure size**: Larger figures (14-16 width) work better for presentations
- **DPI**: Use 150-300 for publication quality
- **Colors**: Dark theme (#1a1a1a background) works well for demos
- **FPS**: 10-15 fps is good balance between smoothness and file size

### Data Collection

- **Frequency**: Capture replays every 25-50 episodes
- **Selection**: Save best/worst/median episodes for comparison
- **Metadata**: Include training step, model checkpoint, hyperparameters
- **Storage**: Compress JSON files or use binary format for large datasets

---

## 📁 File Structure

```
reinforcement_learning/
├── visualization_utils.py       # Main visualization functions
├── example_usage.py             # Example with synthetic data
├── REPLAY_SCHEMA.md            # Data format documentation
├── visualization_outputs/       # Generated visualizations
│   ├── movement_trails.png
│   ├── pass_network.png
│   ├── heatmaps/
│   │   └── all_heatmaps.png
│   ├── replay.gif
│   ├── comparison.gif
│   ├── replay_before.json
│   └── replay_after.json
└── replays/                     # Your replay data (create as needed)
    ├── episode_0.json
    ├── episode_100.json
    └── ...
```

---

## 🐛 Troubleshooting

### "networkx not installed"
Pass network will use fallback matplotlib implementation. Install with:
```bash
pip install networkx
```

### GIFs are too large
Reduce file size:
```python
make_replay_gif(replay, 'replay.gif', fps=8)  # Lower FPS
# Or subsample frames in replay data
```

### Out of memory
For very long episodes:
```python
# Sample every 2nd frame
sampled_frames = replay['frames'][::2]
replay_sampled = {**replay, 'frames': sampled_frames}
```

### Matplotlib backend issues
If you get display errors:
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
```

---

## 📚 Dependencies

- **Required**:
  - `matplotlib >= 3.5.0`
  - `numpy >= 1.21.0`
  - `imageio >= 2.9.0`
  - `seaborn >= 0.11.0`

- **Optional**:
  - `networkx >= 2.6.0` (better pass networks)

---

## 🎓 Citation

If you use these visualizations in your research, please cite:

```bibtex
@misc{football_rl_viz,
  title={Visualization Suite for Multi-Agent Football RL},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/repo}
}
```

---

## 📄 License

MIT License - feel free to use and modify for your projects.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- 3D trajectory visualizations
- Real-time streaming visualizations
- Interactive dashboards (plotly/dash)
- Video export (mp4) in addition to GIF
- Statistical overlays (velocity vectors, acceleration)

---

## ✅ Checklist

- [x] Movement trails visualization
- [x] Pass network diagram
- [x] Positional heatmaps
- [x] Episode replay GIF
- [x] Before/after comparison GIF
- [x] Complete documentation
- [x] Example usage script
- [x] Replay schema specification
- [x] Synthetic data generator
- [x] All functions tested

**Status**: ✅ Complete and ready to use!
