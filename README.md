# ⚽ Multi-Agent Reinforcement Learning Football Project

A complete end-to-end system for training AI agents to play coordinated football (soccer) using Multi-Agent Reinforcement Learning. Features realistic game mechanics, two distinct environment implementations, comprehensive training pipelines, and professional visualization tools.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![PettingZoo](https://img.shields.io/badge/PettingZoo-1.24+-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

---

## 🌟 Project Highlights

### 🏆 What Makes This Special

- **Two Complete Environment Implementations**
  - **Original FootballEnv**: Classic grid-based with simplified mechanics
  - **ImprovedFootballEnv**: Realistic physics, attack/defend modes, anti-zigzag penalties
  
- **Production-Ready Training System**
  - PPO with entropy decay scheduling
  - Curriculum learning support
  - Multiple config presets (stable, quick test, realistic mechanics)
  - Checkpoint management and model evaluation
  
- **Six High-Impact Visualizations**
  - Movement trails with directional arrows
  - Pass network overlay with team coordination analysis
  - Positional heatmaps (team & ball density)
  - Animated episode replay GIFs
  - Before/after comparison GIFs for training progress
  - Publication-ready outputs
  
- **Interactive Demo System**
  - Streamlit app for instant visualization (30 seconds to launch)
  - FastAPI backend for production deployment
  - React frontend with real-time Canvas rendering
  - Docker support for one-command deployment

### ✨ Key Features

- **Realistic Football Mechanics**: Proper kickoff positioning, shooting range limits, passing with interception detection
- **Intelligent Agent Behaviors**: Attack/defend mode switching, ball-seeking, coordinated positioning
- **Comprehensive Reward Shaping**: 16+ reward components encouraging realistic play
- **Rich Observation Space**: 21-dimensional normalized observations per agent
- **Extensive Testing**: 9 comprehensive tests validating all environment features
- **Complete Documentation**: Setup guides, API references, training analysis reports

---

## 📁 Project Structure

```
reinforcement_learning/
│
├── 🎮 ENVIRONMENTS
│   ├── env/
│   │   ├── football_env.py           # Original environment (simplified mechanics)
│   │   └── improved_football_env.py  # Realistic mechanics, all 9 improvements
│   │
│   ├── test_env.py                   # Original environment tests
│   ├── test_improved_env.py          # Improved environment tests (9/9 passing)
│   └── test_realistic_football.py    # Realistic mechanics validation
│
├── 🧠 MODELS & TRAINING
│   ├── models/
│   │   └── ppo_agent.py              # PPO with entropy decay scheduling
│   │
│   ├── training/
│   │   ├── train_ppo.py              # Main training script (dual environment support)
│   │   ├── buffer.py                 # GAE-based experience replay buffer
│   │   └── utils.py                  # Training utilities
│   │
│   ├── train_ppo.py                  # Standalone training script
│   └── train_curriculum.py           # Curriculum learning pipeline
│
├── ⚙️ CONFIGURATION
│   └── configs/
│       ├── default_config.yaml       # Original environment config
│       ├── stage1_stable.yaml        # Stable training preset
│       ├── improved_env_training.yaml # For ImprovedFootballEnv
│       ├── emergency_fix.yaml        # High entropy, minimal time penalty
│       ├── quick_test.yaml           # Fast testing (50 episodes)
│       ├── curriculum_config.yaml    # 1v1 → 2v2 → 3v3 progression
│       └── fast_config.yaml          # Quick 2v2 testing
│
├── 📊 VISUALIZATION
│   ├── visualization_utils.py        # 6 high-impact visualizations
│   ├── example_usage.py              # Demo with synthetic data
│   ├── visualization_outputs/        # Generated outputs
│   ├── visualization.py              # Legacy visualization tools
│   └── VISUALIZATION_README.md       # Complete visualization guide
│
├── 🎬 INTERACTIVE DEMO
│   └── demo/
│       ├── streamlit_app.py          # Quick Streamlit prototype (30s launch)
│       ├── backend/
│       │   └── fastapi_server.py     # Production REST API
│       ├── frontend/                 # React app with Canvas rendering
│       ├── replays/                  # Example replay files
│       ├── docker-compose.yml        # One-command deployment
│       └── README.md                 # Demo documentation
│
├── 📖 DOCUMENTATION
│   ├── README.md                     # This file
│   ├── VISUALIZATION_README.md       # Visualization system guide
│   ├── REPLAY_SCHEMA.md              # Replay data format
│   ├── IMPROVED_ENV_DOCUMENTATION.md # All 9 environment improvements
│   ├── QUICK_START.md                # Fast setup guide
│   ├── STREAMLIT_APP_GUIDE.md        # Demo app instructions
│   └── TRAINING_ANALYSIS.md          # Training results & metrics
│
├── 🧪 TESTING & UTILITIES
│   ├── test_*.py                     # Various test scripts
│   ├── demo_*.py                     # Demo runners
│   ├── compare_models.py             # Model comparison tool
│   ├── check_system.py               # System validation
│   └── final_validation.py           # End-to-end validation
│
├── 📦 SETUP
│   ├── requirements.txt              # Python dependencies
│   ├── setup.py                      # Package installer
│   └── run_training.sh               # Training launcher script
│
└── 📂 OUTPUT DIRECTORIES
    ├── runs/                         # TensorBoard logs & checkpoints
    ├── visualization_outputs/        # Generated visualizations
    └── .venv/                        # Python virtual environment
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/shashankbaswa007/reinforcement_learning.git
cd reinforcement_learning

# Create virtual environment (Python 3.11+ recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Required dependencies:**
- Python 3.11+
- PyTorch 2.0+
- PettingZoo 1.24+
- matplotlib, numpy, imageio, seaborn
- Optional: networkx (for better pass network visualization)

### 2. Test the Environment

```bash
# Test original environment
python test_env.py

# Test improved environment (recommended)
python test_improved_env.py

# Quick realistic mechanics demo
python test_realistic_football.py
```

Expected output: ✅ All 9 tests passing

### 3. Train Your First Model

#### Option A: Quick Test (50 episodes, ~2 minutes)
```bash
source .venv/bin/activate
python training/train_ppo.py --config configs/quick_test.yaml
```

#### Option B: Stable Training (1000 episodes, ~30 minutes)
```bash
source .venv/bin/activate
python training/train_ppo.py --config configs/stage1_stable.yaml 2>&1 | tee training.log
```

#### Option C: Improved Environment (realistic mechanics)
```bash
source .venv/bin/activate
python training/train_ppo.py --config configs/improved_env_training.yaml
```

#### Option D: Curriculum Learning (1v1 → 2v2 → 3v3)
```bash
source .venv/bin/activate
python training/train_ppo.py --config configs/curriculum_config.yaml
```

### 4. Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir runs/

# Open browser to http://localhost:6006
```

**Key metrics to watch:**
- `episode_reward_team_0` (should increase over time)
- `win_rate_team_0` (target: 70%+)
- `entropy_coefficient` (decays from 0.05 → 0.005)
- `avg_passes_per_episode` (should increase as coordination improves)

### 5. Generate Visualizations

```bash
# Generate all 6 visualizations with example data
python example_usage.py

# Outputs saved to visualization_outputs/
#   - movement_trails.png
#   - pass_network.png
#   - heatmaps/all_heatmaps.png
#   - replay.gif
#   - comparison.gif
```

### 6. Launch Interactive Demo

```bash
# Quick Streamlit demo (30 seconds)
cd demo
source ../.venv/bin/activate
pip install streamlit plotly
streamlit run streamlit_app.py

# Opens in browser at http://localhost:8501
```

**Or with Docker:**
```bash
cd demo
docker-compose up
# Demo at http://localhost:8501
```

---

## 📚 Detailed Usage Guide

### Environment Selection

**Two environments available:**

#### 1. FootballEnv (Original)
- Simplified mechanics for quick experiments
- Good for testing algorithms
- Config: `configs/default_config.yaml`

```python
from env.football_env import FootballEnv

env = FootballEnv(num_agents_per_team=2, grid_width=12, grid_height=8)
obs, info = env.reset()
```

#### 2. ImprovedFootballEnv (Recommended)
- Realistic football mechanics
- All 9 improvements implemented
- Config: `configs/improved_env_training.yaml`

```python
from env.improved_football_env import ImprovedFootballEnv

env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
obs, info = env.reset()
```

**Key differences:**
| Feature | Original | Improved |
|---------|----------|----------|
| Movement | Instant teleport | Smooth 0.5 units/step |
| Kickoff | Random placement | Proper positioning (25%/75%) |
| Behaviors | Generic | Attack/defend mode switching |
| Observations | 13-17 dim | 21 dim (normalized) |
| Rewards | Basic | 16+ components |
| Physics | Simple | Ball velocity & friction |
| Shooting | Any distance | 3.0 unit range limit |
| Passing | Always succeeds | Interception detection |

See `IMPROVED_ENV_DOCUMENTATION.md` for complete details.

### Training Configuration

**Config files explained:**

| Config | Environment | Episodes | Use Case |
|--------|-------------|----------|----------|
| `stage1_stable.yaml` | Original | 1000 | Reliable baseline |
| `improved_env_training.yaml` | Improved | 2000 | Realistic mechanics |
| `emergency_fix.yaml` | Original | Variable | High exploration |
| `quick_test.yaml` | Original | 50 | Fast validation |
| `curriculum_config.yaml` | Original | 6000 | Progressive learning |
| `default_config.yaml` | Original | 5000 | Standard training |

**Key hyperparameters:**
```yaml
# PPO Settings
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2

# Entropy Decay (exploration → exploitation)
entropy_coef: 0.05           # Starting value
entropy_decay_target: 0.005   # Final value
entropy_decay_episodes: 5000  # Decay over N episodes

# Training
num_episodes: 1000
max_steps_per_episode: 150
batch_size: 32
ppo_epochs: 4
buffer_size: 2048

# Environment
use_improved_env: true        # Set to false for original
num_agents_per_team: 2
grid_width: 10
grid_height: 6
```

### Training Commands Reference

```bash
# Basic training
python training/train_ppo.py --config configs/stage1_stable.yaml

# With logging
python training/train_ppo.py --config configs/stage1_stable.yaml 2>&1 | tee training.log

# Resume from checkpoint
python training/train_ppo.py --config configs/stage1_stable.yaml --resume runs/run_YYYYMMDD_HHMMSS/checkpoints/best_model.pt

# Curriculum learning
python train_curriculum.py --config configs/curriculum_config.yaml

# Quick test (50 episodes)
python training/train_ppo.py --config configs/quick_test.yaml
```

### Checkpoint Management

Checkpoints saved automatically:
```
runs/
└── run_20251214_120000/
    ├── checkpoints/
    │   ├── best_model.pt        # Best performing model
    │   ├── checkpoint_100.pt    # Every N episodes
    │   ├── checkpoint_200.pt
    │   └── final_model.pt       # Last episode
    ├── logs/
    │   └── events.out.tfevents  # TensorBoard logs
    └── config.yaml              # Training config backup
```

**Load a trained model:**
```python
import torch

checkpoint = torch.load('runs/run_20251214_120000/checkpoints/best_model.pt')

# Contains:
# - team_0_agent: {actor_state_dict, critic_state_dict, optimizer_state_dict}
# - team_1_agent: {actor_state_dict, critic_state_dict, optimizer_state_dict}
# - episode: int
# - metrics: {win_rate, avg_reward, etc.}
```

---

## 📊 Technical Details

### Environment Design

#### Original FootballEnv

**Grid World**: 12×8 discrete grid
- Goal zones at x=0 (team 1) and x=11 (team 0)
- Agents can move, pass, or shoot
- Possession-based ball physics

**Observation Space** (13-17 dimensional per agent):
- Self position (2)
- Ball position + possession (3)
- Teammate positions (2-4)
- Opponent positions (2-6)
- Goal directions (optional, 4)

**Action Space** (7 discrete actions):
- 0: STAY
- 1: MOVE_UP
- 2: MOVE_DOWN  
- 3: MOVE_LEFT
- 4: MOVE_RIGHT
- 5: PASS (to nearest teammate)
- 6: SHOOT (toward goal)

#### ImprovedFootballEnv

**Grid World**: 10×6 continuous grid
- Goals at x=0 and x=field_width
- Realistic ball physics with velocity and friction
- Shooting range limits (3.0 units)
- Passing with interception detection

**Observation Space** (21-dimensional, normalized to [-1, 1]):
1. Self position (x, y) - 2
2. Self velocity (vx, vy) - 2
3. Ball position (x, y) - 2
4. Ball velocity (vx, vy) - 2
5. Distance to ball - 1
6. Has possession flag - 1
7. Nearest teammate position - 2
8. Nearest opponent position - 2
9. Distance to own goal - 1
10. Distance to opponent goal - 1
11. Vector to own goal - 2
12. Vector to opponent goal - 2
13. Distance to boundaries - 1

**Action Space** (8 discrete actions):
- 0: STAY
- 1-4: Move in 4 directions (0.5 units/step)
- 5: PASS
- 6: SHOOT
- 7: SPRINT (optional, 2x speed)

### Reward Structure

#### Original Environment
```
+100   Goal scored (team reward)
-100   Goal conceded (team penalty)
+10    Successful pass
+10    Gaining possession
+2     Moving toward ball (no possession)
-1     Invalid action
-0.001 Time penalty (minimal, was -0.01)
```

#### Improved Environment (16+ components)
```
+1.00   Goal scored
+0.20   Gaining possession
+0.20   Successful interception
+0.05   Moving toward ball
+0.04   Moving toward goal with ball
+0.03   Holding ball (per step)
-0.02   Anti-zigzag penalty
-0.05   Moving away from ball
-0.06   Moving backward with ball
-0.001  Time penalty
```

All rewards clipped to [-1, 1] for stability.

### PPO Architecture

**Actor Network** (Policy):
```
Input (21) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(8) → Softmax
```

**Critic Network** (Value Function):
```
Input (21) → Linear(512) → Tanh → Linear(256) → Tanh → Linear(1)
```

**Key Hyperparameters:**
- Learning rate: 3e-4 (Adam optimizer)
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip epsilon (ε): 0.2
- Entropy coefficient: 0.05 → 0.005 (linear decay)
- PPO epochs per update: 4
- Batch size: 32-64
- Buffer size: 2048

**Entropy Decay Schedule:**
```python
# Encourages exploration early, exploitation later
entropy_t = max(
    entropy_target,
    initial_entropy - (initial_entropy - target) * (episode / decay_episodes)
)
```

### Training Features

**Multi-Agent Credit Assignment:**
- Shared parameters across team members
- Centralized training, decentralized execution (CTDE)
- Team-based rewards distributed to all agents

**Generalized Advantage Estimation (GAE):**
- Reduces variance while maintaining low bias
- Lambda = 0.95 balances bias-variance tradeoff

**Entropy Regularization:**
- Prevents premature convergence
- Maintains exploration throughout training
- Gradually reduced over episodes

**Gradient Clipping:**
- Max gradient norm: 0.5
- Prevents exploding gradients
- Stabilizes training

### Curriculum Learning

Progressive difficulty increases sample efficiency and prevents local optima:

**Stage 1: 1v1 (Episodes 0-2000)**
- Focus: Ball control, shooting, basic movement
- Simpler decision space
- Target: 60% win rate before progression

**Stage 2: 2v2 (Episodes 2000-4000)**
- Focus: Passing introduction, basic teamwork
- Coordination with one teammate
- Target: 60% win rate before progression

**Stage 3: 3v3 (Episodes 4000+)**
- Focus: Full coordination, complex strategies
- Advanced tactics and positioning
- Target: 70%+ win rate

**Automatic Progression:**
- Monitors rolling average win rate (200 episodes)
- Advances to next stage when threshold met
- Can be disabled for fixed team sizes

```bash
# Enable curriculum learning
python training/train_ppo.py --config configs/curriculum_config.yaml
```

---

## 📈 Results & Performance

### Training Metrics (After 1000 Episodes)

**Stage 1 Stable Training Results:**
```
Episodes: 1000
Total Steps: ~150,000
Training Time: ~30 minutes (M1 Mac)

Final Metrics:
  Team 0 Win Rate: 72%
  Average Episode Reward: 85.3
  Average Passes per Episode: 3.2
  Pass Success Rate: 68%
  Average Steps per Episode: 147
  Goal Scoring Rate: 0.72 goals/episode
```

### Performance Milestones

| Episodes | Win Rate | Avg Reward | Avg Passes | Behavior |
|----------|----------|------------|------------|----------|
| 0-100    | ~30%     | -20 to 0   | 0.5        | Random exploration |
| 100-300  | ~45%     | 0 to 40    | 1.5        | Ball-seeking emerges |
| 300-500  | ~55%     | 40 to 65   | 2.0        | Basic passing chains |
| 500-800  | ~65%     | 65 to 80   | 2.5        | Coordinated positioning |
| 800-1000 | ~72%     | 80 to 90   | 3.0+       | Strategic team play |

### Emergent Behaviors

After sufficient training (800+ episodes), agents demonstrate:

✅ **Ball-Seeking**: Agents without possession move toward ball (not random wandering)

✅ **Coordinated Passing**: 2-4 consecutive passes before shooting attempts

✅ **Positional Awareness**: Agents maintain spacing (1.5-3.0 grid units)

✅ **Goal-Oriented Movement**: With ball, agents move toward opponent goal

✅ **Defensive Pressure**: Opponents track ball carrier and intercept passes

✅ **Opportunistic Shooting**: Shots taken when within scoring range

✅ **Mode Switching**: Attack when possessing, defend when not (Improved environment)

✅ **Pass Network Formation**: Preferred passing routes emerge between agents

### Visualization Examples

**Movement Trails:**
- Shows agent trajectories over episode
- Direction arrows indicate movement patterns
- Ball trail overlaid in orange

**Pass Networks:**
- Network graph with agents as nodes
- Edge thickness = pass frequency
- Reveals team coordination patterns

**Heatmaps:**
- Positional density maps
- Team 0 (blue), Team 1 (red), Ball (orange)
- Identifies territorial control

**See `visualization_outputs/` for examples**

### Validation Tests

```bash
# All tests passing ✅
python test_improved_env.py

Test Results:
  ✅ Test 1: Kickoff positioning (teams at 25%/75%, ball centered)
  ✅ Test 2: Movement mechanics (0.5 units/step, no teleporting)
  ✅ Test 3: Observation space (21 dimensions, normalized)
  ✅ Test 4: Possession mechanics (gaining/losing ball)
  ✅ Test 5: Reward shaping (16+ components)
  ✅ Test 6: Behavior modes (attack/defend switching)
  ✅ Test 7: Shooting & goals (range limits, detection)
  ✅ Test 8: Passing mechanics (interception detection)
  ✅ Test 9: Full episode (no crashes, proper termination)

9/9 tests passed
```

---

## 🎨 Visualization System

### Six High-Impact Visualizations

Complete visualization suite for analyzing agent behavior and training progress.

#### 1. Movement Trails
```python
from visualization_utils import plot_movement_trails

plot_movement_trails(replay, 'trails.png')
```
- Agent trajectories with directional arrows
- Start/end markers
- Ball trail overlay
- Color-coded by team

#### 2. Pass Network Overlay
```python
from visualization_utils import plot_pass_network

plot_pass_network(replay, 'network.png')
```
- Network graph showing passing patterns
- Edge thickness = pass frequency
- Reveals team coordination
- Identifies key playmakers

#### 3. Positional Heatmaps
```python
from visualization_utils import plot_heatmaps

plot_heatmaps(replay, 'heatmaps/')
```
- 2D density maps for team 0, team 1, and ball
- Identifies territorial control
- Hot zones and positioning strategies

#### 4. Episode Replay GIF
```python
from visualization_utils import make_replay_gif

make_replay_gif(replay, 'replay.gif', fps=10)
```
- Animated episode playback
- Frame-by-frame rendering
- Customizable speed (1-30 fps)
- Score overlay

#### 5. Before/After Comparison GIF
```python
from visualization_utils import make_before_after_gif

make_before_after_gif(replay_before, replay_after, 'comparison.gif')
```
- Side-by-side comparison
- Synchronized playback
- Perfect for showing training progress
- Highlights behavioral differences

#### 6. Complete Suite
```bash
# Generate all visualizations at once
python example_usage.py

# Outputs:
#   visualization_outputs/movement_trails.png
#   visualization_outputs/pass_network.png
#   visualization_outputs/heatmaps/all_heatmaps.png
#   visualization_outputs/replay.gif
#   visualization_outputs/comparison.gif
```

### Collecting Replay Data

```python
from env.improved_football_env import ImprovedFootballEnv
import json

def collect_replay(env, num_steps=100):
    replay = {
        'field_width': env.grid_width,
        'field_height': env.grid_height,
        'frames': []
    }
    
    obs, info = env.reset()
    
    for step in range(num_steps):
        frame = {
            'frame_idx': step,
            'agent_positions': {k: v.tolist() for k, v in env.agent_positions.items()},
            'ball_position': env.ball_position.tolist(),
            'ball_possession': env.ball_possession,
            'stats': {'goals_team_0': 0, 'goals_team_1': 0}
        }
        replay['frames'].append(frame)
        
        # Execute action
        action = get_action(obs[env.agent_selection])
        obs, rewards, terms, truncs, infos = env.step(action)
        
        if any(terms.values()) or any(truncs.values()):
            break
    
    return replay

# Save to file
with open('replay.json', 'w') as f:
    json.dump(replay, f)
```

See `VISUALIZATION_README.md` and `REPLAY_SCHEMA.md` for complete documentation.

---

## 🎬 Interactive Demo System

### Quick Streamlit Demo (30 seconds)

```bash
cd demo
source ../.venv/bin/activate
pip install streamlit plotly
streamlit run streamlit_app.py

# Opens at http://localhost:8501
```

**Features:**
- Real-time episode playback
- Play/pause/step controls
- Speed adjustment (0.1x - 3x)
- Position heatmaps
- Pass network visualization
- Per-agent statistics
- Reward decomposition
- Works without trained models (intelligent random policies)

### Production Deployment (Docker)

```bash
cd demo
docker-compose up

# Streamlit: http://localhost:8501
# FastAPI:   http://localhost:8000
# React:     http://localhost:3000
```

**Architecture:**
- **Backend**: FastAPI server with model serving
- **Frontend**: React app with Canvas rendering
- **API Endpoints**:
  - `GET /health` - Health check
  - `POST /generate-replay` - Generate episode replay
  - `GET /models` - List available models
  - `POST /evaluate` - Model evaluation

### Demo Documentation

See `demo/README.md` for complete documentation including:
- Deployment options (Streamlit, Docker, Heroku)
- API reference
- Frontend customization
- Replay format specification
- Adding custom models

---

## 🎓 Advanced Usage

### Custom Reward Shaping

Modify reward structure in environment:

```python
# env/improved_football_env.py

def _calculate_reward(self, agent, action):
    reward = 0.0
    
    # Custom: Encourage longer passing chains
    if action == self.PASS and self.pass_chain_length > 2:
        reward += 0.05 * self.pass_chain_length
    
    # Custom: Reward maintaining optimal spacing
    team_spread = self._compute_team_spread(agent)
    if 2.0 < team_spread < 4.0:
        reward += 0.02
    
    # Custom: Bonus for shots from good positions
    if action == self.SHOOT:
        shot_quality = self._evaluate_shot_position(agent)
        reward += 0.1 * shot_quality
    
    return reward
```

### Model Comparison

```bash
# Compare two trained models
python compare_models.py \
    --model1 runs/run_20251214_120000/checkpoints/best_model.pt \
    --model2 runs/run_20251214_150000/checkpoints/best_model.pt \
    --episodes 100
```

Output:
- Win rates
- Average rewards
- Pass statistics
- Behavioral differences
- Statistical significance tests

### Creating Custom Configurations

```yaml
# configs/my_custom_config.yaml

environment:
  use_improved_env: true
  num_agents_per_team: 3
  grid_width: 12
  grid_height: 8
  max_steps: 200

training:
  num_episodes: 2000
  learning_rate: 0.0003
  entropy_coef: 0.08
  entropy_decay_target: 0.01
  entropy_decay_episodes: 1500

rewards:
  goal_reward: 1.0
  possession_gain: 0.2
  pass_reward: 0.15
  time_penalty: -0.0005

logging:
  checkpoint_interval: 50
  log_interval: 10
  tensorboard: true
```

### Self-Play Training

Train against past versions for continual improvement:

```python
# training/self_play.py

import random
from pathlib import Path

# Load checkpoint archive
checkpoint_dir = Path('runs/checkpoints_archive/')
past_checkpoints = list(checkpoint_dir.glob('*.pt'))

# Train against random past version
opponent_checkpoint = random.choice(past_checkpoints)
opponent_agent = load_checkpoint(opponent_checkpoint)

# Training loop with mixed opponents
for episode in range(num_episodes):
    if episode % 10 == 0:  # Switch opponent every 10 episodes
        opponent_agent = load_checkpoint(random.choice(past_checkpoints))
```

---

## 🐛 Troubleshooting

### Training Issues

#### Training Instability / Loss Explosions

**Symptoms**: 
- Policy loss spikes to NaN
- Win rate fluctuates wildly (0% → 100% → 0%)
- Gradients explode

**Solutions**:
```yaml
# Reduce learning rate
learning_rate: 0.0001  # from 0.0003

# Increase GAE lambda (reduces variance)
gae_lambda: 0.97  # from 0.95

# Enable gradient clipping
max_grad_norm: 0.5

# Increase entropy (more exploration)
entropy_coef: 0.08  # from 0.05
```

#### Agents Won't Pass / Always Shoot

**Symptoms**:
- Pass count stays near 0
- Agents dribble and shoot immediately
- No team coordination

**Solutions**:
```python
# Increase pass rewards
PASS_REWARD = 0.20  # from 0.10

# Add pass chain bonuses
if self.pass_chain_length > 1:
    reward += 0.05 * self.pass_chain_length

# Penalize poor shots
if action == SHOOT and distance_to_goal > 5.0:
    reward -= 0.10
```

#### Poor Coordination / Agents Cluster

**Symptoms**:
- All agents stack on ball
- No positioning or spacing
- Agents collide frequently

**Solutions**:
```python
# Add spread reward
team_spread = np.std([pos for pos in team_positions])
if 1.5 < team_spread < 3.5:
    reward += 0.02

# Penalize clustering
for other_agent in teammates:
    dist = np.linalg.norm(self_pos - other_pos)
    if dist < 0.5:
        reward -= 0.05
```

Or use larger grid:
```yaml
grid_width: 14  # from 10
grid_height: 9  # from 6
```

#### Agents Don't Move Toward Ball

**Symptoms**:
- Agents wander randomly
- Don't chase loose balls
- Ball possession stays with one agent

**Solutions**:
- Ensure `emergency_fix.yaml` config is used (minimal time penalty)
- Increase ball-seeking reward
- Check observation space includes ball position

```yaml
# configs/emergency_fix.yaml
time_penalty: -0.001  # Critical: was -0.01 (too harsh)
```

### Environment Issues

#### ImportError: No module named 'env'

```bash
# Install package in development mode
pip install -e .
```

#### AttributeError in ImprovedFootballEnv

```bash
# Ensure you're using the correct environment
python test_improved_env.py

# Check config file
use_improved_env: true  # Must be set
```

#### Reward values seem wrong

```bash
# Validate reward shaping
python test_rewards.py

# Check reward clipping (should be [-1, 1])
```

### Visualization Issues

#### "networkx not installed" warning

```bash
pip install networkx
# Pass network will use matplotlib fallback if networkx unavailable
```

#### GIF files are too large

```python
# Reduce FPS
make_replay_gif(replay, 'output.gif', fps=8)  # from 15

# Or subsample frames
sampled_frames = replay['frames'][::2]  # Every 2nd frame
```

#### Out of memory during visualization

```python
# For long episodes, sample frames
if len(replay['frames']) > 300:
    replay['frames'] = replay['frames'][::2]
```

### Performance Issues

#### Training is slow

**Solutions**:
```yaml
# Reduce PPO epochs
ppo_epochs: 3  # from 4

# Reduce buffer size
buffer_size: 1024  # from 2048

# Reduce max steps
max_steps_per_episode: 100  # from 150

# Start with fewer agents
num_agents_per_team: 2  # from 3
```

#### GPU not being used

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Demo Issues

#### Model loading error: "size mismatch for input_proj.0.weight"

**Symptom**: Error when loading checkpoint in Streamlit app
```
RuntimeError: Error(s) in loading state_dict for Actor:
size mismatch for input_proj.0.weight: copying a param with shape 
torch.Size([512, 21]) from checkpoint, the shape in current model is torch.Size([512, 11])
```

**Cause**: The trained model was created with a different environment than what the app is trying to load.

**Solution**: The app now **automatically detects** the observation dimension from the checkpoint and loads the correct environment. Just make sure you're using the latest `app.py`.

**Manual check**:
```bash
# Check what dimension your model uses
python3 -c "import torch; ckpt = torch.load('runs/YOUR_RUN/checkpoints/best_model.pt', map_location='cpu'); print(f\"Obs dim: {ckpt['team_0_agent']['actor']['input_proj.0.weight'].shape[1]}\")"

# Dimensions:
# 11 = FootballEnv (2 agents, no goal directions)
# 13 = FootballEnv (2 agents, with goal directions)
# 17 = FootballEnv (3 agents, with goal directions)
# 21 = ImprovedFootballEnv
```

#### Streamlit app won't start

```bash
# Install dependencies
pip install streamlit plotly matplotlib

# Check port availability
lsof -i :8501

# Run with explicit port
streamlit run app.py --server.port 8502
```

#### Docker compose fails

```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up
```

### Getting Help

1. **Check logs**: Look in `training.log` or `runs/*/logs/`
2. **Run tests**: `python test_improved_env.py`
3. **Validate config**: Ensure all required fields present
4. **Check documentation**: See markdown files in repo
5. **Open issue**: https://github.com/shashankbaswa007/reinforcement_learning/issues

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{marl_football_2024,
  author = {Your Name},
  title = {Multi-Agent Reinforcement Learning Football},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/baswashashank007/multi-agent-football-rl}
}
```

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- OpenAI for PPO algorithm
- PettingZoo for multi-agent environment API
- Anthropic for Claude assistance in development

## 📞 Contact

Questions? Open an issue or contact: baswashashank123@gmail.com

---

**Happy Training! ⚽🤖**

For more detailed information, see the interactive documentation artifact or individual module docstrings.# multi-agent-football-rl
