# Multi-Agent Reinforcement Learning Football Project

A complete implementation of Multi-Agent RL for learning coordinated football (soccer) strategies using Proximal Policy Optimization (PPO). Agents learn to pass, position, and score through team-based rewards and curriculum learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Key Features

- **Custom Grid-Based Football Environment** (12×8 grid, 3v3)
- **Shared-Parameter PPO** with centralized training, decentralized execution
- **Curriculum Learning** (1v1 → 2v2 → 3v3 progression)
- **Comprehensive Visualization Suite**
  - Step-by-step replay viewer
  - Agent movement heatmaps
  - Pass network analysis
  - Training curve plotting
- **Emergent Behavior Analysis Tools**
- **TensorBoard Integration**
- **Modular, Research-Grade Code**

## 📁 Project Structure

```
reinforcement_learning/
├── env/
│   ├── __init__.py
│   └── football_env.py          # Custom PettingZoo environment
├── models/
│   ├── __init__.py
│   └── ppo_agent.py             # Complete PPO implementation (Actor-Critic)
├── training/
│   ├── __init__.py
│   ├── train_ppo.py             # Main training loop with curriculum
│   ├── buffer.py                # GAE-based experience buffer
│   └── utils.py                 # Helper functions & schedules
├── configs/
│   ├── default_config.yaml      # Standard 3v3 training (20K episodes)
│   ├── curriculum_config.yaml   # Progressive 1v1→2v2→3v3 (30K episodes)
│   ├── fast_config.yaml         # Quick 2v2 testing (5K episodes)
│   └── test_config.yaml         # Minimal 2v2 for validation (50 episodes)
├── visualization.py             # Complete visualization suite
├── test_env.py                  # Unit tests
├── simple_test.py               # Quick environment validation
├── demo.ipynb                   # Interactive Jupyter demo
├── requirements.txt
├── README.md
└── setup.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/multi_agent_football.git
cd multi_agent_football

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Basic Training

```bash
# Standard 3v3 training
python training/train_ppo.py --config configs/default_config.yaml

# Curriculum training (recommended for best results)
python training/train_ppo.py --curriculum --config configs/curriculum_config.yaml

# Fast training for testing
python training/train_ppo.py --config configs/fast_config.yaml
```

### Monitor Training

```bash
# Launch TensorBoard
tensorboard --logdir runs/

# View in browser at http://localhost:6006
```

### Visualize Results

```bash
# Replay trained agent episodes
python visualization/replay_viewer.py --checkpoint checkpoints/best_model.pt --episodes 5

# Generate movement heatmaps
python visualization/heatmap.py --checkpoint checkpoints/best_model.pt --episodes 100

# Analyze passing patterns
python visualization/pass_network.py --checkpoint checkpoints/best_model.pt --episodes 100

# Plot training curves
python visualization/training_plots.py --logdir runs/run_20240101_120000
```

## 📊 Training Methodology

### Environment Design

**Grid World**: 12×8 discrete grid representing football field
- Goal zones at x=0 and x=11
- Agents can move, pass, or shoot
- Ball physics: possession-based with pass accuracy decay

**Observation Space** (17-dimensional per agent):
- Self position (2)
- Ball position + possession flag (3)
- Teammate positions (4)
- Opponent positions (6)
- Own goal position (2)

**Action Space** (7 discrete actions):
- 0: STAY
- 1-4: Move (UP/DOWN/LEFT/RIGHT)
- 5: PASS (to nearest teammate)
- 6: SHOOT (at opponent goal)

### Reward Structure

Team-based rewards encourage cooperation:

```
+100  Goal scored (all team members)
-100  Goal conceded (all team members)
+10   Successful pass
+5    Ball possession gain
+2    Moving toward ball (when not possessing)
-1    Invalid action
-0.01 Time penalty (per step)
```

**Critical**: Reward shaping ratios matter!
- Goal rewards: 10x pass rewards
- Pass rewards: 2x possession rewards
- This balance encourages passing without over-optimization

### PPO Architecture

**Actor Network** (Policy):
```
Input (17) → Linear(256) → Tanh → Linear(256) → Tanh → Linear(7) → Softmax
```

**Critic Network** (Value):
```
Input (17) → Linear(512) → Tanh → Linear(256) → Tanh → Linear(1)
```

**Key Hyperparameters**:
```yaml
learning_rate: 3e-4
gamma: 0.99
gae_lambda: 0.95
clip_epsilon: 0.2
entropy_coef: 0.01 → 0.001 (linear decay)
ppo_epochs: 4
batch_size: 64
```

### Curriculum Learning

Progressive difficulty increases sample efficiency:

**Stage 1: 1v1 (Episodes 0-2000)**
- Focus: Ball control, shooting, basic movement
- Target: 60% win rate

**Stage 2: 2v2 (Episodes 2000-4000)**
- Focus: Passing introduction, basic teamwork
- Target: 60% win rate

**Stage 3: 3v3 (Episodes 4000+)**
- Focus: Full coordination, complex strategies
- Target: 70%+ win rate

Automatic progression when threshold met for 200 consecutive episodes.

## 📈 Expected Results

### Performance Milestones

| Episodes | Win Rate | Avg Passes | Pass Success | Behavior |
|----------|----------|------------|--------------|----------|
| 0-1000   | ~30%     | 0.5        | ~40%         | Random exploration |
| 1000-5000| ~50%     | 2.0        | ~60%         | Basic strategies emerge |
| 5000-10000| ~65%    | 3.5        | ~75%         | Consistent passing chains |
| 10000+   | ~75%     | 4.0        | ~80%         | Coordinated team play |

### Emergent Behaviors

After sufficient training, agents demonstrate:

1. **Passing Chains**: 3-5 consecutive passes before shooting
2. **Positional Play**: Maintaining formation spread (2-4 grid units)
3. **Role Specialization**: Forward/defensive positioning
4. **Opportunistic Shooting**: Taking shots when close to goal
5. **Ball Seeking**: Moving toward ball when not in possession

## 🔬 Analysis Tools

### Coordination Metrics

**Spread Analysis**:
```python
from training.utils import compute_coordination_metrics

metrics = compute_coordination_metrics(agent_positions_history)
# Returns: mean_spread, formation_stability, movement_synchronization
```

**Pass Network Centrality**:
```python
from visualization.pass_network import PassNetworkAnalyzer

analyzer = PassNetworkAnalyzer(checkpoint_path)
pass_matrix, touches = analyzer.collect_pass_data(num_episodes=100)

# Compute betweenness centrality to find playmakers
```

### Strategy Detection

```python
from training.utils import detect_emergent_strategies

strategies = detect_emergent_strategies(episodes_data)
# Returns: passing_chains, direct_play, possession_play, counter_attacks
```

## 🎓 Advanced Usage

### Custom Reward Shaping

Edit `env/football_env.py` to modify reward structure:

```python
# Encourage longer passing chains
if action == PASS and self.pass_chain_length > 2:
    reward += 5 * self.pass_chain_length

# Reward positional spacing
team_spread = compute_team_spread()
if 2.0 < team_spread < 4.0:
    reward += 1
```

### Opponent Customization

Create scripted opponents in `opponents/scripted_opponent.py`:

```python
class ScriptedOpponent:
    def get_action(self, obs):
        # Rule-based policy
        if has_ball:
            return SHOOT if near_goal else PASS
        else:
            return move_toward_ball()
```

### Self-Play Training

Modify `train_ppo.py` to train against past versions:

```python
# Keep archive of past checkpoints
opponent_agent = load_checkpoint(random.choice(checkpoint_archive))
```

### Curriculum Design

Add custom stages in config:

```yaml
curriculum_stages:
  - name: "Shooting Practice"
    num_agents_per_team: 1
    reward_multipliers:
      goal: 2.0
      pass: 0.0  # Disable passing rewards
  
  - name: "Passing Drills"
    num_agents_per_team: 2
    reward_multipliers:
      goal: 0.5
      pass: 5.0  # Emphasize passing
```

## 🐛 Troubleshooting

### Training Instability

**Symptom**: Policy loss explodes, win rate fluctuates wildly

**Solutions**:
1. Reduce learning rate: `lr: 1e-4`
2. Increase GAE lambda: `gae_lambda: 0.97`
3. Enable gradient clipping: `max_grad_norm: 0.5`
4. Use curriculum learning

### Agents Won't Pass

**Symptom**: Agents only dribble and shoot

**Solutions**:
1. Increase pass reward: `pass_reward = 15` (from 10)
2. Add pass chain bonuses
3. Penalize failed shots: `failed_shot_penalty = -10`
4. Use curriculum to force passing in early stages

### Poor Coordination

**Symptom**: Agents cluster together, no positioning

**Solutions**:
1. Add spread reward based on team variance
2. Penalize collisions more heavily
3. Use larger grid (14×10 instead of 12×8)
4. Add role-based position rewards

### Slow Training

**Solutions**:
1. Use GPU: `use_gpu: true`
2. Increase parallel environments: `num_envs: 32`
3. Reduce PPO epochs: `ppo_epochs: 3`
4. Start with 2v2 instead of 3v3

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{marl_football_2024,
  author = {Your Name},
  title = {Multi-Agent Reinforcement Learning Football},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/multi_agent_football}
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

Questions? Open an issue or contact: your.email@example.com

---

**Happy Training! ⚽🤖**

For more detailed information, see the interactive documentation artifact or individual module docstrings.# multi-agent-football-rl
