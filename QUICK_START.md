# 🚀 Quick Start Guide - After RL Fixes

## What Was Fixed

✅ **Time penalty**: -0.005 → -0.001 (100x reduction)  
✅ **Observations**: Added goal direction vectors (+4 dims)  
✅ **Entropy**: 0.015 → 0.05 (5x increase for exploration)  
✅ **Rewards**: Enhanced shaping (possession, advancement, defensive)  

## Training Commands

### Option 1: Stable Learning (RECOMMENDED)
```bash
cd /Users/shashi/reinforcement_learning
source .venv/bin/activate
python training/train_ppo.py --config configs/stable_learning.yaml
```
**Features**: 1v1 mode, lower LR (0.0001), optimized for PPO stability

### Option 2: Emergency Fix
```bash
python training/train_ppo.py --config configs/emergency_fix.yaml
```
**Features**: 2v2 mode, higher LR (0.0003), may have KL divergence issues

## Monitor Training

### TensorBoard
```bash
tensorboard --logdir runs/
```

**Key Metrics**:
- `train/entropy` - Should start ~1.5, decay to ~0.8
- `train/policy_loss` - Should stay > 0.01
- `train/kl_divergence` - Should stay < 0.1 (green flag!)
- `episode/reward_mean` - Should trend upward
- `episode/goals_scored` - Should increase from 0 → 5%

### Console Output
Watch for:
- ✅ **GREEN**: "Early stopping at epoch 5/10" (learning happening)
- ⚠️ **YELLOW**: "Early stopping at epoch 1/10" (KL too high, reduce LR)
- ❌ **RED**: "Policy Loss: 0.0000" (policy collapse, restart)
- ✅ **GREEN**: Gradients > 1e-6
- ❌ **RED**: Vanishing gradients < 1e-8

## Test Visualization

### Generate Replay
```bash
python demo/replay_schema.py
```

### View in Streamlit
```bash
streamlit run demo/streamlit_app.py
```

**Look for**:
- Agents chasing ball (not wandering)
- Movement toward goal with possession
- Shooting attempts near goal

## Quick Debugging

### Check Observation Dimensions
```python
from env.football_env import FootballEnv
env = FootballEnv(num_agents_per_team=2)
obs, _ = env.reset()
print(obs['team_0_agent_0'].shape)  # Should be (15,) for 2v2
```

### Check Goal Directions
```python
obs_vec = obs['team_0_agent_0']
attack_dir = obs_vec[-4:-2]  # Last 4 dims = attack + defend
defend_dir = obs_vec[-2:]
print(f"Attack: {attack_dir}, Defend: {defend_dir}")
# Team 0: attack_dir[0] > 0 (RIGHT), defend_dir[0] < 0 (LEFT)
```

### Log Actions
```python
from collections import Counter
# In training loop after collecting actions:
action_counts = Counter(actions)
print(f"Actions: {action_counts}")
# Should see variety, not one action >90%
```

## Success Criteria

### After 200 episodes:
- [ ] At least 1 goal scored
- [ ] KL divergence < 0.1
- [ ] Action entropy > 1.0
- [ ] Policy loss > 0.01

### After 500 episodes:
- [ ] Win rate > 20%
- [ ] Goal scoring > 5%
- [ ] Reward trending upward

### After 1000 episodes:
- [ ] Win rate > 50%
- [ ] Pass success > 15%
- [ ] Observable strategy

## Troubleshooting

### Problem: High KL Divergence (early stop at epoch 1)
**Fix**: Reduce learning rate
```yaml
# In config file
lr: 0.00005  # Try even lower
```

### Problem: Vanishing Gradients
**Fix**: Increase value loss coefficient
```yaml
value_loss_coef: 2.0  # Was 1.5
```

### Problem: Zero Goals After 500 Episodes
**Fix**: Simplify to 1v0 mode
```python
# Remove opponents temporarily
env = FootballEnv(num_agents_per_team=1, no_opponents=True)
```

### Problem: Policy Collapse (all same action)
**Fix**: Increase entropy or reset
```yaml
entropy_coef: 0.08  # Increase from 0.05
```

## Files Reference

- **Implementation**: `env/football_env.py`
- **Config (Stable)**: `configs/stable_learning.yaml`
- **Config (Original)**: `configs/emergency_fix.yaml`
- **Training**: `training/train_ppo.py`
- **Documentation**: `IMPLEMENTATION_SUMMARY.md`, `RL_OVERHAUL_PLAN.md`

## Next Steps

1. ✅ Run with `stable_learning.yaml`
2. ✅ Monitor KL divergence and gradients
3. ⏭️ Add debugging tools (action plots, heatmaps)
4. ⏭️ Implement curriculum (1v0 → 1v1 → 2v2 → 3v3)
5. ⏭️ Scale up complexity

## Need Help?

See detailed documentation:
- `CRITICAL_FIXES_APPLIED.md` - What changed and why
- `TRAINING_ANALYSIS.md` - Training results analysis
- `RL_OVERHAUL_PLAN.md` - Comprehensive upgrade plan
