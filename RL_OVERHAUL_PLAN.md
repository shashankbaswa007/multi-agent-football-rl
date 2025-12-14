# 🎯 COMPLETE RL FOOTBALL AGENT OVERHAUL
## Comprehensive Fix Plan for Purposeful Agent Behavior

**Current Problem**: Agents wander aimlessly, don't score, don't defend, PPO barely improves.

**Root Cause Analysis**:
1. Reward structure drowns positive signals with time penalties
2. Sparse rewards (goals only) provide insufficient learning signal
3. Observations lack directional information to goal
4. No explicit reward for strategic positioning
5. Entropy collapse causes policy degeneration
6. No defensive behavior rewards

---

## ✅ PART 1: REWARD SHAPING OVERHAUL

### Current Fatal Issues
- Time penalty (-0.02) overwhelms sparse goal rewards (100 over 200 steps = 96 net loss!)
- No rewards for defensive actions
- Failed passes penalized too harshly, discouraging passing
- No reward for moving toward goal without ball

### NEW REWARD SYSTEM (Dense + Sparse)

```python
# BASE REWARDS (per step)
TIME_PENALTY = -0.001  # Minimal, just prevents infinite loops

# POSSESSION REWARDS
GAIN_POSSESSION = +5.0        # Picking up ball
KEEP_POSSESSION = +0.05       # Per step with ball
LOSE_POSSESSION = -2.0        # Lost to opponent

# MOVEMENT REWARDS (with ball)
ADVANCE_TOWARD_GOAL = +0.15   # Per unit distance closer to goal
RETREAT_FROM_GOAL = -0.10     # Per unit distance away from goal

# MOVEMENT REWARDS (without ball)
MOVE_TO_BALL = +0.08          # Moving toward loose ball
DEFENSIVE_POSITIONING = +0.05  # Defender between ball and own goal

# PASSING REWARDS
SUCCESSFUL_PASS = +1.5        # Completed pass
FORWARD_PASS_BONUS = +0.8     # Pass that advances toward goal
FAILED_PASS = -0.5            # Intercepted/dropped
PASS_TO_BETTER_POSITION = +1.0 # Teammate is closer to goal

# SHOOTING REWARDS
SHOT_ATTEMPT = +0.5           # Reward trying (encourages exploration)
SHOT_ON_TARGET = +2.0         # Close shot that could score
GOAL_SCORED = +150.0          # MASSIVE reward for scoring
GOAL_ASSISTED = +50.0         # Passer of goal-scoring play

# DEFENSIVE REWARDS
INTERCEPTION = +3.0           # Taking ball from opponent
BLOCK_SHOT = +5.0             # Preventing shot attempt
TACKLE = +2.0                 # Dispossessing opponent
GOOD_DEFENSE_POSITION = +0.03 # Between attacker and goal

# POSITIONING REWARDS
SPREAD_OUT_BONUS = +0.02      # Not clustering (avg distance > 3)
CLUSTERING_PENALTY = -0.05    # Too close to teammates (< 1.5 units)
STRATEGIC_SPACING = +0.08     # Triangle formation detection

# PENALTIES
COLLISION = -0.5              # Running into walls/agents
INVALID_ACTION = -0.3         # Trying impossible action
STAY_STILL_WITH_BALL = -0.15  # Holding ball without moving
OUT_OF_BOUNDS = -1.0          # Attempting to leave field
```

### Reward Normalization Strategy

```python
class RunningMeanStd:
    """Tracks running mean and std for normalization"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
    
    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        self.var = M2 / total_count
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

# Use in environment
reward = self.reward_normalizer.normalize(reward)
reward = np.clip(reward, -10, 10)  # Clip to prevent explosions
```

### Potential-Based Reward Shaping

```python
def compute_potential(agent_pos, ball_pos, goal_pos, has_ball):
    """Potential function for shaping"""
    if has_ball:
        # Potential = negative distance to goal
        return -np.linalg.norm(agent_pos - goal_pos)
    else:
        # Potential = negative distance to ball
        return -0.5 * np.linalg.norm(agent_pos - ball_pos)

# In step():
old_potential = compute_potential(old_pos, old_ball_pos, goal, old_has_ball)
new_potential = compute_potential(new_pos, new_ball_pos, goal, new_has_ball)
reward += 0.1 * (new_potential - old_potential)  # Shaping term
```

---

## ✅ PART 2: ENVIRONMENT FIXES

### Bug Checklist & Fixes

#### Bug 1: Wrong Goal Direction
```python
# WRONG (current):
target_goal = self.goal_team_1 if team == 0 else self.goal_team_0  # Confusing!

# RIGHT:
# Team 0 attacks RIGHT (toward x=width), Team 1 attacks LEFT (toward x=0)
if team == 0:
    attack_goal = np.array([self.grid_width - 1, self.grid_height // 2])
    defend_goal = np.array([0, self.grid_height // 2])
else:
    attack_goal = np.array([0, self.grid_height // 2])
    defend_goal = np.array([self.grid_width - 1, self.grid_height // 2])
```

#### Bug 2: Reward Magnitude Mismatch
```python
# WRONG: Time penalty overwhelms sparse rewards
reward = -0.02 * steps  # -4.0 over 200 steps
goal_reward = 100  # BUT only happens 0-2 times per episode

# RIGHT: Make time penalty negligible
reward = -0.001 * steps  # -0.2 over 200 steps  
goal_reward = 150  # Goal dominates
```

#### Bug 3: Missing Directional Obs
```python
# OLD observation (blind to goal):
obs = [self_pos, ball_pos, teammates, opponents]

# NEW observation (goal-aware):
obs = [
    self_pos,                    # 2
    ball_pos,                    # 2
    has_ball_flag,               # 1
    distance_to_ball,            # 1
    direction_to_ball,           # 2 (unit vector)
    distance_to_attack_goal,     # 1
    direction_to_attack_goal,    # 2 (unit vector)
    distance_to_defend_goal,     # 1
    direction_to_defend_goal,    # 2 (unit vector)
    teammates_relative_pos,      # 4 (for 3v3)
    opponents_relative_pos,      # 6
    team_score_diff,             # 1 (normalized)
    velocity_vector,             # 2 (if tracked)
]
# Total: ~27 dimensions
```

#### Bug 4: Ball Physics Issues
```python
# WRONG: Ball teleports
self.ball_position = teammate_pos.copy()

# RIGHT: Ball moves realistically
direction = (teammate_pos - passer_pos) / np.linalg.norm(teammate_pos - passer_pos)
ball_speed = 2.0  # units per step
self.ball_position += direction * ball_speed
# Check if ball reaches teammate next step
```

#### Bug 5: No Termination Variety
```python
# OLD: Only goal or max_steps
if goal_scored or steps >= max_steps:
    terminate()

# NEW: Multiple termination conditions
if goal_scored:
    terminate(reason="goal")
elif steps >= max_steps:
    terminate(reason="timeout")
elif all_agents_stuck_for_N_steps:  # Detect stalemate
    terminate(reason="stalemate")
elif ball_out_of_bounds_count > threshold:
    terminate(reason="too_many_turnovers")
```

### Improved Observation Space

```python
def _get_obs_improved(self, agent):
    pos = self.agent_positions[agent]
    team = 0 if 'team_0' in agent else 1
    
    # Attack goal (what team should score on)
    attack_goal = np.array([self.grid_width-1, self.grid_height//2]) if team == 0 \
                  else np.array([0, self.grid_height//2])
    defend_goal = np.array([0, self.grid_height//2]) if team == 0 \
                  else np.array([self.grid_width-1, self.grid_height//2])
    
    # Self position (normalized)
    norm_pos = pos / np.array([self.grid_width, self.grid_height])
    
    # Ball information
    norm_ball = self.ball_position / np.array([self.grid_width, self.grid_height])
    has_ball = 1.0 if self.ball_possession == agent else 0.0
    
    # Distance and direction to ball
    ball_vec = self.ball_position - pos
    ball_dist = np.linalg.norm(ball_vec) / self.grid_width  # Normalized
    ball_dir = ball_vec / (np.linalg.norm(ball_vec) + 1e-6)  # Unit vector
    
    # Distance and direction to attack goal
    attack_vec = attack_goal - pos
    attack_dist = np.linalg.norm(attack_vec) / self.grid_width
    attack_dir = attack_vec / (np.linalg.norm(attack_vec) + 1e-6)
    
    # Distance and direction to defend goal
    defend_vec = defend_goal - pos
    defend_dist = np.linalg.norm(defend_vec) / self.grid_width
    defend_dir = defend_vec / (np.linalg.norm(defend_vec) + 1e-6)
    
    # Teammates (relative positions)
    teammates = [a for a in self.agents if f'team_{team}' in a and a != agent]
    teammate_vecs = []
    for tm in teammates:
        rel_vec = (self.agent_positions[tm] - pos) / self.grid_width
        teammate_vecs.extend(rel_vec)
    
    # Opponents (relative positions)
    opponents = [a for a in self.agents if f'team_{1-team}' in a]
    opponent_vecs = []
    for opp in opponents:
        rel_vec = (self.agent_positions[opp] - pos) / self.grid_width
        opponent_vecs.extend(rel_vec)
    
    # Score difference (normalized)
    score_diff = (self.episode_stats[f'goals_team_{team}'] - 
                  self.episode_stats[f'goals_team_{1-team}']) / 3.0
    
    obs = np.concatenate([
        norm_pos,           # 2
        norm_ball,          # 2
        [has_ball],         # 1
        [ball_dist],        # 1
        ball_dir,           # 2
        [attack_dist],      # 1
        attack_dir,         # 2
        [defend_dist],      # 1
        defend_dir,         # 2
        teammate_vecs,      # 4 for 3v3
        opponent_vecs,      # 6 for 3v3
        [score_diff],       # 1
    ])
    
    return obs.astype(np.float32)
```

---

## ✅ PART 3: PPO TRAINING FIXES

### Optimal Hyperparameters

```python
PPO_CONFIG = {
    # Learning rates
    'lr': 3e-4,                  # Standard PPO lr
    'lr_schedule': 'linear',     # Decay to 1e-5 over training
    
    # PPO-specific
    'gamma': 0.99,               # Higher for credit assignment
    'gae_lambda': 0.95,          # Standard GAE
    'clip_ratio': 0.2,           # PPO clip parameter
    'vf_clip': 10.0,             # Value function clip
    'value_loss_coef': 0.5,      # Value loss weight
    'entropy_coef_start': 0.05,  # High initial exploration
    'entropy_coef_end': 0.001,   # Low final exploration
    'entropy_decay_steps': 5000, # Gradual decay
    
    # Training
    'ppo_epochs': 10,            # Epochs per update
    'mini_batch_size': 128,      # Batch size for SGD
    'buffer_size': 4096,         # Experience buffer
    'update_frequency': 8,       # Episodes between updates
    
    # Gradient
    'max_grad_norm': 0.5,        # Gradient clipping
    'normalize_advantages': True, # Advantage normalization
    'normalize_observations': True,
    'normalize_rewards': True,
    
    # Exploration
    'epsilon_greedy_start': 0.2, # Initial random action prob
    'epsilon_greedy_end': 0.01,  # Final random action prob
    'epsilon_decay_steps': 3000,
}
```

### Critical Training Loop Improvements

```python
def train_with_health_checks():
    \"\"\"Training loop with gradient and policy health monitoring\"\"\"
    
    for episode in range(max_episodes):
        # Collect episode
        obs_list, actions, rewards, values, log_probs = collect_episode()
        
        # === HEALTH CHECK 1: Reward statistics ===
        reward_mean = np.mean(rewards)
        reward_std = np.std(rewards)
        if reward_std < 0.01:
            print(f"⚠️  WARNING: Reward variance collapsed: {reward_std}")
        
        # === HEALTH CHECK 2: Action distribution ===
        action_counts = np.bincount(actions, minlength=7)
        action_probs = action_counts / len(actions)
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        
        if action_entropy < 0.5:
            print(f"⚠️  WARNING: Policy collapse detected. Entropy: {action_entropy:.3f}")
            # FIX: Increase entropy coefficient temporarily
            agent.entropy_coef *= 2.0
        
        # === HEALTH CHECK 3: Value function ===
        value_mean = np.mean(values)
        value_std = np.std(values)
        if abs(value_mean) > 100:
            print(f"⚠️  WARNING: Value function explosion: {value_mean:.2f}")
        
        # Update agent
        for epoch in range(ppo_epochs):
            for batch in get_mini_batches():
                policy_loss, value_loss, entropy = agent.update(batch)
                
                # === HEALTH CHECK 4: Gradient norms ===
                actor_grad_norm = get_grad_norm(agent.actor)
                critic_grad_norm = get_grad_norm(agent.critic)
                
                if actor_grad_norm > 10.0:
                    print(f"⚠️  Actor gradient explosion: {actor_grad_norm:.2f}")
                if actor_grad_norm < 1e-6:
                    print(f"⚠️  Actor gradient vanishing: {actor_grad_norm:.2e}")
                
                # === HEALTH CHECK 5: Policy loss ===
                if policy_loss.item() < 1e-6:
                    print(f"⚠️  Policy loss near zero: {policy_loss:.2e}")
                    print("Possible causes: deterministic policy or clipping issues")
        
        # Log metrics
        writer.add_scalar('Health/ActionEntropy', action_entropy, episode)
        writer.add_scalar('Health/ActorGradNorm', actor_grad_norm, episode)
        writer.add_scalar('Health/RewardStd', reward_std, episode)

def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5
```

### Entropy Scheduling

```python
class EntropyScheduler:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.current_step = 0
    
    def get_value(self):
        progress = min(1.0, self.current_step / self.steps)
        return self.start + (self.end - self.start) * progress
    
    def step(self):
        self.current_step += 1

# Usage
entropy_scheduler = EntropyScheduler(0.05, 0.001, 5000)
agent.entropy_coef = entropy_scheduler.get_value()
```

### Detecting Policy Collapse

```python
def detect_policy_collapse(action_log_probs):
    \"\"\"Detect if policy has collapsed to deterministic\"\"\"
    # If all log probs are very similar, policy is deterministic
    std = torch.std(action_log_probs)
    if std < 0.01:
        return True, "Deterministic policy"
    
    # Check if one action dominates
    probs = torch.exp(action_log_probs)
    max_prob = torch.max(probs)
    if max_prob > 0.95:
        return True, f"Action {torch.argmax(probs)} dominates with p={max_prob:.3f}"
    
    return False, "Policy healthy"
```

---

## ✅ PART 4: BEHAVIOR CONSTRAINTS

### Directional Incentives

```python
def compute_directional_reward(agent_pos, ball_pos, goal_pos, action, team):
    \"\"\"Reward moving in strategically correct directions\"\"\"
    reward = 0
    
    # If agent has ball
    if has_ball:
        # Moving toward goal is good
        to_goal = goal_pos - agent_pos
        direction = get_action_vector(action)
        alignment = np.dot(direction, to_goal) / (np.linalg.norm(to_goal) + 1e-6)
        reward += 0.1 * alignment  # [-0.1, +0.1]
    else:
        # Moving toward ball is good
        to_ball = ball_pos - agent_pos
        direction = get_action_vector(action)
        alignment = np.dot(direction, to_ball) / (np.linalg.norm(to_ball) + 1e-6)
        reward += 0.05 * alignment
    
    return reward
```

### Anti-Clustering Reward

```python
def compute_spacing_reward(agent_pos, teammate_positions, min_distance=2.5):
    \"\"\"Reward agents for maintaining spacing\"\"\"
    if not teammate_positions:
        return 0
    
    distances = [np.linalg.norm(agent_pos - tm_pos) for tm_pos in teammate_positions]
    avg_distance = np.mean(distances)
    min_dist = np.min(distances)
    
    # Penalty for being too close
    if min_dist < 1.5:
        return -0.1
    
    # Reward for good spacing
    if avg_distance > min_distance:
        return 0.05
    
    return 0
```

### Triangle Formation Detection

```python
def detect_triangle_formation(team_positions):
    \"\"\"Reward triangle passing formations\"\"\"
    if len(team_positions) < 3:
        return 0
    
    # Check if agents form roughly equilateral triangle
    pos1, pos2, pos3 = team_positions[:3]
    
    d12 = np.linalg.norm(pos1 - pos2)
    d23 = np.linalg.norm(pos2 - pos3)
    d31 = np.linalg.norm(pos3 - pos1)
    
    # Check if sides are roughly equal (within 30%)
    avg_side = (d12 + d23 + d31) / 3
    max_diff = max(abs(d12 - avg_side), abs(d23 - avg_side), abs(d31 - avg_side))
    
    if max_diff < 0.3 * avg_side and avg_side > 2.0 and avg_side < 5.0:
        return 0.2  # Bonus for good formation
    
    return 0
```

### Defender Zone Shaping

```python
def compute_defender_reward(agent_pos, ball_pos, defend_goal, team):
    \"\"\"Reward defenders for good defensive positioning\"\"\"
    # Defender should be between ball and own goal
    to_goal = defend_goal - ball_pos
    to_agent = agent_pos - ball_pos
    
    # Check if agent is on defensive side of ball
    defensive_side = np.dot(to_goal, to_agent) > 0
    
    if not defensive_side:
        return -0.05  # Penalty for being out of position
    
    # Reward for being on the line between ball and goal
    goal_ball_line = defend_goal - ball_pos
    goal_ball_dist = np.linalg.norm(goal_ball_line)
    goal_ball_unit = goal_ball_line / (goal_ball_dist + 1e-6)
    
    # Project agent position onto line
    agent_ball = agent_pos - ball_pos
    projection_length = np.dot(agent_ball, goal_ball_unit)
    projection_point = ball_pos + projection_length * goal_ball_unit
    
    # Distance from ideal defensive line
    dist_from_line = np.linalg.norm(agent_pos - projection_point)
    
    if dist_from_line < 1.5:
        return 0.08  # Good defensive position
    
    return 0
```

---

## ✅ PART 5: DEBUGGING TOOLS (CRITICAL)

### Action Distribution Monitor

```python
def plot_action_distribution(actions, episode, save_dir='debug/actions'):
    \"\"\"Plot action histogram to detect stuck policies\"\"\"
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    action_names = ['Stay', 'Up', 'Down', 'Left', 'Right', 'Pass', 'Shoot']
    counts = np.bincount(actions, minlength=7)
    
    ax.bar(action_names, counts)
    ax.set_title(f'Action Distribution - Episode {episode}')
    ax.set_ylabel('Count')
    
    # Add entropy text
    probs = counts / np.sum(counts)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    ax.text(0.7, 0.95, f'Entropy: {entropy:.3f}', transform=ax.transAxes)
    
    plt.savefig(f'{save_dir}/actions_ep{episode}.png')
    plt.close()
```

### Value vs Returns Scatter

```python
def plot_value_vs_returns(predicted_values, actual_returns, episode, save_dir='debug/value'):
    \"\"\"Check if critic is learning\"\"\"
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(actual_returns, predicted_values, alpha=0.5)
    
    # Add y=x line
    min_val = min(min(actual_returns), min(predicted_values))
    max_val = max(max(actual_returns), max(predicted_values))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    # Calculate R²
    correlation = np.corrcoef(actual_returns, predicted_values)[0, 1]
    ax.text(0.05, 0.95, f'R² = {correlation**2:.3f}', transform=ax.transAxes)
    
    ax.set_xlabel('Actual Returns')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Value Function Quality - Episode {episode}')
    ax.legend()
    
    plt.savefig(f'{save_dir}/value_scatter_ep{episode}.png')
    plt.close()
```

### Heatmap of Agent Positions

```python
def generate_position_heatmap(position_history, save_path):
    \"\"\"Visualize where agents spend time\"\"\"
    import matplotlib.pyplot as plt
    
    # Create 2D histogram
    x_coords = [pos[0] for pos in position_history]
    y_coords = [pos[1] for pos in position_history]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    heatmap, xedges, yedges = np.histogram2d(x_coords, y_coords, bins=[12, 8])
    
    im = ax.imshow(heatmap.T, origin='lower', cmap='hot', interpolation='nearest')
    ax.set_title('Agent Position Heatmap')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    plt.colorbar(im)
    
    plt.savefig(save_path)
    plt.close()
```

### Reward Decomposition Logger

```python
class RewardLogger:
    def __init__(self):
        self.components = defaultdict(list)
    
    def log(self, component, value):
        self.components[component].append(value)
    
    def print_summary(self, episode):
        print(f"\n=== Episode {episode} Reward Breakdown ===")
        for component, values in self.components.items():
            total = sum(values)
            avg = np.mean(values)
            print(f"{component:25s}: Total={total:7.2f}, Avg={avg:6.3f}, Count={len(values)}")
        print("=" * 50)
    
    def reset(self):
        self.components = defaultdict(list)

# Usage
reward_logger = RewardLogger()
reward_logger.log('possession', 0.05)
reward_logger.log('goal', 150.0)
reward_logger.print_summary(episode)
```

### Episode Play-by-Play Text

```python
def log_episode_trace(episode_data, filename):
    \"\"\"Generate human-readable episode log\"\"\"
    with open(filename, 'w') as f:
        f.write(f"Episode Trace - {len(episode_data)} steps\n")
        f.write("=" * 80 + "\n\n")
        
        for step, data in enumerate(episode_data):
            f.write(f"Step {step}:\n")
            f.write(f"  Ball at: {data['ball_pos']}\n")
            f.write(f"  Possession: {data['possession']}\n")
            
            for agent, action in data['actions'].items():
                reward = data['rewards'][agent]
                f.write(f"    {agent}: {action} (reward: {reward:+.2f})\n")
            
            if 'goal' in data:
                f.write(f"  🎯 GOAL SCORED by team {data['goal']}!\n")
            
            f.write("\n")
```

### Check If Agents Reach Goal

```python
def track_goal_proximity(agent_positions, goal_pos, threshold=3.0):
    \"\"\"Track how often agents get close to goal\"\"\"
    distances = [np.linalg.norm(pos - goal_pos) for pos in agent_positions]
    min_dist = min(distances)
    
    return {
        'min_distance': min_dist,
        'reached_goal_area': min_dist < threshold,
        'avg_distance': np.mean(distances)
    }

# In training loop
goal_stats = track_goal_proximity(positions, goal_pos)
if not goal_stats['reached_goal_area']:
    print("⚠️  WARNING: No agent reached goal area this episode!")
```

---

## ✅ PART 6: CURRICULUM LEARNING

```python
CURRICULUM_STAGES = [
    {
        'name': '1v0_practice',
        'team_0_agents': 1,
        'team_1_agents': 0,  # No opponents
        'episodes': 500,
        'success_threshold': 0.8,  # Score 80% of time
        'reward_scale': 1.0,
    },
    {
        'name': '1v1_basic',
        'team_0_agents': 1,
        'team_1_agents': 1,
        'episodes': 1000,
        'success_threshold': 0.5,
        'reward_scale': 1.0,
    },
    {
        'name': '2v2_intermediate',
        'team_0_agents': 2,
        'team_1_agents': 2,
        'episodes': 1500,
        'success_threshold': 0.4,
        'reward_scale': 1.0,
    },
    {
        'name': '3v3_full',
        'team_0_agents': 3,
        'team_1_agents': 3,
        'episodes': 3000,
        'success_threshold': 0.3,
        'reward_scale': 1.0,
    },
]
```

---

## ✅ PART 7: COMMON RL FAILURE FIXES

### Sparse Reward Collapse
**Problem**: Agents never discover rewards  
**Fix**: Add dense shaping rewards, potential-based rewards, curiosity bonus

### Zero-Gradient Updates
**Problem**: Policy loss becomes zero  
**Fix**: Check clip ratio (may be too small), ensure PPO advantage not normalized to zero, add gradient noise

### Freeze Behavior
**Problem**: Agents learn to stay still  
**Fix**: Penalize STAY action when holding ball, reward movement toward objectives

### Agent Loops
**Problem**: Agents move back and forth  
**Fix**: Add position history penalty, reward novel states, add progress tracking

### No Exploration
**Problem**: Policy becomes deterministic too fast  
**Fix**: Higher initial entropy (0.05+), slower decay, epsilon-greedy exploration

### Wrong Termination
**Problem**: Episode ends before learning signal  
**Fix**: Extend max_steps, add intermediate rewards, check termination logic

### Deterministic Collapse
**Problem**: Policy outputs same action always  
**Fix**: Increase entropy coefficient, add action regularization, reset policy periodically

---

## ✅ TOP 3 FIXES TO IMPLEMENT IMMEDIATELY

### 🚨 FIX #1: REDUCE TIME PENALTY (5 min fix)
**Current**: `reward = -0.02` per step  
**New**: `reward = -0.001` per step  
**Impact**: Prevents time penalty from overwhelming goal rewards (100x reduction)

### 🚨 FIX #2: ADD DIRECTION-TO-GOAL OBSERVATION (10 min fix)
```python
# Add to observation:
attack_goal_vec = target_goal - agent_pos
attack_goal_dir = attack_goal_vec / (np.linalg.norm(attack_goal_vec) + 1e-6)
obs = np.concatenate([obs, attack_goal_dir])  # Add 2 dims
```
**Impact**: Agents can now "see" which direction to move

### 🚨 FIX #3: INCREASE ENTROPY COEFFICIENT (2 min fix)
**Current**: `entropy_coef = 0.015`  
**New**: `entropy_coef = 0.05`, decay to `0.005` over 5000 episodes  
**Impact**: Prevents premature policy collapse, enables exploration

---

## 📊 SUCCESS METRICS

After implementing fixes, you should see within 500 episodes:
- ✅ Agents move toward goal (heatmap shows directional bias)
- ✅ Action entropy > 1.0 (diverse actions)
- ✅ At least 1 goal scored per 10 episodes
- ✅ Policy loss > 0.01 (learning happening)
- ✅ Value function R² > 0.3 (critic learning)
- ✅ Possession time > 30% of episode
- ✅ Pass success rate > 40%

If not seeing these, revisit reward normalization and observation space.
