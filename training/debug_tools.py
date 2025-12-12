"""
Debugging and visualization tools for training diagnostics
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def log_episode_trace(env, agents, episode_num, max_steps=50):
    """Detailed logging for debugging early episodes"""
    if episode_num > 5:
        return
    
    print(f"\n{'='*70}")
    print(f"EPISODE {episode_num} DETAILED TRACE")
    print(f"{'='*70}")
    
    observations, _ = env.reset()
    step_rewards = {agent: [] for agent in env.agents}
    actions_taken = {agent: [] for agent in env.agents}
    
    for step in range(min(max_steps, env.max_steps)):
        print(f"\nStep {step}:")
        action_dict = {}
        
        for agent in env.agents:
            obs = observations[agent]
            team_id = 0 if 'team_0' in agent else 1
            agent_obj = agents[team_id]
            action, log_prob, value, entropy = agent_obj.get_action(obs)
            action_dict[agent] = action
            
            print(f"  {agent}:")
            print(f"    Action: {action}, Value: {value:.3f}, Entropy: {entropy:.3f}")
            actions_taken[agent].append(action)
        
        # Step environment
        for agent in env.agent_iter():
            observations_dict, rewards_dict, terminations, truncations, info = env.step(action_dict[agent])
            step_rewards[agent].append(rewards_dict[agent])
            
            if step < 3:  # Only print first few steps
                print(f"    Reward: {rewards_dict[agent]:.3f}")
            
            if terminations[agent] or truncations[agent]:
                break
        
        observations = observations_dict
        
        if all(terminations.values()) or all(truncations.values()):
            print(f"\nEpisode ended at step {step}")
            break
    
    # Summary
    print(f"\n{'='*70}")
    print("EPISODE SUMMARY:")
    for agent in env.agents:
        total_reward = sum(step_rewards[agent])
        action_counts = np.bincount(actions_taken[agent], minlength=7)
        print(f"{agent}:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Actions: Stay={action_counts[0]}, Moves={sum(action_counts[1:5])}, "
              f"Pass={action_counts[5]}, Shoot={action_counts[6]}")
    print(f"{'='*70}\n")


def plot_action_distribution(actions_log, save_path):
    """Plot action distribution over training to detect policy collapse"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    action_names = ['Stay', 'Up', 'Down', 'Left', 'Right', 'Pass', 'Shoot']
    
    episode_ranges = [(0, 50), (50, 150), (150, 300), (300, -1)]
    
    for idx, (start, end) in enumerate(episode_ranges):
        ax = axes[idx // 2, idx % 2]
        
        if end == -1:
            episode_actions = actions_log[start:]
        else:
            episode_actions = actions_log[start:end]
        
        if len(episode_actions) == 0:
            continue
            
        flat_actions = np.concatenate(episode_actions)
        action_counts = np.bincount(flat_actions, minlength=7)
        action_probs = action_counts / action_counts.sum()
        
        ax.bar(range(7), action_probs)
        ax.set_xticks(range(7))
        ax.set_xticklabels(action_names, rotation=45, ha='right')
        ax.set_ylabel('Probability')
        ax.set_title(f'Episodes {start}-{end if end != -1 else "End"}')
        ax.set_ylim(0, 1)
        
        # Check for collapse (one action > 80%)
        if action_probs.max() > 0.8:
            dominant_action = action_names[action_probs.argmax()]
            ax.text(0.5, 0.9, f'⚠️ COLLAPSED: {dominant_action}', 
                   transform=ax.transAxes, ha='center', 
                   bbox=dict(boxstyle='round', facecolor='red', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Action distribution plot saved to {save_path}")
    plt.close()


def plot_value_vs_return(values, returns, save_path):
    """Scatter plot to check if value function is learning"""
    plt.figure(figsize=(10, 6))
    
    values_flat = np.array(values).flatten()
    returns_flat = np.array(returns).flatten()
    
    # Sample if too many points
    if len(values_flat) > 5000:
        indices = np.random.choice(len(values_flat), 5000, replace=False)
        values_flat = values_flat[indices]
        returns_flat = returns_flat[indices]
    
    plt.scatter(values_flat, returns_flat, alpha=0.3, s=10)
    
    # Perfect prediction line
    min_val = min(values_flat.min(), returns_flat.min())
    max_val = max(values_flat.max(), returns_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Value Predictions')
    plt.ylabel('Actual Returns')
    plt.title('Value Function Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    correlation = np.corrcoef(values_flat, returns_flat)[0, 1]
    mse = np.mean((values_flat - returns_flat) ** 2)
    
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nMSE: {mse:.2f}',
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Value vs Return plot saved to {save_path}")
    print(f"  Correlation: {correlation:.3f} (should be > 0.5)")
    print(f"  MSE: {mse:.2f}")
    plt.close()


def check_training_health(stats_dict):
    """Run diagnostic checks on training statistics"""
    print("\n" + "="*70)
    print("TRAINING HEALTH CHECK")
    print("="*70)
    
    issues_found = []
    
    # Check 1: Policy loss near zero
    recent_policy_loss = np.mean(stats_dict['policy_loss'][-50:])
    if abs(recent_policy_loss) < 0.001:
        issues_found.append("⚠️  Policy loss near zero - no learning happening")
    
    # Check 2: Value loss not decreasing
    if len(stats_dict['value_loss']) > 100:
        early_value_loss = np.mean(stats_dict['value_loss'][:50])
        recent_value_loss = np.mean(stats_dict['value_loss'][-50:])
        if recent_value_loss >= early_value_loss:
            issues_found.append("⚠️  Value loss not decreasing - value function not learning")
    
    # Check 3: Entropy collapsed
    recent_entropy = np.mean(stats_dict['entropy'][-50:])
    if recent_entropy < 0.1:
        issues_found.append(f"⚠️  Entropy collapsed ({recent_entropy:.3f}) - no exploration")
    
    # Check 4: KL divergence too high
    recent_kl = np.mean(stats_dict['kl_divergence'][-50:])
    if recent_kl > 0.05:
        issues_found.append(f"⚠️  High KL divergence ({recent_kl:.3f}) - unstable updates")
    
    # Check 5: Clip fraction
    recent_clip = np.mean(stats_dict['clip_fraction'][-50:])
    if recent_clip > 0.8:
        issues_found.append(f"⚠️  High clip fraction ({recent_clip:.2f}) - need larger epsilon")
    elif recent_clip < 0.05:
        issues_found.append(f"⚠️  Low clip fraction ({recent_clip:.2f}) - updates too conservative")
    
    if issues_found:
        print("\nISSUES DETECTED:")
        for issue in issues_found:
            print(f"  {issue}")
    else:
        print("\n✓ Training appears healthy!")
    
    print("\nCURRENT METRICS:")
    print(f"  Policy Loss: {recent_policy_loss:.4f}")
    print(f"  Entropy: {recent_entropy:.4f}")
    print(f"  KL Divergence: {recent_kl:.4f}")
    print(f"  Clip Fraction: {recent_clip:.4f}")
    print("="*70 + "\n")
    
    return len(issues_found) == 0
