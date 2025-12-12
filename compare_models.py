#!/usr/bin/env python3
"""
Comparison script to demonstrate improvements between original and enhanced models.
Runs both configurations and compares key metrics.
"""

import subprocess
import sys
import time
import re
from pathlib import Path

def extract_metrics(output):
    """Extract key metrics from training output"""
    metrics = {
        'episodes': [],
        'rewards': [],
        'win_rates': [],
        'policy_losses': []
    }
    
    for line in output.split('\n'):
        if 'Episode' in line and '/' in line:
            match = re.search(r'Episode (\d+)/\d+', line)
            if match:
                metrics['episodes'].append(int(match.group(1)))
        elif 'Team 0 Reward:' in line:
            match = re.search(r'Team 0 Reward: ([-\d.]+)', line)
            if match:
                metrics['rewards'].append(float(match.group(1)))
        elif 'Win Rate:' in line:
            match = re.search(r'Win Rate: ([\d.]+)%', line)
            if match:
                metrics['win_rates'].append(float(match.group(1)))
        elif 'Policy Loss:' in line:
            match = re.search(r'Policy Loss: ([-\d.]+)', line)
            if match:
                metrics['policy_losses'].append(float(match.group(1)))
    
    return metrics

def run_training(config_file, label):
    """Run training with specified config"""
    print(f"\n{'='*60}")
    print(f"Running {label}")
    print(f"Config: {config_file}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable,
        'training/train_ppo.py',
        '--config',
        config_file
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✓ Completed in {elapsed:.1f} seconds")
            return extract_metrics(result.stdout), elapsed
        else:
            print(f"✗ Failed with error:")
            print(result.stderr)
            return None, elapsed
            
    except subprocess.TimeoutExpired:
        print(f"✗ Timeout after 5 minutes")
        return None, 300

def print_comparison(original_metrics, improved_metrics, orig_time, imp_time):
    """Print comparison of results"""
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}\n")
    
    print(f"Training Time:")
    print(f"  Original: {orig_time:.1f}s")
    print(f"  Improved: {imp_time:.1f}s")
    print(f"  Difference: {((imp_time/orig_time - 1) * 100):+.1f}%\n")
    
    if original_metrics and improved_metrics:
        print(f"Final Episode Reward:")
        print(f"  Original: {original_metrics['rewards'][-1]:.2f}")
        print(f"  Improved: {improved_metrics['rewards'][-1]:.2f}")
        print(f"  Difference: {(improved_metrics['rewards'][-1] - original_metrics['rewards'][-1]):+.2f}\n")
        
        print(f"Average Reward (last 5 episodes):")
        orig_avg = sum(original_metrics['rewards'][-5:]) / 5
        imp_avg = sum(improved_metrics['rewards'][-5:]) / 5
        print(f"  Original: {orig_avg:.2f}")
        print(f"  Improved: {imp_avg:.2f}")
        print(f"  Difference: {(imp_avg - orig_avg):+.2f}\n")
        
        print(f"Win Rate:")
        print(f"  Original: {original_metrics['win_rates'][-1]:.1f}%")
        print(f"  Improved: {improved_metrics['win_rates'][-1]:.1f}%")
        print(f"  Difference: {(improved_metrics['win_rates'][-1] - original_metrics['win_rates'][-1]):+.1f}%\n")
    
    print(f"{'='*60}")
    print("KEY IMPROVEMENTS:")
    print(f"{'='*60}")
    print("✓ Enhanced neural architecture with residual connections")
    print("✓ Layer normalization for stable training")  
    print("✓ Improved reward shaping for better learning signals")
    print("✓ Advanced PPO features (early stopping, KL monitoring)")
    print("✓ Adaptive learning rate scheduling")
    print("✓ Better hyperparameter tuning")
    print(f"{'='*60}\n")

def main():
    """Main comparison function"""
    print("\n" + "="*60)
    print("MODEL IMPROVEMENT COMPARISON")
    print("="*60)
    print("\nThis script compares original vs improved model architectures")
    print("Running 50 episodes each for quick comparison...\n")
    
    # Run original model
    print("\n[1/2] Testing ORIGINAL model...")
    original_metrics, orig_time = run_training(
        'configs/test_config.yaml',
        'ORIGINAL MODEL'
    )
    
    # Run improved model
    print("\n[2/2] Testing IMPROVED model...")
    improved_metrics, imp_time = run_training(
        'configs/quick_improved_test.yaml',
        'IMPROVED MODEL'
    )
    
    # Print comparison
    print_comparison(original_metrics, improved_metrics, orig_time, imp_time)
    
    print("\nFor full training comparison, run:")
    print("  Original: python training/train_ppo.py --config configs/default_config.yaml")
    print("  Improved: python training/train_ppo.py --config configs/improved_config.yaml\n")

if __name__ == '__main__':
    main()
