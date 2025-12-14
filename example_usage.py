"""
Example Usage of Visualization Suite
=====================================

This script demonstrates how to use all visualization functions
with synthetic replay data.

Run with: python example_usage.py
"""

import numpy as np
import json
from pathlib import Path
from visualization_utils import (
    plot_movement_trails,
    plot_pass_network,
    plot_heatmaps,
    make_replay_gif,
    make_before_after_gif,
    save_replay_to_json
)


def generate_synthetic_replay(num_frames=100, field_width=10, field_height=6, 
                              num_agents_per_team=2, seed=42):
    """
    Generate synthetic replay data for testing visualizations.
    
    Returns replay in standard format.
    """
    np.random.seed(seed)
    
    replay = {
        'field_width': field_width,
        'field_height': field_height,
        'num_agents_per_team': num_agents_per_team,
        'frames': []
    }
    
    # Initialize agent positions
    agent_positions = {}
    agent_velocities = {}
    
    for i in range(num_agents_per_team):
        # Team 0 starts on left
        agent_positions[f'team_0_agent_{i}'] = np.array([
            field_width * 0.25 + np.random.randn() * 0.5,
            field_height / 2 + np.random.randn() * 0.5
        ])
        agent_velocities[f'team_0_agent_{i}'] = np.random.randn(2) * 0.1
        
        # Team 1 starts on right
        agent_positions[f'team_1_agent_{i}'] = np.array([
            field_width * 0.75 + np.random.randn() * 0.5,
            field_height / 2 + np.random.randn() * 0.5
        ])
        agent_velocities[f'team_1_agent_{i}'] = np.random.randn(2) * 0.1
    
    # Ball starts at center
    ball_position = np.array([field_width / 2, field_height / 2])
    ball_velocity = np.zeros(2)
    ball_possession = None
    
    goals_team_0 = 0
    goals_team_1 = 0
    
    for frame_idx in range(num_frames):
        # Simulate agent movement toward ball
        for agent_name, pos in agent_positions.items():
            # Move toward ball with some randomness
            to_ball = ball_position - pos
            direction = to_ball / (np.linalg.norm(to_ball) + 1e-6)
            
            # Add some strategy
            if 'team_0' in agent_name:
                # Team 0 also moves right
                direction += np.array([0.3, 0])
            else:
                # Team 1 also moves left
                direction -= np.array([0.3, 0])
            
            # Update velocity with momentum
            agent_velocities[agent_name] = (
                0.7 * agent_velocities[agent_name] + 
                0.3 * direction * 0.2 +
                np.random.randn(2) * 0.05
            )
            
            # Update position
            agent_positions[agent_name] = pos + agent_velocities[agent_name]
            
            # Clamp to field
            agent_positions[agent_name][0] = np.clip(agent_positions[agent_name][0], 0, field_width)
            agent_positions[agent_name][1] = np.clip(agent_positions[agent_name][1], 0, field_height)
        
        # Check possession
        closest_agent = None
        closest_dist = float('inf')
        for agent_name, pos in agent_positions.items():
            dist = np.linalg.norm(pos - ball_position)
            if dist < closest_dist:
                closest_dist = dist
                closest_agent = agent_name
        
        if closest_dist < 0.5:
            ball_possession = closest_agent
        
        # Move ball
        if ball_possession:
            ball_position = agent_positions[ball_possession].copy()
        else:
            ball_velocity *= 0.95
            ball_position += ball_velocity
            ball_position[0] = np.clip(ball_position[0], 0, field_width)
            ball_position[1] = np.clip(ball_position[1], 0, field_height)
        
        # Randomly shoot
        if ball_possession and np.random.random() < 0.02:
            if 'team_0' in ball_possession:
                # Shoot right
                ball_velocity = np.array([2.0, np.random.randn() * 0.5])
            else:
                # Shoot left
                ball_velocity = np.array([-2.0, np.random.randn() * 0.5])
            ball_possession = None
        
        # Check goals
        goal_scored = None
        if ball_position[0] <= 0.2 and abs(ball_position[1] - field_height/2) < 1.2:
            goals_team_1 += 1
            goal_scored = 'team_1'
            # Reset
            ball_position = np.array([field_width / 2, field_height / 2])
            ball_velocity = np.zeros(2)
        elif ball_position[0] >= field_width - 0.2 and abs(ball_position[1] - field_height/2) < 1.2:
            goals_team_0 += 1
            goal_scored = 'team_0'
            # Reset
            ball_position = np.array([field_width / 2, field_height / 2])
            ball_velocity = np.zeros(2)
        
        # Randomly pass
        pass_from = None
        pass_to = None
        if ball_possession and np.random.random() < 0.05:
            # Find teammate
            teammates = [a for a in agent_positions.keys() 
                        if ('team_0' in a) == ('team_0' in ball_possession) and a != ball_possession]
            if teammates:
                pass_from = ball_possession
                pass_to = np.random.choice(teammates)
                ball_position = agent_positions[pass_to].copy()
                ball_possession = pass_to
        
        # Store frame
        frame = {
            'frame_idx': frame_idx,
            'agent_positions': {k: v.tolist() for k, v in agent_positions.items()},
            'ball_position': ball_position.tolist(),
            'ball_possession': ball_possession,
            'stats': {
                'goals_team_0': goals_team_0,
                'goals_team_1': goals_team_1
            }
        }
        
        if pass_from and pass_to:
            frame['pass_from'] = pass_from
            frame['pass_to'] = pass_to
        
        replay['frames'].append(frame)
    
    return replay


def main():
    """Run all visualization examples"""
    print("="*70)
    print(" VISUALIZATION SUITE - EXAMPLE USAGE")
    print("="*70)
    
    # Create output directory
    output_dir = Path('visualization_outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Generate synthetic replays
    print("\n1. Generating synthetic replay data...")
    replay_before = generate_synthetic_replay(num_frames=100, seed=42)
    replay_after = generate_synthetic_replay(num_frames=100, seed=123)  # Different behavior
    
    # Save to JSON (optional)
    save_replay_to_json(replay_before, output_dir / 'replay_before.json')
    save_replay_to_json(replay_after, output_dir / 'replay_after.json')
    print("   ✅ Replay data generated and saved")
    
    # Visualization 1: Movement Trails
    print("\n2. Creating movement trails visualization...")
    plot_movement_trails(
        replay_before,
        save_path=output_dir / 'movement_trails.png',
        arrow_interval=10
    )
    
    # Visualization 2: Pass Network
    print("\n3. Creating pass network visualization...")
    plot_pass_network(
        replay_before,
        save_path=output_dir / 'pass_network.png'
    )
    
    # Visualization 3: Heatmaps
    print("\n4. Creating positional heatmaps...")
    plot_heatmaps(
        replay_before,
        save_dir=output_dir / 'heatmaps'
    )
    
    # Visualization 4: Episode Replay GIF
    print("\n5. Creating episode replay GIF...")
    print("   (This may take a minute...)")
    make_replay_gif(
        replay_before,
        save_path=output_dir / 'replay.gif',
        fps=10,
        show_trails=True
    )
    
    # Visualization 5: Before/After Comparison GIF
    print("\n6. Creating before/after comparison GIF...")
    print("   (This may take a minute...)")
    make_before_after_gif(
        replay_before,
        replay_after,
        save_path=output_dir / 'comparison.gif',
        fps=10
    )
    
    print("\n" + "="*70)
    print(" ✅ ALL VISUALIZATIONS COMPLETE!")
    print("="*70)
    print(f"\nOutputs saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  1. movement_trails.png - Agent trajectory visualization")
    print("  2. pass_network.png - Pass connections between agents")
    print("  3. heatmaps/all_heatmaps.png - Positional heatmaps")
    print("  4. replay.gif - Animated episode replay")
    print("  5. comparison.gif - Before/after side-by-side comparison")
    print("  6. replay_before.json - Replay data (before)")
    print("  7. replay_after.json - Replay data (after)")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
