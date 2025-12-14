"""
Complete Visualization Suite for Multi-Agent Football RL
========================================================

Six high-impact visualizations:
1. Movement Trails - Agent trajectories over time
2. Pass Network Overlay - Pass connections between agents
3. Heatmaps - Positional heatmaps for agents and ball
4. Episode Replay GIF - Animated replay of episode
5. Before/After Comparison GIF - Side-by-side comparison

All functions accept replay data in standard format and produce publication-quality figures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch, Arrow
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import defaultdict
import imageio
from pathlib import Path
import json

# Try to import networkx, provide fallback
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Pass network visualization will use simple matplotlib.")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def draw_field(ax, field_width=10, field_height=6, goal_width=2.4):
    """Draw football field with goals, center line, and boundaries"""
    ax.set_xlim(-0.5, field_width + 0.5)
    ax.set_ylim(-0.5, field_height + 0.5)
    ax.set_aspect('equal')
    
    # Field boundary
    field_rect = Rectangle((0, 0), field_width, field_height, 
                           fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(field_rect)
    
    # Center line
    ax.axvline(field_width / 2, color='white', linestyle='--', linewidth=1, alpha=0.6)
    
    # Center circle
    center_circle = Circle((field_width / 2, field_height / 2), 
                           radius=field_height * 0.2, 
                           fill=False, edgecolor='white', linewidth=1, alpha=0.6)
    ax.add_patch(center_circle)
    
    # Goals
    goal_y_min = (field_height - goal_width) / 2
    goal_y_max = goal_y_min + goal_width
    
    # Left goal (Team 0 defends, Team 1 attacks)
    left_goal = Rectangle((0, goal_y_min), -0.3, goal_width,
                          fill=True, facecolor='cyan', edgecolor='white', 
                          linewidth=2, alpha=0.5)
    ax.add_patch(left_goal)
    
    # Right goal (Team 1 defends, Team 0 attacks)
    right_goal = Rectangle((field_width, goal_y_min), 0.3, goal_width,
                           fill=True, facecolor='red', edgecolor='white',
                           linewidth=2, alpha=0.5)
    ax.add_patch(right_goal)
    
    ax.set_facecolor('#2d5016')  # Grass green
    ax.set_xlabel('Field X', color='white', fontsize=10)
    ax.set_ylabel('Field Y', color='white', fontsize=10)
    ax.tick_params(colors='white')
    

def extract_positions_from_replay(replay):
    """Extract position data from replay dictionary"""
    positions = defaultdict(list)
    ball_positions = []
    
    for frame in replay['frames']:
        for agent_name, pos in frame['agent_positions'].items():
            positions[agent_name].append(pos)
        ball_positions.append(frame['ball_position'])
    
    return positions, ball_positions


def get_team_color(agent_name):
    """Get color for agent based on team"""
    if 'team_0' in agent_name:
        return '#4A90E2'  # Blue
    else:
        return '#E24A4A'  # Red


# ============================================================================
# VISUALIZATION 1: MOVEMENT TRAILS
# ============================================================================

def plot_movement_trails(replay, save_path='movement_trails.png', 
                        arrow_interval=10, figsize=(12, 8)):
    """
    Plot movement trails for all agents showing their trajectories over time.
    
    Parameters:
    -----------
    replay : dict
        Replay data with frames containing agent positions
    save_path : str
        Path to save the figure
    arrow_interval : int
        Add direction arrows every N steps
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d5016')
    
    field_width = replay.get('field_width', 10)
    field_height = replay.get('field_height', 6)
    draw_field(ax, field_width, field_height)
    
    positions, ball_positions = extract_positions_from_replay(replay)
    
    # Plot each agent's trail
    for agent_name, pos_list in positions.items():
        if len(pos_list) < 2:
            continue
            
        pos_array = np.array(pos_list)
        color = get_team_color(agent_name)
        
        # Draw trail
        ax.plot(pos_array[:, 0], pos_array[:, 1], 
               color=color, linewidth=2, alpha=0.7, 
               label=agent_name)
        
        # Add arrows at intervals
        for i in range(0, len(pos_array) - 1, arrow_interval):
            if i + 1 < len(pos_array):
                dx = pos_array[i+1, 0] - pos_array[i, 0]
                dy = pos_array[i+1, 1] - pos_array[i, 1]
                if np.sqrt(dx**2 + dy**2) > 0.1:  # Only draw if moved
                    ax.arrow(pos_array[i, 0], pos_array[i, 1], dx, dy,
                            head_width=0.2, head_length=0.15,
                            fc=color, ec=color, alpha=0.8, linewidth=0)
        
        # Mark start and end
        ax.plot(pos_array[0, 0], pos_array[0, 1], 'o', 
               color=color, markersize=12, markeredgecolor='white', 
               markeredgewidth=2, label=f'{agent_name} start')
        ax.plot(pos_array[-1, 0], pos_array[-1, 1], 's',
               color=color, markersize=12, markeredgecolor='white',
               markeredgewidth=2, label=f'{agent_name} end')
    
    # Plot ball trail
    if ball_positions:
        ball_array = np.array(ball_positions)
        ax.plot(ball_array[:, 0], ball_array[:, 1],
               color='yellow', linewidth=3, alpha=0.6, 
               linestyle=':', label='Ball trail')
        ax.plot(ball_array[0, 0], ball_array[0, 1], 'o',
               color='yellow', markersize=10, markeredgecolor='black',
               markeredgewidth=2)
    
    ax.set_title('Agent Movement Trails', color='white', fontsize=16, weight='bold')
    
    # Legend outside plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
             facecolor='#1a1a1a', edgecolor='white', 
             labelcolor='white', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"✅ Movement trails saved to {save_path}")
    
    return fig, ax


# ============================================================================
# VISUALIZATION 2: PASS NETWORK OVERLAY
# ============================================================================

def plot_pass_network(replay, save_path='pass_network.png', figsize=(14, 8)):
    """
    Plot pass network showing connections between agents.
    
    Parameters:
    -----------
    replay : dict
        Replay data with frames containing pass events
    save_path : str
        Path to save the figure
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#2d5016')
    
    field_width = replay.get('field_width', 10)
    field_height = replay.get('field_height', 6)
    draw_field(ax, field_width, field_height)
    
    # Build pass adjacency matrix
    pass_counts = defaultdict(lambda: defaultdict(int))
    agent_positions_avg = defaultdict(lambda: [0, 0, 0])  # [sum_x, sum_y, count]
    
    for frame in replay['frames']:
        # Track average positions
        for agent_name, pos in frame['agent_positions'].items():
            agent_positions_avg[agent_name][0] += pos[0]
            agent_positions_avg[agent_name][1] += pos[1]
            agent_positions_avg[agent_name][2] += 1
        
        # Count passes
        if 'pass_from' in frame and 'pass_to' in frame:
            pass_counts[frame['pass_from']][frame['pass_to']] += 1
    
    # Calculate average positions
    avg_positions = {}
    for agent, sums in agent_positions_avg.items():
        avg_positions[agent] = [sums[0] / sums[2], sums[1] / sums[2]]
    
    if HAS_NETWORKX:
        # Use NetworkX for better layout
        G = nx.DiGraph()
        
        # Add nodes
        for agent in avg_positions.keys():
            G.add_node(agent, pos=avg_positions[agent])
        
        # Add edges with weights
        for from_agent, to_dict in pass_counts.items():
            for to_agent, count in to_dict.items():
                G.add_edge(from_agent, to_agent, weight=count)
        
        pos = nx.get_node_attributes(G, 'pos')
        
        # Draw nodes
        for agent, position in pos.items():
            color = get_team_color(agent)
            circle = Circle(position, 0.3, color=color, ec='white', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(position[0], position[1], agent.split('_')[-1],
                   ha='center', va='center', color='white', 
                   fontsize=10, weight='bold', zorder=11)
        
        # Draw edges
        for (from_agent, to_agent), count in nx.get_edge_attributes(G, 'weight').items():
            from_pos = pos[from_agent]
            to_pos = pos[to_agent]
            
            # Arrow thickness based on pass count
            width = min(count * 0.5, 5)
            
            arrow = FancyArrowPatch(from_pos, to_pos,
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=width, color='yellow',
                                   alpha=0.6, zorder=5)
            ax.add_patch(arrow)
            
            # Add pass count label
            mid_x = (from_pos[0] + to_pos[0]) / 2
            mid_y = (from_pos[1] + to_pos[1]) / 2
            ax.text(mid_x, mid_y, str(count), fontsize=9, 
                   color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    else:
        # Fallback: simple matplotlib implementation
        for agent, position in avg_positions.items():
            color = get_team_color(agent)
            circle = Circle(position, 0.3, color=color, ec='white', linewidth=2, zorder=10)
            ax.add_patch(circle)
            ax.text(position[0], position[1], agent.split('_')[-1],
                   ha='center', va='center', color='white',
                   fontsize=10, weight='bold', zorder=11)
        
        # Draw pass arrows
        for from_agent, to_dict in pass_counts.items():
            for to_agent, count in to_dict.items():
                if from_agent in avg_positions and to_agent in avg_positions:
                    from_pos = avg_positions[from_agent]
                    to_pos = avg_positions[to_agent]
                    
                    width = min(count * 0.5, 5)
                    arrow = FancyArrowPatch(from_pos, to_pos,
                                           arrowstyle='->', mutation_scale=20,
                                           linewidth=width, color='yellow',
                                           alpha=0.6, zorder=5)
                    ax.add_patch(arrow)
                    
                    mid_x = (from_pos[0] + to_pos[0]) / 2
                    mid_y = (from_pos[1] + to_pos[1]) / 2
                    ax.text(mid_x, mid_y, str(count), fontsize=9,
                           color='white', weight='bold',
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    
    ax.set_title('Pass Network (Arrow thickness = pass frequency)', 
                color='white', fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"✅ Pass network saved to {save_path}")
    
    return fig, ax


# ============================================================================
# VISUALIZATION 3: HEATMAPS
# ============================================================================

def plot_heatmaps(replay, save_dir='heatmaps', figsize=(18, 5)):
    """
    Generate positional heatmaps for teams and ball.
    
    Parameters:
    -----------
    replay : dict
        Replay data
    save_dir : str
        Directory to save heatmap figures
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    field_width = replay.get('field_width', 10)
    field_height = replay.get('field_height', 6)
    
    # Extract positions
    positions, ball_positions = extract_positions_from_replay(replay)
    
    # Separate by team
    team_0_positions = []
    team_1_positions = []
    
    for agent_name, pos_list in positions.items():
        if 'team_0' in agent_name:
            team_0_positions.extend(pos_list)
        else:
            team_1_positions.extend(pos_list)
    
    team_0_positions = np.array(team_0_positions)
    team_1_positions = np.array(team_1_positions)
    ball_positions_array = np.array(ball_positions)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.patch.set_facecolor('#1a1a1a')
    
    titles = ['Team 0 (Blue) Heatmap', 'Team 1 (Red) Heatmap', 'Ball Heatmap']
    data_arrays = [team_0_positions, team_1_positions, ball_positions_array]
    cmaps = ['Blues', 'Reds', 'YlOrRd']
    
    for idx, (ax, title, data, cmap) in enumerate(zip(axes, titles, data_arrays, cmaps)):
        ax.set_facecolor('#2d5016')
        draw_field(ax, field_width, field_height)
        
        if len(data) > 0:
            # Create 2D histogram
            H, xedges, yedges = np.histogram2d(
                data[:, 0], data[:, 1],
                bins=[int(field_width * 4), int(field_height * 4)],
                range=[[0, field_width], [0, field_height]]
            )
            
            # Plot heatmap
            im = ax.imshow(H.T, origin='lower', extent=[0, field_width, 0, field_height],
                          cmap=cmap, alpha=0.7, interpolation='bilinear')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Density', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        ax.set_title(title, color='white', fontsize=12, weight='bold')
    
    plt.tight_layout()
    
    # Save individual heatmaps
    save_path = Path(save_dir) / 'all_heatmaps.png'
    plt.savefig(save_path, dpi=150, facecolor='#1a1a1a', bbox_inches='tight')
    print(f"✅ Heatmaps saved to {save_path}")
    
    return fig, axes


# ============================================================================
# VISUALIZATION 4: EPISODE REPLAY GIF
# ============================================================================

def make_replay_gif(replay, save_path='replay.gif', fps=10, show_trails=True, trail_length=20):
    """
    Create animated GIF of episode replay.
    
    Parameters:
    -----------
    replay : dict
        Replay data
    save_path : str
        Path to save GIF
    fps : int
        Frames per second
    show_trails : bool
        Whether to show recent movement trails
    trail_length : int
        Number of recent frames to show in trail
    
    Returns:
    --------
    None (saves GIF to file)
    """
    field_width = replay.get('field_width', 10)
    field_height = replay.get('field_height', 6)
    frames = replay['frames']
    
    images = []
    
    for frame_idx, frame in enumerate(frames):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#2d5016')
        
        draw_field(ax, field_width, field_height)
        
        # Draw trails if enabled
        if show_trails and frame_idx > 0:
            start_idx = max(0, frame_idx - trail_length)
            for agent_name in frame['agent_positions'].keys():
                trail_positions = []
                for past_frame in frames[start_idx:frame_idx]:
                    if agent_name in past_frame['agent_positions']:
                        trail_positions.append(past_frame['agent_positions'][agent_name])
                
                if len(trail_positions) > 1:
                    trail_array = np.array(trail_positions)
                    color = get_team_color(agent_name)
                    ax.plot(trail_array[:, 0], trail_array[:, 1],
                           color=color, linewidth=1, alpha=0.3)
        
        # Draw agents
        for agent_name, pos in frame['agent_positions'].items():
            color = get_team_color(agent_name)
            circle = Circle(pos, 0.3, color=color, ec='white', linewidth=2, zorder=10)
            ax.add_patch(circle)
            
            # Agent number
            agent_num = agent_name.split('_')[-1]
            ax.text(pos[0], pos[1], agent_num, ha='center', va='center',
                   color='white', fontsize=10, weight='bold', zorder=11)
        
        # Draw ball
        ball_pos = frame['ball_position']
        ball_circle = Circle(ball_pos, 0.2, color='yellow', ec='black', 
                            linewidth=2, zorder=12)
        ax.add_patch(ball_circle)
        
        # Add frame info
        stats = frame.get('stats', {})
        info_text = f"Frame: {frame_idx}/{len(frames)}\n"
        info_text += f"Score: {stats.get('goals_team_0', 0)}-{stats.get('goals_team_1', 0)}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
               color='white', weight='bold')
        
        ax.set_title(f'Episode Replay - Frame {frame_idx}', 
                    color='white', fontsize=14, weight='bold')
        
        # Convert to image
        fig.canvas.draw()
        # Use buffer_rgba() for compatibility
        buf = np.array(fig.canvas.buffer_rgba())
        image = buf[:, :, :3]  # Drop alpha channel
        images.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, images, fps=fps)
    print(f"✅ Replay GIF saved to {save_path} ({len(images)} frames @ {fps} fps)")


# ============================================================================
# VISUALIZATION 5: BEFORE/AFTER COMPARISON GIF
# ============================================================================

def make_before_after_gif(replay_before, replay_after, save_path='comparison.gif', 
                         fps=10, show_trails=True):
    """
    Create side-by-side comparison GIF of two replays.
    
    Parameters:
    -----------
    replay_before : dict
        First replay (e.g., before training)
    replay_after : dict
        Second replay (e.g., after training)
    save_path : str
        Path to save GIF
    fps : int
        Frames per second
    show_trails : bool
        Show movement trails
    
    Returns:
    --------
    None (saves GIF to file)
    """
    field_width = replay_before.get('field_width', 10)
    field_height = replay_before.get('field_height', 6)
    
    frames_before = replay_before['frames']
    frames_after = replay_after['frames']
    
    # Use minimum length
    num_frames = min(len(frames_before), len(frames_after))
    
    images = []
    
    for frame_idx in range(num_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Draw both frames
        for ax, frames, title_suffix in [(ax1, frames_before, 'BEFORE'),
                                         (ax2, frames_after, 'AFTER')]:
            ax.set_facecolor('#2d5016')
            draw_field(ax, field_width, field_height)
            
            frame = frames[frame_idx]
            
            # Draw trails
            if show_trails and frame_idx > 0:
                start_idx = max(0, frame_idx - 20)
                for agent_name in frame['agent_positions'].keys():
                    trail_positions = []
                    for past_frame in frames[start_idx:frame_idx]:
                        if agent_name in past_frame['agent_positions']:
                            trail_positions.append(past_frame['agent_positions'][agent_name])
                    
                    if len(trail_positions) > 1:
                        trail_array = np.array(trail_positions)
                        color = get_team_color(agent_name)
                        ax.plot(trail_array[:, 0], trail_array[:, 1],
                               color=color, linewidth=1, alpha=0.3)
            
            # Draw agents
            for agent_name, pos in frame['agent_positions'].items():
                color = get_team_color(agent_name)
                circle = Circle(pos, 0.3, color=color, ec='white', linewidth=2, zorder=10)
                ax.add_patch(circle)
                ax.text(pos[0], pos[1], agent_name.split('_')[-1],
                       ha='center', va='center', color='white',
                       fontsize=10, weight='bold', zorder=11)
            
            # Draw ball
            ball_pos = frame['ball_position']
            ball_circle = Circle(ball_pos, 0.2, color='yellow', ec='black',
                                linewidth=2, zorder=12)
            ax.add_patch(ball_circle)
            
            # Stats
            stats = frame.get('stats', {})
            info_text = f"{title_suffix}\nFrame: {frame_idx}\n"
            info_text += f"Score: {stats.get('goals_team_0', 0)}-{stats.get('goals_team_1', 0)}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   color='white', weight='bold')
            
            ax.set_title(title_suffix, color='white', fontsize=14, weight='bold')
        
        plt.suptitle(f'Before vs After Comparison - Frame {frame_idx}',
                    color='white', fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        # Use buffer_rgba() for compatibility
        buf = np.array(fig.canvas.buffer_rgba())
        image = buf[:, :, :3]  # Drop alpha channel
        images.append(image)
        
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(save_path, images, fps=fps)
    print(f"✅ Before/After comparison GIF saved to {save_path} ({len(images)} frames @ {fps} fps)")


# ============================================================================
# UTILITY: LOAD REPLAY FROM JSON
# ============================================================================

def load_replay_from_json(filepath):
    """Load replay data from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_replay_to_json(replay, filepath):
    """Save replay data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(replay, f, indent=2)
