"""
Multi-Agent RL Football Visualization - Streamlit Demo
======================================================

Interactive visualization dashboard for watching multi-agent reinforcement learning
agents play football. Features play/pause/step controls, heatmaps, pass networks,
agent statistics, and reward decomposition.

IMPROVED VERSION: Uses matplotlib for better real-time rendering and visibility.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import json
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyArrow
from collections import defaultdict
import io
from PIL import Image
import imageio

# Page config
st.set_page_config(
    page_title="Multi-Agent RL Football Demo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
FIELD_WIDTH = 12
FIELD_HEIGHT = 8
AGENT_RADIUS = 0.3
BALL_RADIUS = 0.15

# Team colors
TEAM_COLORS = {
    0: '#FF4444',  # Red
    1: '#4444FF'   # Blue
}

ACTION_NAMES = {
    0: 'Hold',
    1: 'Up ↑',
    2: 'Down ↓',
    3: 'Left ←',
    4: 'Right →',
    5: 'Pass ⚽',
    6: 'Shoot 🎯'
}


def load_replay(filepath):
    """Load replay JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def render_field_matplotlib(timestep_data, show_trails=False, trail_history=None, 
                           show_heatmap=False, heatmap_data=None, heatmap_team=None,
                           show_pass_network=False, pass_network_data=None):
    """
    Render football field using Matplotlib for better real-time performance.
    
    Args:
        timestep_data: Dict with agents, ball_position, score
        show_trails: Show agent movement trails
        trail_history: List of past agent positions
        show_heatmap: Show position heatmap overlay
        heatmap_data: Precomputed heatmap data
        heatmap_team: Team to show heatmap for (0 or 1)
        show_pass_network: Show pass network overlay
        pass_network_data: Dict with pass counts between agents
    """
    # Create figure with better DPI for clarity
    fig, ax = plt.subplots(figsize=(14, 9), dpi=100)
    
    # Draw field background
    field = Rectangle((0, 0), FIELD_WIDTH, FIELD_HEIGHT, 
                      facecolor='#2ECC40', edgecolor='white', linewidth=3)
    ax.add_patch(field)
    
    # Draw center line
    ax.plot([FIELD_WIDTH/2, FIELD_WIDTH/2], [0, FIELD_HEIGHT], 
            'w--', linewidth=2, alpha=0.7)
    
    # Draw center circle
    center_circle = Circle((FIELD_WIDTH/2, FIELD_HEIGHT/2), 1.5, 
                          fill=False, edgecolor='white', linewidth=2, alpha=0.7)
    ax.add_patch(center_circle)
    
    # Draw goals
    goal_width = 0.3
    goal_height = FIELD_HEIGHT / 3
    goal_y = (FIELD_HEIGHT - goal_height) / 2
    
    # Left goal
    left_goal = Rectangle((-goal_width, goal_y), goal_width, goal_height,
                         facecolor='none', edgecolor='white', linewidth=3)
    ax.add_patch(left_goal)
    
    # Right goal
    right_goal = Rectangle((FIELD_WIDTH, goal_y), goal_width, goal_height,
                          facecolor='none', edgecolor='white', linewidth=3)
    ax.add_patch(right_goal)
    
    # Show heatmap if requested
    if show_heatmap and heatmap_data is not None and heatmap_team is not None:
        team_heatmap = heatmap_data[heatmap_team]
        extent = [0, FIELD_WIDTH, 0, FIELD_HEIGHT]
        ax.imshow(team_heatmap, extent=extent, origin='lower', 
                 cmap='hot', alpha=0.4, aspect='auto', zorder=1)
    
    # Draw trails if requested
    if show_trails and trail_history:
        for agent_id, positions in trail_history.items():
            if len(positions) > 1:
                x_coords = [p[0] for p in positions]
                y_coords = [p[1] for p in positions]
                ax.plot(x_coords, y_coords, 'gray', alpha=0.3, linewidth=2, zorder=2)
    
    # Show pass network if requested
    if show_pass_network and pass_network_data:
        agents = timestep_data['agents']
        agent_positions = {a['agent_id']: a['position'] for a in agents}
        
        for (agent_a, agent_b), count in pass_network_data.items():
            if agent_a in agent_positions and agent_b in agent_positions:
                pos_a = agent_positions[agent_a]
                pos_b = agent_positions[agent_b]
                
                # Draw arrow with width proportional to pass count
                width = min(0.15, 0.03 + count * 0.02)
                dx = pos_b[0] - pos_a[0]
                dy = pos_b[1] - pos_a[1]
                
                arrow = FancyArrow(pos_a[0], pos_a[1], dx*0.9, dy*0.9,
                                  width=width, head_width=width*2, 
                                  head_length=0.2, fc='yellow', 
                                  ec='orange', alpha=0.6, zorder=3)
                ax.add_patch(arrow)
    
    # Draw agents with clear visibility - ALWAYS VISIBLE
    agents = timestep_data['agents']
    for agent in agents:
        pos = agent['position']
        team = agent['team']
        has_ball = agent.get('has_ball', False)
        
        # Agent circle with high zorder to be on top
        color = TEAM_COLORS[team]
        agent_size = AGENT_RADIUS * 1.3 if has_ball else AGENT_RADIUS
        
        # Draw outer glow if has ball for better visibility
        if has_ball:
            glow = Circle(pos, agent_size * 1.5, 
                         facecolor='yellow', alpha=0.3, zorder=4)
            ax.add_patch(glow)
        
        # Agent body - solid and visible
        agent_circle = Circle(pos, agent_size, 
                            facecolor=color, edgecolor='white' if not has_ball else 'yellow',
                            linewidth=3 if has_ball else 2, zorder=5)
        ax.add_patch(agent_circle)
        
        # Agent label (number) - always visible
        agent_num = agent['agent_id'].split('_')[-1]
        ax.text(pos[0], pos[1], agent_num, 
               ha='center', va='center', 
               fontsize=14, fontweight='bold', 
               color='white', zorder=6,
               bbox=dict(boxstyle='circle,pad=0.1', facecolor=color, alpha=0.8, edgecolor='none'))
    
    # Draw ball with MAXIMUM visibility - ALWAYS ON TOP
    ball_pos = timestep_data['ball_position']
    
    # Ball shadow for depth
    ball_shadow = Circle((ball_pos[0]+0.05, ball_pos[1]-0.05), BALL_RADIUS * 1.3, 
                        facecolor='black', alpha=0.3, zorder=7)
    ax.add_patch(ball_shadow)
    
    # Ball itself with bright white color
    ball = Circle(ball_pos, BALL_RADIUS, 
                 facecolor='white', edgecolor='black', 
                 linewidth=3, zorder=10)  # Highest zorder!
    ax.add_patch(ball)
    
    # Add small pattern to ball for visibility
    ax.plot([ball_pos[0] - BALL_RADIUS*0.5, ball_pos[0] + BALL_RADIUS*0.5],
           [ball_pos[1], ball_pos[1]], 'k-', linewidth=2, zorder=11)
    ax.plot([ball_pos[0], ball_pos[0]],
           [ball_pos[1] - BALL_RADIUS*0.5, ball_pos[1] + BALL_RADIUS*0.5], 
           'k-', linewidth=2, zorder=11)
    
    # Scoreboard at top with larger text
    score = timestep_data['score']
    ax.text(FIELD_WIDTH/2, FIELD_HEIGHT + 0.8, 
           f"Team 0 (🔴): {score[0]}  |  Team 1 (🔵): {score[1]}",
           ha='center', va='center', fontsize=22, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.7', facecolor='white', 
                    edgecolor='black', linewidth=3))
    
    # Set axis properties
    ax.set_xlim(-0.5, FIELD_WIDTH + 0.5)
    ax.set_ylim(-0.5, FIELD_HEIGHT + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def compute_heatmap(replay_data, grid_size=20):
    """Compute position heatmap for each team from replay data."""
    timesteps = replay_data['timesteps']
    
    heatmaps = {
        0: np.zeros((grid_size, grid_size)),
        1: np.zeros((grid_size, grid_size))
    }
    
    for ts in timesteps:
        for agent in ts['agents']:
            team = agent['team']
            pos = agent['position']
            
            # Convert to grid coordinates
            x_idx = int((pos[0] / FIELD_WIDTH) * grid_size)
            y_idx = int((pos[1] / FIELD_HEIGHT) * grid_size)
            
            x_idx = max(0, min(grid_size-1, x_idx))
            y_idx = max(0, min(grid_size-1, y_idx))
            
            heatmaps[team][y_idx, x_idx] += 1
    
    # Normalize
    for team in [0, 1]:
        max_val = heatmaps[team].max()
        if max_val > 0:
            heatmaps[team] = heatmaps[team] / max_val
    
    return heatmaps


def compute_pass_network(replay_data):
    """Compute pass network (agent -> agent pass counts)."""
    timesteps = replay_data['timesteps']
    pass_counts = defaultdict(int)
    
    for i in range(len(timesteps) - 1):
        ts = timesteps[i]
        
        for agent in ts['agents']:
            if agent['action_name'] == 'Pass ⚽':
                # Find who receives the pass
                next_ts = timesteps[i + 1]
                passer_team = agent['team']
                passer_id = agent['agent_id']
                
                for next_agent in next_ts['agents']:
                    if next_agent['team'] == passer_team and next_agent.get('has_ball', False):
                        receiver_id = next_agent['agent_id']
                        if receiver_id != passer_id:
                            pass_counts[(passer_id, receiver_id)] += 1
                        break
    
    return dict(pass_counts)


def compute_agent_stats(replay_data):
    """Compute per-agent statistics across the replay."""
    timesteps = replay_data['timesteps']
    agent_stats = defaultdict(lambda: {
        'total_reward': 0,
        'pass_count': 0,
        'shot_count': 0,
        'possession_time': 0,
        'actions': defaultdict(int)
    })
    
    for ts in timesteps:
        for agent in ts['agents']:
            agent_id = agent['agent_id']
            stats = agent_stats[agent_id]
            
            stats['total_reward'] += agent['reward']
            stats['actions'][agent['action_name']] += 1
            
            if agent['action_name'] == 'Pass ⚽':
                stats['pass_count'] += 1
            if agent['action_name'] == 'Shoot 🎯':
                stats['shot_count'] += 1
            if agent.get('has_ball', False):
                stats['possession_time'] += 1
    
    return dict(agent_stats)


def generate_gif(replay_data, output_path, fps=10):
    """Generate GIF from replay data using matplotlib."""
    timesteps = replay_data['timesteps']
    frames = []
    
    for ts in timesteps:
        fig = render_field_matplotlib(ts)
        
        # Convert to image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        buf.seek(0)
        img = Image.open(buf)
        frames.append(np.array(img))
        plt.close(fig)
        buf.close()
    
    # Save GIF
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("⚽ Multi-Agent RL Football Visualization")
    st.markdown("*Watch AI agents learn to play football together*")
    
    # Sidebar controls
    st.sidebar.header("🎮 Controls")
    
    # Replay selection
    replay_dir = Path("replays")
    replay_files = list(replay_dir.glob("*.json"))
    
    if not replay_files:
        st.error("No replay files found in 'replays/' directory. Generate some first!")
        st.info("Run: `python replay_schema.py` to generate example replays")
        return
    
    replay_file = st.sidebar.selectbox(
        "Select Replay",
        replay_files,
        format_func=lambda x: x.name
    )
    
    # Load replay
    if 'replay_data' not in st.session_state or st.session_state.get('current_replay') != str(replay_file):
        st.session_state.replay_data = load_replay(replay_file)
        st.session_state.current_replay = str(replay_file)
        st.session_state.current_step = 0
        st.session_state.playing = False
        st.session_state.trail_history = defaultdict(list)
        st.session_state.heatmaps = compute_heatmap(st.session_state.replay_data)
        st.session_state.pass_network = compute_pass_network(st.session_state.replay_data)
        st.session_state.agent_stats = compute_agent_stats(st.session_state.replay_data)
    
    replay_data = st.session_state.replay_data
    metadata = replay_data['metadata']
    timesteps = replay_data['timesteps']
    total_steps = len(timesteps)
    
    # Display metadata
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Replay Info")
    st.sidebar.write(f"**Scenario:** {metadata['scenario']}")
    st.sidebar.write(f"**Total Steps:** {total_steps}")
    st.sidebar.write(f"**Final Score:** {metadata['final_score'][0]} - {metadata['final_score'][1]}")
    winner = metadata.get('winner')
    if winner is not None:
        st.sidebar.write(f"**Winner:** Team {winner} 🏆")
    else:
        st.sidebar.write("**Winner:** Draw")
    
    # Playback controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏯️ Playback")
    
    col1, col2, col3 = st.sidebar.columns(3)
    
    with col1:
        if st.button("⏮️ Reset"):
            st.session_state.current_step = 0
            st.session_state.playing = False
            st.session_state.trail_history = defaultdict(list)
            st.rerun()
    
    with col2:
        if st.button("⏸️ Pause" if st.session_state.playing else "▶️ Play"):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    
    with col3:
        if st.button("⏭️ Step"):
            if st.session_state.current_step < total_steps - 1:
                st.session_state.current_step += 1
            st.session_state.playing = False
            st.rerun()
    
    # Speed control
    speed = st.sidebar.slider("Playback Speed", 0.1, 3.0, 1.0, 0.1)
    
    # Step slider
    new_step = st.sidebar.slider(
        "Current Step",
        0, total_steps - 1,
        st.session_state.current_step
    )
    
    if new_step != st.session_state.current_step:
        st.session_state.current_step = new_step
        st.session_state.playing = False
    
    # Visualization options
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Visualization")
    
    show_trails = st.sidebar.checkbox("Show Trails", value=False)
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
    if show_heatmap:
        heatmap_team = st.sidebar.radio("Heatmap Team", [0, 1])
    else:
        heatmap_team = None
    
    show_pass_network = st.sidebar.checkbox("Show Pass Network", value=False)
    
    # Export options
    st.sidebar.markdown("---")
    if st.sidebar.button("💾 Export Replay as GIF"):
        with st.spinner("Generating GIF..."):
            gif_path = f"replays/{metadata['replay_id']}.gif"
            generate_gif(replay_data, gif_path, fps=10)
            st.sidebar.success(f"✅ Saved to {gif_path}")
    
    # Main content area
    current_step = st.session_state.current_step
    timestep_data = timesteps[current_step]
    
    # Update trail history
    if show_trails:
        for agent in timestep_data['agents']:
            agent_id = agent['agent_id']
            st.session_state.trail_history[agent_id].append(agent['position'])
            # Keep only last 20 positions
            if len(st.session_state.trail_history[agent_id]) > 20:
                st.session_state.trail_history[agent_id].pop(0)
    
    # Render field with matplotlib - IMPROVED FOR VISIBILITY
    fig = render_field_matplotlib(
        timestep_data,
        show_trails=show_trails,
        trail_history=st.session_state.trail_history if show_trails else None,
        show_heatmap=show_heatmap,
        heatmap_data=st.session_state.heatmaps if show_heatmap else None,
        heatmap_team=heatmap_team,
        show_pass_network=show_pass_network,
        pass_network_data=st.session_state.pass_network if show_pass_network else None
    )
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    
    # Agent details and stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👥 Agent States")
        for agent in timestep_data['agents']:
            team_color = "🔴" if agent['team'] == 0 else "🔵"
            ball_icon = "⚽" if agent.get('has_ball', False) else ""
            st.markdown(
                f"**{team_color} {agent['agent_id']}** {ball_icon}  \n"
                f"Action: {agent['action_name']} | Reward: {agent['reward']:.3f}"
            )
    
    with col2:
        st.markdown("### 📈 Reward Breakdown")
        breakdown = timestep_data.get('reward_breakdown', {})
        for key, value in breakdown.items():
            st.write(f"**{key}:** {value:.2f}")
    
    # Agent statistics table
    st.markdown("---")
    st.markdown("### 📊 Cumulative Agent Statistics")
    
    stats = st.session_state.agent_stats
    stats_data = []
    for agent_id, agent_stats in stats.items():
        stats_data.append({
            'Agent': agent_id,
            'Total Reward': f"{agent_stats['total_reward']:.2f}",
            'Passes': agent_stats['pass_count'],
            'Shots': agent_stats['shot_count'],
            'Possession Time': agent_stats['possession_time']
        })
    
    st.table(stats_data)
    
    # Auto-play logic
    if st.session_state.playing:
        time.sleep(0.1 / speed)  # Base delay adjusted by speed
        if st.session_state.current_step < total_steps - 1:
            st.session_state.current_step += 1
            st.rerun()
        else:
            st.session_state.playing = False
            st.rerun()


if __name__ == "__main__":
    main()
