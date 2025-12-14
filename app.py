"""
Live Trained Model Demo - Streamlit Interactive Visualization
=============================================================

Watch your trained reinforcement learning agents play football in real-time!
Features continuous play, live statistics, and interactive controls.

Run with: streamlit run app.py
"""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
from pathlib import Path
import time
import sys
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from easy_start_env import EasyStartEnv
from env.football_env import FootballEnv
from models.ppo_agent import PPOAgent

# Page config
st.set_page_config(
    page_title="🏆 Trained Model Demo",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:24px !important;
    font-weight: bold;
}
.stat-box {
    background-color: #f0f2f6;
    padding: 15px;
    border-radius: 10px;
    margin: 10px 0;
}
.goal-alert {
    background-color: #FFD700;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
    animation: pulse 0.5s ease-in-out;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'env' not in st.session_state:
    st.session_state.env = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'step_count' not in st.session_state:
    st.session_state.step_count = 0
if 'goals_team0' not in st.session_state:
    st.session_state.goals_team0 = 0
if 'goals_team1' not in st.session_state:
    st.session_state.goals_team1 = 0
if 'episode_reward' not in st.session_state:
    st.session_state.episode_reward = 0
if 'last_goal_scorer' not in st.session_state:
    st.session_state.last_goal_scorer = None


@st.cache_resource
def load_trained_model(checkpoint_path, config_path, use_curriculum=True):
    """Load trained model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # First, load checkpoint to detect observation dimension
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Detect observation dimension from model architecture
    try:
        actor_state = checkpoint['team_0_agent']['actor']
        if 'input_proj.0.weight' in actor_state:
            trained_obs_dim = actor_state['input_proj.0.weight'].shape[1]
        elif 'fc1.weight' in actor_state:
            trained_obs_dim = actor_state['fc1.weight'].shape[1]
        else:
            st.error("Could not detect model observation dimension")
            trained_obs_dim = 11
    except Exception as e:
        st.warning(f"Could not auto-detect obs dim: {e}. Using default 11.")
        trained_obs_dim = 11
    
    # Create environment matching the trained model
    # 11 dim = FootballEnv without goal directions
    # 13 dim = FootballEnv with 2 agents and goal directions
    # 17 dim = FootballEnv with 3 agents and goal directions
    # 21 dim = ImprovedFootballEnv
    
    if trained_obs_dim == 21:
        # Use ImprovedFootballEnv
        try:
            from env.improved_football_env import ImprovedFootballEnv
            env = ImprovedFootballEnv(
                num_agents_per_team=config.get('num_agents_per_team', 2),
                grid_width=config.get('grid_width', 10),
                grid_height=config.get('grid_height', 6)
            )
            st.info("✓ Loaded ImprovedFootballEnv (21-dim observations)")
        except ImportError:
            st.error("ImprovedFootballEnv not found. Using FootballEnv instead.")
            env = FootballEnv(
                num_agents_per_team=config['num_agents_per_team'],
                grid_width=config['grid_width'],
                grid_height=config['grid_height'],
                max_steps=config.get('max_steps', 150)
            )
    else:
        # Use original FootballEnv
        if use_curriculum:
            env = EasyStartEnv(
                num_agents_per_team=config['num_agents_per_team'],
                grid_width=config['grid_width'],
                grid_height=config['grid_height'],
                max_steps=config.get('max_steps', 150),
                difficulty='easy'
            )
        else:
            env = FootballEnv(
                num_agents_per_team=config['num_agents_per_team'],
                grid_width=config['grid_width'],
                grid_height=config['grid_height'],
                max_steps=config.get('max_steps', 150)
            )
        st.info(f"✓ Loaded FootballEnv ({trained_obs_dim}-dim observations)")
    
    obs, _ = env.reset()
    agent_name = env.agents[0]
    obs_dim = obs[agent_name].shape[0]
    action_dim = env.action_space(agent_name).n
    
    # Verify dimensions match
    if obs_dim != trained_obs_dim:
        st.error(f"⚠️ Mismatch: Model expects {trained_obs_dim}-dim obs, environment provides {obs_dim}-dim")
        st.error("This model may not work correctly. Try a different checkpoint.")
    else:
        st.success(f"✓ Observation dimensions match: {obs_dim}")
    
    # Create agent
    ppo_params = config.get('ppo_params', {})
    agent = PPOAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        lr=ppo_params.get('lr', 0.0003),
        gamma=ppo_params.get('gamma', 0.99),
        clip_epsilon=ppo_params.get('clip_epsilon', 0.2),
        value_loss_coef=ppo_params.get('value_loss_coef', 0.5),
        entropy_coef=ppo_params.get('entropy_coef', 0.01)
    )
    
    # Load checkpoint state
    team_agent_state = checkpoint['team_0_agent']
    
    try:
        agent.actor.load_state_dict(team_agent_state['actor'])
        agent.actor.eval()
        st.success("✓ Model weights loaded successfully!")
    except RuntimeError as e:
        st.error(f"Error loading model weights: {e}")
        st.error("The checkpoint may be incompatible with the current environment configuration.")
        raise
    
    return agent, env, config


def render_field(env, show_info=True):
    """Render the football field with agents and ball."""
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
    
    # Field background
    field = Rectangle((0, 0), env.grid_width, env.grid_height, 
                      facecolor='#2ECC40', edgecolor='white', linewidth=3)
    ax.add_patch(field)
    
    # Center line
    ax.plot([env.grid_width/2, env.grid_width/2], [0, env.grid_height], 
            'w--', linewidth=2, alpha=0.7)
    
    # Center circle
    center_circle = Circle((env.grid_width/2, env.grid_height/2), 1.5, 
                          fill=False, edgecolor='white', linewidth=2, alpha=0.7)
    ax.add_patch(center_circle)
    
    # Goals
    goal_height = env.grid_height / 3
    goal_y = (env.grid_height - goal_height) / 2
    
    left_goal = Rectangle((-0.3, goal_y), 0.3, goal_height,
                         facecolor='yellow', edgecolor='white', linewidth=2)
    right_goal = Rectangle((env.grid_width, goal_y), 0.3, goal_height,
                          facecolor='yellow', edgecolor='white', linewidth=2)
    ax.add_patch(left_goal)
    ax.add_patch(right_goal)
    
    # Draw agents
    for agent_name in env.agents:
        pos = env.agent_positions[agent_name]
        team = 0 if 'team_0' in agent_name else 1
        color = '#FF4444' if team == 0 else '#4444FF'
        
        # Agent circle
        agent_circle = Circle(pos, 0.3, facecolor=color, edgecolor='white', 
                            linewidth=2, zorder=10)
        ax.add_patch(agent_circle)
        
        # Agent number
        agent_num = agent_name.split('_')[-1]
        ax.text(pos[0], pos[1], agent_num, color='white', 
               fontsize=14, fontweight='bold', ha='center', va='center', zorder=11)
        
        # Show possession indicator
        if env.ball_possession == agent_name:
            possession_ring = Circle(pos, 0.45, fill=False, edgecolor='gold', 
                                   linewidth=3, zorder=9)
            ax.add_patch(possession_ring)
    
    # Draw ball
    if env.ball_possession is None:
        ball_circle = Circle(env.ball_position, 0.15, facecolor='white', 
                           edgecolor='black', linewidth=2, zorder=12)
        ax.add_patch(ball_circle)
    
    # Labels
    ax.text(env.grid_width/2, -0.5, 'TRAINED MODEL DEMO', 
           fontsize=20, fontweight='bold', ha='center', color='white')
    ax.text(1, -0.5, f'Team 0 (Red): {st.session_state.goals_team0}', 
           fontsize=14, color='#FF4444', fontweight='bold')
    ax.text(env.grid_width - 1, -0.5, f'Team 1 (Blue): {st.session_state.goals_team1}', 
           fontsize=14, color='#4444FF', fontweight='bold')
    
    ax.set_xlim(-1, env.grid_width + 1)
    ax.set_ylim(-1, env.grid_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return fig


def main():
    st.title("⚽ Live Trained Model Demo")
    st.markdown("### Watch Your RL Agents Play Football!")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎮 Controls")
        
        # Model selection
        runs_dir = Path("runs")
        checkpoint_path = None
        config_path = Path("configs/stage1_stable.yaml")
        
        if runs_dir.exists():
            run_dirs = sorted(runs_dir.glob("run_*"))
            if run_dirs:
                selected_run = st.selectbox(
                    "Select Training Run",
                    [r.name for r in run_dirs],
                    index=len(run_dirs)-1
                )
                
                checkpoint_dir = runs_dir / selected_run / "checkpoints"
                checkpoints = []
                if (checkpoint_dir / "best_model.pt").exists():
                    checkpoints.append("best_model.pt")
                if (checkpoint_dir / "final_model.pt").exists():
                    checkpoints.append("final_model.pt")
                checkpoints.extend([f.name for f in sorted(checkpoint_dir.glob("episode_*.pt"))])
                
                if checkpoints:
                    selected_checkpoint = st.selectbox("Select Checkpoint", checkpoints, index=0)
                    checkpoint_path = checkpoint_dir / selected_checkpoint
            else:
                st.warning("No training runs found in 'runs/' directory")
        else:
            st.error("'runs/' directory not found")
        
        # Environment settings
        st.subheader("🏟️ Environment")
        use_curriculum = st.checkbox("Use Curriculum (Easy Start)", value=True)
        
        # Load model button
        if st.button("🔄 Load Model", use_container_width=True, disabled=checkpoint_path is None):
            with st.spinner("🔍 Detecting model architecture..."):
                try:
                    agent, env, config = load_trained_model(
                        str(checkpoint_path),
                        str(config_path),
                        use_curriculum
                    )
                    st.session_state.agent = agent
                    st.session_state.env = env
                    st.session_state.config = config
                    obs, _ = env.reset()
                    st.session_state.obs = obs
                    st.session_state.step_count = 0
                    st.session_state.goals_team0 = 0
                    st.session_state.goals_team1 = 0
                    st.session_state.episode_reward = 0
                    st.success("✅ Model loaded and ready to play!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")
                    st.error("Make sure the checkpoint is compatible with the environment.")
                    import traceback
                    with st.expander("🔍 View full error trace"):
                        st.code(traceback.format_exc())
        
        st.divider()
        
        # Playback controls
        st.subheader("▶️ Playback")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Play", disabled=st.session_state.agent is None, use_container_width=True):
                st.session_state.running = True
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.running = False
        
        if st.button("🔄 Reset Episode", disabled=st.session_state.agent is None, use_container_width=True):
            obs, _ = st.session_state.env.reset()
            st.session_state.obs = obs
            st.session_state.step_count = 0
            st.session_state.goals_team0 = 0
            st.session_state.goals_team1 = 0
            st.session_state.episode_reward = 0
            st.session_state.last_goal_scorer = None
            st.rerun()
        
        speed = st.slider("Speed", 0.1, 2.0, 0.5, 0.1)
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Steps", st.session_state.step_count)
        st.metric("Episode Reward", f"{st.session_state.episode_reward:.1f}")
        st.metric("Goals Team 0", st.session_state.goals_team0)
        st.metric("Goals Team 1", st.session_state.goals_team1)
    
    # Main display area
    if st.session_state.agent is None:
        st.info("👈 Load a trained model from the sidebar to begin!")
        
        # Show helpful getting started info
        st.markdown("""
        ### Getting Started
        
        1. **Select a training run** from the sidebar dropdown
        2. **Choose a checkpoint** (recommend `best_model.pt`)
        3. **Check "Use Curriculum"** if using Stage 1 model
        4. **Click "🔄 Load Model"**
        5. **Click "▶️ Play"** to watch your agents!
        
        ---
        
        #### Expected Model Location
        The app looks for trained models in: `runs/run_*/checkpoints/*.pt`
        
        If you see "No training runs found", make sure you have:
        - Completed training with `train_ppo.py`
        - Saved checkpoints in the `runs/` directory
        
        #### Recommended Checkpoint
        - `best_model.pt` - Best performing model during training ⭐
        - `final_model.pt` - Last epoch (may have overfitted)
        """)
        return
    
    # Show goal alert
    if st.session_state.last_goal_scorer is not None:
        team = "Team 0 (Red)" if 'team_0' in st.session_state.last_goal_scorer else "Team 1 (Blue)"
        st.markdown(f'<div class="goal-alert">🎉 GOAL! {team} scored! 🎉</div>', unsafe_allow_html=True)
        time.sleep(1)
        st.session_state.last_goal_scorer = None
    
    # Display field
    field_placeholder = st.empty()
    
    # Auto-play loop
    if st.session_state.running:
        env = st.session_state.env
        agent = st.session_state.agent
        
        # Get current agent
        agent_name = env.agent_selection
        
        # Check if done
        if env.terminations[agent_name] or env.truncations[agent_name]:
            env.step(None)
            # Update observations
            st.session_state.obs = env._get_observations()
            
            # Check if episode truly complete
            if all(env.terminations.values()) or all(env.truncations.values()):
                st.session_state.running = False
                st.success("Episode complete!")
                st.rerun()
            else:
                # Continue to next agent
                st.rerun()
        else:
            # Get observation for current agent
            obs = st.session_state.obs[agent_name]
            
            # Get action from model
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_probs = agent.actor(obs_tensor)
                action = torch.argmax(action_probs, dim=1).item()
            
            # Execute action
            env.step(action)
            reward = env.rewards[agent_name]
            
            # Update observations
            st.session_state.obs = env._get_observations()
            
            # Update stats
            if 'team_0' in agent_name:
                st.session_state.episode_reward += reward
                
                # Check for goal (big reward spike)
                if reward > 90:
                    st.session_state.goals_team0 += 1
                    st.session_state.last_goal_scorer = agent_name
                    st.rerun()
            elif reward < -90:
                st.session_state.goals_team1 += 1
                st.session_state.last_goal_scorer = agent_name
                st.rerun()
            
            st.session_state.step_count += 1
            
            # Render
            fig = render_field(env)
            field_placeholder.pyplot(fig)
            plt.close(fig)
            
            # Control speed
            time.sleep(1.0 / speed)
            st.rerun()
    else:
        # Show current state
        fig = render_field(st.session_state.env)
        field_placeholder.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
