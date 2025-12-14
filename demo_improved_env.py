"""
Visual demonstration of the improved football environment
Shows agent behaviors, movement patterns, and gameplay
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from env.improved_football_env import ImprovedFootballEnv, MOVE_RIGHT, MOVE_LEFT, MOVE_UP, MOVE_DOWN, SHOOT, PASS, STAY
import time

def visualize_state(env, ax, title=""):
    """Visualize current environment state"""
    ax.clear()
    
    # Draw field
    ax.set_xlim(-0.5, env.grid_width + 0.5)
    ax.set_ylim(-0.5, env.grid_height + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Draw goals
    goal_y_min = env.goal_center_y - env.goal_width / 2
    goal_y_max = env.goal_center_y + env.goal_width / 2
    
    # Left goal (Team 1's goal, Team 0 attacks this)
    ax.add_patch(Rectangle((0, goal_y_min), 0.2, env.goal_width, 
                           fill=True, color='red', alpha=0.3, label='Team 1 Goal'))
    # Right goal (Team 0's goal, Team 1 attacks this)
    ax.add_patch(Rectangle((env.grid_width - 0.2, goal_y_min), 0.2, env.goal_width, 
                           fill=True, color='blue', alpha=0.3, label='Team 0 Goal'))
    
    # Draw center line
    ax.axvline(env.grid_width / 2, color='gray', linestyle='--', alpha=0.5)
    
    # Draw agents
    for i, agent in enumerate(env.agents):
        pos = env.agent_positions[agent]
        vel = env.agent_velocities[agent]
        mode = env.agent_modes.get(agent, 'attack')
        
        # Color based on team
        if 'team_0' in agent:
            color = 'blue'
            marker = 'o'
        else:
            color = 'red'
            marker = 's'
        
        # Draw agent
        circle = Circle(pos, 0.3, color=color, alpha=0.7)
        ax.add_patch(circle)
        
        # Draw mode indicator
        mode_color = 'green' if mode == 'attack' else 'orange'
        ax.plot(pos[0], pos[1], marker=marker, markersize=15, 
               markeredgecolor=mode_color, markeredgewidth=2, alpha=0)
        
        # Draw velocity arrow
        if np.linalg.norm(vel) > 0.01:
            arrow = FancyArrowPatch(pos, pos + vel * 2, 
                                   color=color, arrowstyle='->', 
                                   mutation_scale=20, linewidth=2, alpha=0.6)
            ax.add_patch(arrow)
        
        # Label
        ax.text(pos[0], pos[1] - 0.5, agent.split('_')[-1], 
               ha='center', va='top', fontsize=8, color=color)
        ax.text(pos[0], pos[1] + 0.5, mode[0].upper(), 
               ha='center', va='bottom', fontsize=7, color=mode_color, weight='bold')
    
    # Draw ball
    ball_color = 'yellow' if env.ball_possession is None else 'gold'
    ball = Circle(env.ball_position, 0.2, color=ball_color, edgecolor='black', linewidth=2, zorder=10)
    ax.add_patch(ball)
    
    # If ball is possessed, draw connection
    if env.ball_possession:
        possessor_pos = env.agent_positions[env.ball_possession]
        ax.plot([possessor_pos[0], env.ball_position[0]], 
               [possessor_pos[1], env.ball_position[1]], 
               'k--', linewidth=1, alpha=0.5)
        ax.text(env.ball_position[0], env.ball_position[1] + 0.4, 
               env.ball_possession.split('_')[1], 
               ha='center', fontsize=7, color='black', weight='bold')
    
    # Title with stats
    stats = env.episode_stats
    ax.set_title(f"{title}\nGoals: {stats['goals_team_0']}-{stats['goals_team_1']} | "
                f"Possession: {env.ball_possession or 'None'} | "
                f"Kickoff timer: {env.kickoff_timer}", 
                fontsize=10)
    
    ax.set_xlabel('Field X')
    ax.set_ylabel('Field Y')

def run_demonstration(num_steps=100, delay=0.1):
    """Run a demonstration with visualization"""
    print("="*70)
    print(" IMPROVED FOOTBALL ENVIRONMENT - VISUAL DEMONSTRATION")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6, debug=True)
    obs, info = env.reset()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Initial state
    visualize_state(env, axes[0], "Initial Kickoff")
    
    # Run simulation
    for step in range(num_steps):
        agent = env.agent_selection
        team_id = 0 if 'team_0' in agent else 1
        
        # Smart action selection based on mode
        mode = env.agent_modes.get(agent, 'attack')
        pos = env.agent_positions[agent]
        
        # Simple heuristic policy
        if mode == 'attack':
            if env.ball_possession == agent:
                # Have ball, move toward goal or shoot if close
                goal_pos = np.array([env.grid_width, env.grid_height / 2]) if team_id == 0 else np.array([0, env.grid_height / 2])
                dist_to_goal = np.linalg.norm(pos - goal_pos)
                
                if dist_to_goal < 3.0 and np.random.random() < 0.3:
                    action = SHOOT
                elif np.random.random() < 0.2:
                    action = PASS
                else:
                    # Move toward goal
                    to_goal = goal_pos - pos
                    if abs(to_goal[0]) > abs(to_goal[1]):
                        action = MOVE_RIGHT if to_goal[0] > 0 else MOVE_LEFT
                    else:
                        action = MOVE_DOWN if to_goal[1] > 0 else MOVE_UP
            else:
                # Move toward ball
                to_ball = env.ball_position - pos
                if abs(to_ball[0]) > abs(to_ball[1]):
                    action = MOVE_RIGHT if to_ball[0] > 0 else MOVE_LEFT
                else:
                    action = MOVE_DOWN if to_ball[1] > 0 else MOVE_UP
        else:  # defend mode
            if env.ball_possession and ('team_0' in env.ball_possession) != (team_id == 0):
                # Intercept ball carrier
                opponent_pos = env.agent_positions[env.ball_possession]
                to_opponent = opponent_pos - pos
                if abs(to_opponent[0]) > abs(to_opponent[1]):
                    action = MOVE_RIGHT if to_opponent[0] > 0 else MOVE_LEFT
                else:
                    action = MOVE_DOWN if to_opponent[1] > 0 else MOVE_UP
            else:
                # Move toward ball
                to_ball = env.ball_position - pos
                if abs(to_ball[0]) > abs(to_ball[1]):
                    action = MOVE_RIGHT if to_ball[0] > 0 else MOVE_LEFT
                else:
                    action = MOVE_DOWN if to_ball[1] > 0 else MOVE_UP
        
        # Execute action
        obs, rewards, terms, truncs, infos = env.step(action)
        
        # Visualize every few steps
        if step % 5 == 0:
            visualize_state(env, axes[1], f"Step {step}")
            plt.pause(delay)
        
        if any(terms.values()) or any(truncs.values()):
            print(f"\nEpisode ended at step {step}")
            break
    
    # Final state
    visualize_state(env, axes[1], f"Final State (Step {step})")
    
    print("\n" + "="*70)
    print(" FINAL STATISTICS")
    print("="*70)
    stats = env.episode_stats
    print(f"Goals: Team 0: {stats['goals_team_0']}, Team 1: {stats['goals_team_1']}")
    print(f"Shots: Team 0: {stats['shots_team_0']}, Team 1: {stats['shots_team_1']}")
    print(f"Passes: {stats['successful_passes']}/{stats['passes']}")
    print(f"Interceptions: {stats['interceptions']}")
    print(f"Possession time: Team 0: {stats['possession_time'][0]}, Team 1: {stats['possession_time'][1]}")
    
    plt.tight_layout()
    plt.savefig('improved_env_demonstration.png', dpi=150, bbox_inches='tight')
    print("\n✅ Visualization saved to 'improved_env_demonstration.png'")
    plt.show()

if __name__ == "__main__":
    run_demonstration(num_steps=100, delay=0.1)
