"""
Complete Visualization Suite for MARL Football
Includes replay viewer, heatmaps, pass networks, and training curves
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.animation import FuncAnimation
import torch
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.football_env import FootballEnv
from models.ppo_agent import PPOAgent


# ============================================================================
# REPLAY VIEWER
# ============================================================================

class ReplayViewer:
    """Step-by-step episode replay viewer"""
    
    def __init__(self, checkpoint_path, render_mode='ascii'):
        self.render_mode = render_mode
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        # Create environment
        self.env = FootballEnv(
            num_agents_per_team=config['num_agents_per_team'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            max_steps=config['max_steps'],
            render_mode='ansi'
        )
        
        # Create agent
        agent = self.env.agents[0]
        obs_dim = self.env.observation_space(agent).shape[0]
        action_dim = self.env.action_space(agent).n
        
        self.agent = PPOAgent(obs_dim, action_dim)
        self.agent.actor.load_state_dict(checkpoint['team_0_agent']['actor'])
        self.agent.critic.load_state_dict(checkpoint['team_0_agent']['critic'])
        self.agent.actor.eval()
        self.agent.critic.eval()
    
    def replay_episode(self, num_steps=None):
        """Replay one episode"""
        observations, _ = self.env.reset()
        done = False
        step_count = 0
        
        frames = []
        
        while not done:
            # Render current state
            if self.render_mode == 'ascii':
                frame = self.env._render_ansi()
                print(frame)
                print(f"\nPress Enter for next step...")
                input()
            
            frames.append(self._capture_frame())
            
            # Get actions
            actions = {}
            for agent in self.env.agents:
                obs = observations[agent]
                action, _, _, _ = self.agent.get_action(obs, deterministic=True)
                actions[agent] = action
            
            # Step
            for agent in self.env.agent_iter():
                observation, reward, termination, truncation, info = \
                    self.env.step(actions[agent])
                observations[agent] = observation
                
                if termination or truncation:
                    done = True
                    break
            
            step_count += 1
            if num_steps and step_count >= num_steps:
                break
        
        if self.render_mode == 'ascii':
            print("\n" + "="*50)
            print("EPISODE COMPLETE")
            print(f"Final Score: {self.env.episode_stats['goals_team_0']}-"
                  f"{self.env.episode_stats['goals_team_1']}")
            print(f"Passes: {self.env.episode_stats['passes']}")
            print(f"Pass Success: {self.env.episode_stats['successful_passes']}/")
            print(f"{self.env.episode_stats['passes']}")
        
        return frames
    
    def _capture_frame(self):
        """Capture current environment state"""
        return {
            'agent_positions': self.env.agent_positions.copy(),
            'ball_position': self.env.ball_position.copy(),
            'ball_possession': self.env.ball_possession,
            'score': (self.env.episode_stats['goals_team_0'],
                     self.env.episode_stats['goals_team_1'])
        }


# ============================================================================
# HEATMAP GENERATOR
# ============================================================================

class HeatmapGenerator:
    """Generate position heatmaps for agents"""
    
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        self.env = FootballEnv(
            num_agents_per_team=config['num_agents_per_team'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            max_steps=config['max_steps']
        )
        
        agent = self.env.agents[0]
        obs_dim = self.env.observation_space(agent).shape[0]
        action_dim = self.env.action_space(agent).n
        
        self.agent = PPOAgent(obs_dim, action_dim)
        self.agent.actor.load_state_dict(checkpoint['team_0_agent']['actor'])
        self.agent.actor.eval()
        
        self.grid_width = config['grid_width']
        self.grid_height = config['grid_height']
    
    def generate_heatmaps(self, num_episodes=100):
        """Generate heatmaps for all agents"""
        # Position tracking
        position_counts = defaultdict(lambda: np.zeros((self.grid_height, self.grid_width)))
        
        for ep in range(num_episodes):
            observations, _ = self.env.reset()
            done = False
            
            while not done:
                # Track positions
                for agent_name, pos in self.env.agent_positions.items():
                    x, y = int(pos[0]), int(pos[1])
                    if 0 <= x < self.grid_width and 0 <= y < self.grid_height:
                        position_counts[agent_name][y, x] += 1
                
                # Get actions
                actions = {}
                for agent in self.env.agents:
                    obs = observations[agent]
                    action, _, _, _ = self.agent.get_action(obs, deterministic=True)
                    actions[agent] = action
                
                # Step
                for agent in self.env.agent_iter():
                    observation, reward, termination, truncation, info = \
                        self.env.step(actions[agent])
                    observations[agent] = observation
                    
                    if termination or truncation:
                        done = True
                        break
        
        return position_counts
    
    def plot_heatmaps(self, position_counts, save_path=None):
        """Plot heatmaps"""
        num_agents = len(position_counts)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (agent_name, counts) in enumerate(position_counts.items()):
            ax = axes[idx]
            
            # Normalize
            counts_norm = counts / (counts.max() + 1e-6)
            
            # Plot
            sns.heatmap(counts_norm, ax=ax, cmap='YlOrRd', 
                       xticklabels=False, yticklabels=False, cbar=True)
            ax.set_title(agent_name)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {save_path}")
        
        plt.show()


# ============================================================================
# PASS NETWORK ANALYZER
# ============================================================================

class PassNetworkAnalyzer:
    """Analyze and visualize passing patterns"""
    
    def __init__(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config = checkpoint['config']
        
        self.env = FootballEnv(
            num_agents_per_team=config['num_agents_per_team'],
            grid_width=config['grid_width'],
            grid_height=config['grid_height'],
            max_steps=config['max_steps']
        )
        
        agent = self.env.agents[0]
        obs_dim = self.env.observation_space(agent).shape[0]
        action_dim = self.env.action_space(agent).n
        
        self.agent = PPOAgent(obs_dim, action_dim)
        self.agent.actor.load_state_dict(checkpoint['team_0_agent']['actor'])
        self.agent.actor.eval()
    
    def collect_pass_data(self, num_episodes=100):
        """Collect passing statistics"""
        pass_matrix = defaultdict(lambda: defaultdict(int))
        touches = defaultdict(int)
        
        for ep in range(num_episodes):
            observations, _ = self.env.reset()
            done = False
            
            last_possession = None
            
            while not done:
                current_possession = self.env.ball_possession
                
                # Detect pass
                if last_possession and current_possession and \
                   last_possession != current_possession:
                    # Check if same team
                    last_team = 0 if 'team_0' in last_possession else 1
                    curr_team = 0 if 'team_0' in current_possession else 1
                    
                    if last_team == curr_team:
                        pass_matrix[last_possession][current_possession] += 1
                
                if current_possession:
                    touches[current_possession] += 1
                    last_possession = current_possession
                
                # Get actions
                actions = {}
                for agent in self.env.agents:
                    obs = observations[agent]
                    action, _, _, _ = self.agent.get_action(obs, deterministic=True)
                    actions[agent] = action
                
                # Step
                for agent in self.env.agent_iter():
                    observation, reward, termination, truncation, info = \
                        self.env.step(actions[agent])
                    observations[agent] = observation
                    
                    if termination or truncation:
                        done = True
                        break
        
        return pass_matrix, touches
    
    def plot_pass_network(self, pass_matrix, touches, save_path=None):
        """Visualize pass network using NetworkX"""
        G = nx.DiGraph()
        
        # Add nodes
        all_agents = set(touches.keys())
        for agent in all_agents:
            G.add_node(agent, touches=touches[agent])
        
        # Add edges
        for src in pass_matrix:
            for dst in pass_matrix[src]:
                weight = pass_matrix[src][dst]
                if weight > 0:
                    G.add_edge(src, dst, weight=weight)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw
        plt.figure(figsize=(12, 8))
        
        # Node sizes based on touches
        node_sizes = [touches[node] * 10 for node in G.nodes()]
        
        # Node colors by team
        node_colors = ['skyblue' if 'team_0' in node else 'salmon' 
                      for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color=node_colors, alpha=0.7)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        nx.draw_networkx_edges(G, pos, width=[w/max_weight*5 for w in weights],
                              alpha=0.5, edge_color='gray',
                              arrows=True, arrowsize=20)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        
        plt.title('Pass Network (Node size = touches, Edge width = pass frequency)')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Pass network saved to {save_path}")
        
        plt.show()
        
        # Print statistics
        print("\n=== Pass Network Statistics ===")
        print(f"Total touches: {sum(touches.values())}")
        print(f"Total passes: {sum(sum(dst.values()) for dst in pass_matrix.values())}")
        
        # Centrality
        if len(G.edges()) > 0:
            betweenness = nx.betweenness_centrality(G)
            print("\nMost central players (betweenness):")
            for agent, score in sorted(betweenness.items(), 
                                      key=lambda x: x[1], reverse=True)[:3]:
                print(f"  {agent}: {score:.3f}")


# ============================================================================
# TRAINING CURVE PLOTTER
# ============================================================================

class TrainingCurvePlotter:
    """Plot training metrics from TensorBoard logs"""
    
    def __init__(self, logdir):
        self.logdir = logdir
        
        # Try to import tensorboard
        try:
            from tensorboard.backend.event_processing import event_accumulator
            self.ea = event_accumulator.EventAccumulator(logdir)
            self.ea.Reload()
        except ImportError:
            print("TensorBoard not installed. Install with: pip install tensorboard")
            self.ea = None
    
    def plot_metrics(self, metrics, save_path=None):
        """Plot specified metrics"""
        if not self.ea:
            print("Cannot plot: TensorBoard not available")
            return
        
        num_metrics = len(metrics)
        fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4*num_metrics))
        
        if num_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            try:
                events = self.ea.Scalars(metric)
                steps = [e.step for e in events]
                values = [e.value for e in events]
                
                axes[idx].plot(steps, values, alpha=0.3, label='Raw')
                
                # Smoothed
                window = min(100, len(values) // 10)
                if window > 1:
                    smoothed = np.convolve(values, 
                                          np.ones(window)/window, 
                                          mode='valid')
                    axes[idx].plot(steps[window-1:], smoothed, 
                                  label=f'Smoothed ({window})')
                
                axes[idx].set_xlabel('Episode')
                axes[idx].set_ylabel(metric)
                axes[idx].set_title(metric)
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                
            except KeyError:
                axes[idx].text(0.5, 0.5, f'Metric not found: {metric}',
                              ha='center', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()


# ============================================================================
# MAIN CLI FUNCTIONS
# ============================================================================

def main_replay():
    """CLI for replay viewer"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--render_mode', default='ascii')
    args = parser.parse_args()
    
    viewer = ReplayViewer(args.checkpoint, args.render_mode)
    for i in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{args.episodes}")
        print(f"{'='*60}\n")
        viewer.replay_episode()


def main_heatmap():
    """CLI for heatmap generator"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', default='heatmaps.png')
    args = parser.parse_args()
    
    generator = HeatmapGenerator(args.checkpoint)
    counts = generator.generate_heatmaps(args.episodes)
    generator.plot_heatmaps(counts, args.output)


def main_pass_network():
    """CLI for pass network analyzer"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--output', default='pass_network.png')
    args = parser.parse_args()
    
    analyzer = PassNetworkAnalyzer(args.checkpoint)
    pass_matrix, touches = analyzer.collect_pass_data(args.episodes)
    analyzer.plot_pass_network(pass_matrix, touches, args.output)


def main_plots():
    """CLI for training curves"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', required=True)
    parser.add_argument('--output', default='training_curves.png')
    args = parser.parse_args()
    
    plotter = TrainingCurvePlotter(args.logdir)
    metrics = [
        'Episode/team_0_reward',
        'Episode/win_rate',
        'Episode/pass_success_rate',
        'Train/policy_loss_team_0',
        'Train/entropy_team_0'
    ]
    plotter.plot_metrics(metrics, args.output)


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visualization.py <command>")
        print("Commands: replay, heatmap, pass_network, plots")
        sys.exit(1)
    
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == 'replay':
        main_replay()
    elif command == 'heatmap':
        main_heatmap()
    elif command == 'pass_network':
        main_pass_network()
    elif command == 'plots':
        main_plots()
    else:
        print(f"Unknown command: {command}")