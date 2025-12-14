"""Quick test to verify app.py model loading works"""
import yaml
import torch
from env.football_env import FootballEnv
from models.ppo_agent import PPOAgent

print("=" * 60)
print("Testing Streamlit App Model Loading Fix")
print("=" * 60)

# Load config
config_path = 'configs/stage1_stable.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Load checkpoint to detect obs dim
checkpoint_path = 'runs/run_20251212_233217/checkpoints/best_model.pt'
print(f"\n1️⃣  Loading checkpoint: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Detect observation dimension
actor_state = checkpoint['team_0_agent']['actor']
if 'input_proj.0.weight' in actor_state:
    trained_obs_dim = actor_state['input_proj.0.weight'].shape[1]
    print(f"   ✓ Model observation dimension: {trained_obs_dim}")

# Create matching environment
print(f"\n2️⃣  Creating environment with matching observation space...")
env = FootballEnv(
    num_agents_per_team=config['num_agents_per_team'],
    grid_width=config['grid_width'],
    grid_height=config['grid_height'],
    max_steps=config.get('max_steps', 150)
)

obs, _ = env.reset()
agent_name = env.agents[0]
obs_dim = obs[agent_name].shape[0]
action_dim = env.action_space(agent_name).n

print(f"   ✓ Environment observation dimension: {obs_dim}")
print(f"   ✓ Action space dimension: {action_dim}")

# Verify dimensions match
if obs_dim != trained_obs_dim:
    print(f"\n❌ DIMENSION MISMATCH!")
    print(f"   Model expects: {trained_obs_dim}")
    print(f"   Environment provides: {obs_dim}")
    exit(1)

print(f"\n3️⃣  Creating PPO agent...")
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
print(f"   ✓ Agent created successfully")

print(f"\n4️⃣  Loading model weights...")
try:
    team_agent_state = checkpoint['team_0_agent']
    agent.actor.load_state_dict(team_agent_state['actor'])
    agent.actor.eval()
    print(f"   ✓ Model weights loaded successfully!")
except RuntimeError as e:
    print(f"   ❌ Error loading weights: {e}")
    exit(1)

print(f"\n5️⃣  Testing inference...")
obs_tensor = torch.FloatTensor(obs[agent_name]).unsqueeze(0)
with torch.no_grad():
    action_probs = agent.actor(obs_tensor)
    action = torch.argmax(action_probs, dim=1).item()
print(f"   ✓ Inference works! Action: {action}")

print("\n" + "=" * 60)
print("✅✅✅ ALL TESTS PASSED! ✅✅✅")
print("=" * 60)
print("\nThe Streamlit app should now work correctly!")
print("Run: streamlit run app.py")
print("=" * 60)
