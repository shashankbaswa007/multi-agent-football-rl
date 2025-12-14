"""
FastAPI Backend Server for Multi-Agent RL Football Demo
========================================================

Provides REST API endpoints for:
1. Running simulations with trained policy or random fallback
2. Retrieving saved replays
3. Real-time episode generation

Run with: uvicorn fastapi_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
from pathlib import Path
import json
import uuid
from datetime import datetime
import torch
import numpy as np

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from env.football_env import FootballEnv
    from models.ppo_agent import PPOAgent
    ENV_AVAILABLE = True
except ImportError:
    ENV_AVAILABLE = False
    print("Warning: Could not import environment. Using fallback simulation only.")

from replay_schema import ReplayWriter, ReplayReader, generate_example_replay, convert_numpy_types

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent RL Football API",
    description="REST API for multi-agent football simulation and replay",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key for production (optional security)
API_KEY = os.getenv("FOOTBALL_API_KEY", "dev-key-12345")

# Global model cache
MODEL_CACHE = {}


# ============================================================================
# Pydantic Models
# ============================================================================

class SimulationRequest(BaseModel):
    """Request model for simulation endpoint."""
    scenario: str = "3v3"  # "1v1", "2v2", "3v3"
    num_steps: int = 100
    seed: int = 42
    use_trained_model: bool = False
    model_checkpoint: Optional[str] = None  # Path to .pt file


class SimulationResponse(BaseModel):
    """Response model for simulation endpoint."""
    replay_id: str
    scenario: str
    num_steps: int
    final_score: List[int]
    replay_data: Dict[str, Any]


class ReplayListResponse(BaseModel):
    """Response model for listing replays."""
    replays: List[Dict[str, Any]]


# ============================================================================
# Helper Functions
# ============================================================================

def verify_api_key(x_api_key: Optional[str] = Header(None)):
    """Verify API key for production security."""
    if os.getenv("ENVIRONMENT") == "production":
        if x_api_key != API_KEY:
            raise HTTPException(status_code=403, detail="Invalid API key")


def load_model(checkpoint_path: str):
    """Load trained PPO model from checkpoint."""
    if checkpoint_path in MODEL_CACHE:
        return MODEL_CACHE[checkpoint_path]
    
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=404, detail=f"Model checkpoint not found: {checkpoint_path}")
    
    try:
        # Initialize environment to get observation space
        env = FootballEnv()
        obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
        action_dim = env.action_space(env.possible_agents[0]).n
        
        # Initialize agent
        agent = PPOAgent(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=512,
            lr=3e-4
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.actor.eval()
        agent.critic.eval()
        
        MODEL_CACHE[checkpoint_path] = agent
        return agent
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


def run_simulation_with_model(
    scenario: str,
    num_steps: int,
    seed: int,
    agent: Any  # PPOAgent type
) -> Dict[str, Any]:
    """Run simulation using trained model."""
    if not ENV_AVAILABLE:
        raise HTTPException(status_code=501, detail="Environment not available. Install required packages.")
    
    env = FootballEnv()
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reset environment
    env.reset(seed=seed)
    
    writer = ReplayWriter(scenario=scenario, seed=seed)
    
    step = 0
    episode_done = False
    
    while step < num_steps and not episode_done:
        agents_data = []
        reward_totals = {0: 0, 1: 0}
        
        # Collect actions for all agents
        for agent_name in env.agents:
            obs = env.observe(agent_name)
            
            if obs is None:
                continue
            
            # Get action from policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.get_action(obs_tensor)
            
            # Step environment
            env.step(action.item())
            
            # Get reward
            reward = env.rewards.get(agent_name, 0)
            
            # Get agent state
            agent_idx = int(agent_name.split('_')[-1])
            team = 0 if agent_idx < 3 else 1
            
            # Get position from environment state
            if hasattr(env, 'agent_positions'):
                position = env.agent_positions.get(agent_name, [0, 0])
            else:
                position = [0, 0]  # Fallback
            
            has_ball = (env.ball_carrier == agent_name) if hasattr(env, 'ball_carrier') else False
            
            agents_data.append({
                'agent_id': agent_name,
                'team': team,
                'position': list(position),
                'action': action.item(),
                'action_name': ['Hold', 'Up', 'Down', 'Left', 'Right', 'Pass', 'Shoot'][action.item()],
                'reward': float(reward),
                'has_ball': has_ball
            })
            
            reward_totals[team] += reward
        
        # Get ball position and score
        ball_position = env.ball_position if hasattr(env, 'ball_position') else [6, 4]
        score = env.score if hasattr(env, 'score') else [0, 0]
        
        reward_breakdown = {
            'team0_total': reward_totals[0],
            'team1_total': reward_totals[1]
        }
        
        writer.add_timestep(
            step=step,
            agents_data=agents_data,
            ball_position=list(ball_position),
            score=list(score),
            episode_done=len(env.agents) == 0,
            reward_breakdown=reward_breakdown
        )
        
        step += 1
        episode_done = len(env.agents) == 0
    
    # Build replay data
    metadata = {
        'replay_id': writer.replay_id,
        'timestamp': datetime.now().isoformat(),
        'scenario': scenario,
        'num_agents': len(writer.timesteps[0].agents) if writer.timesteps else 0,
        'teams': [0, 1],
        'agent_names': [a['agent_id'] for a in writer.timesteps[0].agents] if writer.timesteps else [],
        'seed': seed,
        'total_steps': len(writer.timesteps),
        'final_score': score,
        'winner': 0 if score[0] > score[1] else (1 if score[1] > score[0] else None)
    }
    
    replay_data = {
        'metadata': metadata,
        'timesteps': [
            {
                'step': ts.step,
                'agents': ts.agents,
                'ball_position': ts.ball_position,
                'score': ts.score,
                'episode_done': ts.episode_done,
                'reward_breakdown': ts.reward_breakdown
            }
            for ts in writer.timesteps
        ]
    }
    
    return convert_numpy_types(replay_data)


def run_fallback_simulation(
    scenario: str,
    num_steps: int,
    seed: int
) -> Dict[str, Any]:
    """Run simulation with random policy (fallback)."""
    return generate_example_replay(scenario=scenario, num_steps=num_steps, seed=seed)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Multi-Agent RL Football API",
        "version": "1.0.0",
        "endpoints": {
            "POST /simulate": "Run a new simulation",
            "GET /replay/{replay_id}": "Get a saved replay",
            "GET /replays": "List all saved replays",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environment_available": ENV_AVAILABLE,
        "model_cache_size": len(MODEL_CACHE)
    }


@app.post("/simulate", response_model=SimulationResponse)
async def simulate(
    request: SimulationRequest,
    background_tasks: BackgroundTasks,
    x_api_key: Optional[str] = Header(None)
):
    """
    Run a simulation and return replay data.
    
    Args:
        request: Simulation configuration
        x_api_key: API key for authentication (production only)
    
    Returns:
        Replay data with metadata and timesteps
    """
    # Verify API key in production
    if os.getenv("ENVIRONMENT") == "production":
        verify_api_key(x_api_key)
    
    # Validate scenario
    if request.scenario not in ["1v1", "2v2", "3v3"]:
        raise HTTPException(status_code=400, detail="Invalid scenario. Must be '1v1', '2v2', or '3v3'")
    
    # Run simulation
    try:
        if request.use_trained_model and request.model_checkpoint:
            # Use trained model
            agent = load_model(request.model_checkpoint)
            replay_data = run_simulation_with_model(
                scenario=request.scenario,
                num_steps=request.num_steps,
                seed=request.seed,
                agent=agent
            )
        else:
            # Use fallback random simulation
            replay_data = run_fallback_simulation(
                scenario=request.scenario,
                num_steps=request.num_steps,
                seed=request.seed
            )
        
        # Save replay in background
        replay_id = replay_data['metadata']['replay_id']
        replay_path = Path("replays") / f"{replay_id}.json"
        
        def save_replay():
            with open(replay_path, 'w') as f:
                json.dump(replay_data, f, indent=2)
        
        background_tasks.add_task(save_replay)
        
        return SimulationResponse(
            replay_id=replay_id,
            scenario=request.scenario,
            num_steps=len(replay_data['timesteps']),
            final_score=replay_data['metadata']['final_score'],
            replay_data=replay_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@app.get("/replay/{replay_id}")
async def get_replay(replay_id: str):
    """
    Get a saved replay by ID.
    
    Args:
        replay_id: Unique replay identifier
    
    Returns:
        Replay data
    """
    replay_path = Path("replays") / f"{replay_id}.json"
    
    if not replay_path.exists():
        # Try with different extensions
        candidates = list(Path("replays").glob(f"*{replay_id}*.json"))
        if candidates:
            replay_path = candidates[0]
        else:
            raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")
    
    try:
        with open(replay_path, 'r') as f:
            replay_data = json.load(f)
        return JSONResponse(content=replay_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading replay: {str(e)}")


@app.get("/replays", response_model=ReplayListResponse)
async def list_replays():
    """
    List all saved replays.
    
    Returns:
        List of replay metadata
    """
    replay_dir = Path("replays")
    replay_dir.mkdir(exist_ok=True)
    
    replays = []
    for replay_file in replay_dir.glob("*.json"):
        try:
            with open(replay_file, 'r') as f:
                data = json.load(f)
                replays.append({
                    "filename": replay_file.name,
                    "replay_id": data['metadata']['replay_id'],
                    "scenario": data['metadata']['scenario'],
                    "timestamp": data['metadata']['timestamp'],
                    "final_score": data['metadata']['final_score'],
                    "total_steps": data['metadata']['total_steps']
                })
        except Exception as e:
            print(f"Error reading {replay_file}: {e}")
            continue
    
    return ReplayListResponse(replays=replays)


@app.delete("/replay/{replay_id}")
async def delete_replay(replay_id: str, x_api_key: Optional[str] = Header(None)):
    """
    Delete a saved replay.
    
    Args:
        replay_id: Unique replay identifier
        x_api_key: API key for authentication (production only)
    """
    # Verify API key in production
    if os.getenv("ENVIRONMENT") == "production":
        verify_api_key(x_api_key)
    
    replay_path = Path("replays") / f"{replay_id}.json"
    
    if not replay_path.exists():
        raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")
    
    try:
        replay_path.unlink()
        return {"message": f"Replay {replay_id} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting replay: {str(e)}")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    # Create replays directory
    Path("replays").mkdir(exist_ok=True)
    print("🚀 FastAPI server started")
    print(f"Environment available: {ENV_AVAILABLE}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("👋 FastAPI server shutting down")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
