# ⚽ Multi-Agent RL Football Visualization

**Watch AI agents learn to play football together through interactive, real-time visualizations.**

A complete demo system for visualizing multi-agent reinforcement learning in a custom 3v3 football environment. Features both a fast Streamlit prototype for quick demos and a production-ready FastAPI + React stack for integration into larger applications.

![Demo Screenshot](docs/screenshot.png) *(Place screenshot here)*

## 🎯 Features

### Visualization
- **Real-time playback** with play/pause/step controls and adjustable speed (0.1x - 3x)
- **Position heatmaps** showing where each team spends time on the field
- **Pass networks** visualizing passing patterns between teammates
- **Agent trails** tracking movement history
- **Scoreboard** with live score updates
- **Per-agent stats** including rewards, passes, shots, and possession time
- **Reward decomposition** showing how rewards are calculated

### Deployment Options
- **Streamlit Prototype**: Fast, interactive demo ready in 30 seconds
- **FastAPI Backend**: Production REST API with model serving
- **React Frontend**: Modern, responsive UI with Canvas rendering
- **Docker Support**: One-command deployment with docker-compose

### Smart Fallbacks
- Works without trained models using intelligent random policies
- Generates realistic football behavior for instant demos
- Gracefully handles missing dependencies

---

## 📁 Repository Structure

```
demo/
├── streamlit_app.py           # Streamlit demo (QUICKSTART)
├── replay_schema.py            # Replay JSON format, reader/writer
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Multi-container orchestration
├── Procfile                    # Heroku deployment
│
├── backend/
│   └── fastapi_server.py      # REST API server
│
├── frontend/
│   ├── package.json           # Node dependencies
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js             # Main React app
│       ├── App.css            # Styles
│       ├── index.js           # React entry point
│       └── components/
│           └── FieldCanvas.jsx # Canvas renderer
│
├── replays/                   # Saved replay JSON files
│   ├── example_1v1.json
│   ├── example_2v2.json
│   └── example_3v3.json
│
└── tests/
    └── test_replay.py         # Unit tests
```

---

## 🚀 Quick Start

### Option 1: Streamlit Demo (30 seconds)

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
cd demo
pip install -r requirements.txt

# 3. Generate example replays (if not already done)
python replay_schema.py

# 4. Run Streamlit app
streamlit run streamlit_app.py
```

**Open browser to http://localhost:8501** and start watching agents play!

### Option 2: FastAPI + React (Full Stack)

**Terminal 1 - Backend:**
```bash
cd demo
source ../.venv/bin/activate
uvicorn backend.fastapi_server:app --reload --port 8000
```

**Terminal 2 - React Frontend:**
```bash
cd demo/frontend
npm install
npm start
```

**Open browser to http://localhost:3000**

### Option 3: Docker (Production)

```bash
cd demo
docker-compose up --build
```

- **Streamlit**: http://localhost:8501
- **FastAPI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📖 Usage Guide

### Loading Replays

**Streamlit**: Use the sidebar dropdown to select from saved replays

**React**: Replays auto-load from the API. Use the dropdown in the controls panel.

**API**:
```bash
# List all replays
curl http://localhost:8000/replays

# Get specific replay
curl http://localhost:8000/replay/abc12345
```

### Running New Simulations

**Streamlit**: 
1. Click "🎲 New Simulation" in sidebar
2. Select scenario (1v1, 2v2, 3v3)
3. Click "▶️ Run Simulation"

**React**:
1. Select scenario from dropdown
2. Click "▶️ Run Simulation"

**API**:
```bash
curl -X POST http://localhost:8000/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "scenario": "3v3",
    "num_steps": 100,
    "seed": 42,
    "use_trained_model": false
  }'
```

### Using Trained Models

To use your trained PPO model instead of the random fallback:

```python
# In API request
{
  "scenario": "3v3",
  "num_steps": 100,
  "use_trained_model": true,
  "model_checkpoint": "../models/policy.pt"  # Path to your .pt file
}
```

The model checkpoint should contain `actor_state_dict` and `critic_state_dict`.

### Visualization Controls

| Control | Description |
|---------|-------------|
| **Play/Pause** | Start/stop automatic playback |
| **Step** | Advance one frame at a time |
| **Speed Slider** | Adjust playback speed (0.1x - 3x) |
| **Show Trails** | Display agent movement history (last 20 positions) |
| **Show Heatmap** | Overlay position density for selected team |
| **Show Pass Network** | Visualize passing connections with weighted arrows |

### Exporting Replays

**Streamlit**: Click "💾 Export Replay as GIF" in sidebar

**Programmatically**:
```python
from replay_schema import generate_example_replay, convert_numpy_types
import json

# Generate replay
replay = generate_example_replay(scenario="3v3", num_steps=100, seed=42)

# Save to file
with open('my_replay.json', 'w') as f:
    json.dump(convert_numpy_types(replay), f, indent=2)
```

---

## 📊 Replay JSON Schema

```json
{
  "metadata": {
    "replay_id": "abc12345",
    "timestamp": "2025-12-12T10:30:00",
    "scenario": "3v3",
    "num_agents": 6,
    "teams": [0, 1],
    "agent_names": ["team0_agent0", "team0_agent1", ...],
    "seed": 42,
    "total_steps": 87,
    "final_score": [2, 1],
    "winner": 0
  },
  "timesteps": [
    {
      "step": 0,
      "agents": [
        {
          "agent_id": "team0_agent0",
          "team": 0,
          "position": [3.5, 4.2],
          "action": 5,
          "action_name": "Pass ⚽",
          "reward": 1.0,
          "has_ball": true
        },
        ...
      ],
      "ball_position": [3.5, 4.2],
      "score": [0, 0],
      "episode_done": false,
      "reward_breakdown": {
        "team0_total": 2.5,
        "team1_total": -0.8,
        "team0_goal": 0,
        "team1_goal": 0
      }
    },
    ...
  ]
}
```

---

## 🧪 Testing

```bash
# Run unit tests
cd demo
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run specific test
python -m pytest tests/test_replay.py::TestReplaySchema::test_replay_writer_basic
```

### Smoke Test (No Model Required)

```bash
# Test replay generation
python replay_schema.py

# Test Streamlit loads
streamlit run streamlit_app.py --server.headless true &
sleep 5
curl http://localhost:8501/_stcore/health
kill %1

# Test FastAPI
uvicorn backend.fastapi_server:app --port 8000 &
sleep 3
curl http://localhost:8000/health
curl http://localhost:8000/replays
kill %1
```

---

## 🚢 Deployment

### Hugging Face Spaces (Streamlit)

1. Create new Space: https://huggingface.co/spaces
2. Select "Streamlit" as SDK
3. Upload files:
   ```
   streamlit_app.py
   replay_schema.py
   requirements.txt
   replays/
   ```
4. Space will auto-deploy at `https://huggingface.co/spaces/YOUR_USERNAME/SPACE_NAME`

### Render (FastAPI + React)

**Backend (FastAPI)**:
1. Create new Web Service: https://dashboard.render.com
2. Connect your GitHub repo
3. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn backend.fastapi_server:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Add `ENVIRONMENT=production`
4. Deploy

**Frontend (React)**:
1. Create new Static Site
2. Configure:
   - **Build Command**: `cd frontend && npm install && npm run build`
   - **Publish Directory**: `frontend/build`
   - **Environment**: Add `REACT_APP_API_URL=https://your-backend.onrender.com`
3. Deploy

### Vercel (React Only)

```bash
cd demo/frontend
npm install -g vercel
vercel --prod
```

Set environment variable: `REACT_APP_API_URL=https://your-api.com`

### Heroku (FastAPI)

```bash
cd demo

# Login and create app
heroku login
heroku create your-football-api

# Deploy
git add .
git commit -m "Deploy to Heroku"
git push heroku main

# Open
heroku open
```

### Docker Production

```bash
# Build
docker build -t football-rl-viz .

# Run
docker run -p 8000:8000 -e ENVIRONMENT=production football-rl-viz

# Or use docker-compose
docker-compose up -d
```

---

## 🔌 SportsHub Integration

### Embedding Streamlit as iframe

```html
<iframe 
  src="https://your-space.hf.space" 
  width="100%" 
  height="800px"
  frameborder="0"
></iframe>
```

### React Micro-Frontend

```javascript
// Import as a module
import FootballViz from './football-rl-viz';

// Use in your component
<FootballViz apiUrl="https://your-api.com" />
```

### API Integration Best Practices

```javascript
// Async simulation with caching
const runSimulation = async (scenario) => {
  const cacheKey = `sim_${scenario}_${Date.now()}`;
  
  try {
    const response = await fetch('https://api.example.com/simulate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': process.env.REACT_APP_API_KEY
      },
      body: JSON.stringify({
        scenario,
        num_steps: 100,
        use_trained_model: true,
        model_checkpoint: 'models/policy.pt'
      })
    });
    
    const data = await response.json();
    
    // Cache the replay
    localStorage.setItem(cacheKey, JSON.stringify(data.replay_data));
    
    return data;
  } catch (error) {
    console.error('Simulation failed:', error);
    // Fallback to cached data
    return JSON.parse(localStorage.getItem(cacheKey));
  }
};
```

**Recommended patterns**:
- Cache replays in browser storage to reduce API calls
- Use WebSockets for real-time updates during long simulations
- Implement rate limiting on the API (100 requests/hour)
- Add authentication headers for production

---

## 🔐 Security Notes

### Production API

Add API key authentication in `backend/fastapi_server.py`:

```python
@app.post("/simulate")
async def simulate(
    request: SimulationRequest,
    x_api_key: str = Header(...)
):
    if x_api_key != os.getenv("FOOTBALL_API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    # ... rest of endpoint
```

Set environment variable:
```bash
export FOOTBALL_API_KEY="your-secret-key-here"
```

In React, pass the key:
```javascript
fetch('/simulate', {
  headers: { 'X-API-Key': process.env.REACT_APP_API_KEY }
})
```

---

## 🎨 Customization

### Adding New Visualizations

**Streamlit** - Add to `streamlit_app.py`:
```python
def render_custom_overlay(fig, timestep_data):
    # Your custom visualization
    fig.add_trace(...)
    return fig
```

**React** - Add to `FieldCanvas.jsx`:
```javascript
// In useEffect, after drawing agents
if (showCustomOverlay) {
  ctx.fillStyle = 'rgba(255, 0, 0, 0.3)';
  // Your custom drawing code
}
```

### Custom Action Spaces

Update `ACTION_NAMES` in both `streamlit_app.py` and `FieldCanvas.jsx` to match your environment's action space.

---

## 🐛 Troubleshooting

**"No replay files found"**
→ Run `python replay_schema.py` to generate examples

**"Environment not available"**
→ The demo uses fallback simulation. To use real environment, ensure `env/football_env.py` exists

**FastAPI CORS errors**
→ Check `allow_origins` in `fastapi_server.py`. For local dev, it's set to `["*"]`

**React can't connect to API**
→ Ensure FastAPI is running on port 8000 and `proxy` in `package.json` points to it

**Model checkpoint not loading**
→ Verify the .pt file contains `actor_state_dict` and `critic_state_dict` keys

---

## 📜 License

MIT License - feel free to use in your projects!

---

## 🎓 Resume One-Liner

*"Built interactive multi-agent RL visualization platform with Streamlit and React, featuring real-time playback, heatmaps, pass networks, and REST API for model serving - deployed to Hugging Face Spaces and Render with Docker"*

---

## 🤝 Contributing

Pull requests welcome! Areas for improvement:
- Additional visualization overlays (shot heatmaps, passing lanes)
- Real-time training visualization
- Multi-replay comparison mode
- 3D visualization option
- Video export (MP4) in addition to GIF

---

**Built with ❤️ for the RL community**
