# Multi-Agent RL Football Demo - System Architecture
# ==================================================

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACES                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────┐         ┌──────────────────────┐         │
│  │  STREAMLIT DEMO     │         │  REACT FRONTEND      │         │
│  │  (Prototype)        │         │  (Production)        │         │
│  │                     │         │                      │         │
│  │  • Play/Pause/Step  │         │  • Canvas Rendering  │         │
│  │  • Speed Slider     │         │  • API Integration   │         │
│  │  • Heatmaps         │         │  • Modern UI         │         │
│  │  • Pass Networks    │         │  • Responsive        │         │
│  │  • Agent Stats      │         │  • 60 FPS            │         │
│  │  • GIF Export       │         │                      │         │
│  │                     │         │                      │         │
│  │  Port: 8501         │         │  Port: 3000          │         │
│  └──────────┬──────────┘         └─────────┬────────────┘         │
│             │                               │                      │
└─────────────┼───────────────────────────────┼──────────────────────┘
              │                               │
              │                               │
              ▼                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND LAYER                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────┐         │
│  │           FASTAPI REST API SERVER                     │         │
│  │           (backend/fastapi_server.py)                 │         │
│  │                                                        │         │
│  │  Endpoints:                                            │         │
│  │  ┌──────────────────────────────────────────┐        │         │
│  │  │ POST /simulate                            │        │         │
│  │  │ ├─ Parse scenario, num_steps, seed        │        │         │
│  │  │ ├─ Load model or use fallback             │        │         │
│  │  │ └─ Return replay JSON                     │        │         │
│  │  └──────────────────────────────────────────┘        │         │
│  │                                                        │         │
│  │  ┌──────────────────────────────────────────┐        │         │
│  │  │ GET /replays                              │        │         │
│  │  │ └─ List all saved replays                 │        │         │
│  │  └──────────────────────────────────────────┘        │         │
│  │                                                        │         │
│  │  ┌──────────────────────────────────────────┐        │         │
│  │  │ GET /replay/{id}                          │        │         │
│  │  │ └─ Fetch specific replay                  │        │         │
│  │  └──────────────────────────────────────────┘        │         │
│  │                                                        │         │
│  │  Features:                                             │         │
│  │  • CORS middleware                                     │         │
│  │  • API key auth (production)                           │         │
│  │  • Model caching                                       │         │
│  │  • Background tasks                                    │         │
│  │  • OpenAPI docs                                        │         │
│  │                                                        │         │
│  │  Port: 8000                                            │         │
│  └────────────────────┬───────────────────────┬─────────┘         │
│                       │                       │                   │
└───────────────────────┼───────────────────────┼───────────────────┘
                        │                       │
                        ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────────────┐      ┌──────────────────────┐         │
│  │  REPLAY SYSTEM         │      │  MODEL CHECKPOINT     │         │
│  │  (replay_schema.py)    │      │  (models/policy.pt)   │         │
│  │                        │      │                       │         │
│  │  Classes:              │      │  PyTorch State Dict:  │         │
│  │  • ReplayWriter        │      │  • actor_state_dict   │         │
│  │  • ReplayReader        │      │  • critic_state_dict  │         │
│  │  • generate_example()  │      │  • optimizer_state    │         │
│  │                        │      │                       │         │
│  │  Functions:            │      │  Cached in Memory     │         │
│  │  • convert_numpy()     │      │  (MODEL_CACHE dict)   │         │
│  │  • write/read JSON     │      │                       │         │
│  └────────────────────────┘      └──────────────────────┘         │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────────────────────────────────┐             │
│  │         REPLAY FILES (replays/)                   │             │
│  │                                                    │             │
│  │  • example_1v1.json                               │             │
│  │  • example_2v2.json                               │             │
│  │  • example_3v3.json                               │             │
│  │  • user_episode_*.json                            │             │
│  │                                                    │             │
│  │  JSON Structure:                                  │             │
│  │  {                                                 │             │
│  │    "metadata": {...},                             │             │
│  │    "timesteps": [                                 │             │
│  │      {                                             │             │
│  │        "step": 0,                                 │             │
│  │        "agents": [...],                           │             │
│  │        "ball_position": [x, y],                   │             │
│  │        "score": [0, 0],                           │             │
│  │        "reward_breakdown": {...}                  │             │
│  │      }                                             │             │
│  │    ]                                               │             │
│  │  }                                                 │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Replay JSON → Reader → Timestep Data → Renderer → Visual Output   │
│                                                                     │
│  Streamlit Path:                                                    │
│  ┌─────────────────────────────────────────────────────┐          │
│  │ JSON → ReplayReader → Plotly Figure → Streamlit UI │          │
│  │                                                      │          │
│  │ Features:                                            │          │
│  │ • Plotly scatter/heatmap traces                     │          │
│  │ • Interactive hover tooltips                        │          │
│  │ • Auto-refresh on step change                       │          │
│  └─────────────────────────────────────────────────────┘          │
│                                                                     │
│  React Path:                                                        │
│  ┌─────────────────────────────────────────────────────┐          │
│  │ JSON → State → FieldCanvas → Canvas 2D Context     │          │
│  │                                                      │          │
│  │ Features:                                            │          │
│  │ • Direct Canvas API drawing                         │          │
│  │ • 60 FPS rendering                                  │          │
│  │ • Arrow helpers for pass network                    │          │
│  └─────────────────────────────────────────────────────┘          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Hugging Face     │  │ Render           │  │ Heroku          │  │
│  │ Spaces           │  │ (FastAPI)        │  │ (FastAPI)       │  │
│  │                  │  │                  │  │                 │  │
│  │ • Streamlit SDK  │  │ • Web Service    │  │ • Uses Procfile │  │
│  │ • Auto-deploy    │  │ • Build command  │  │ • git push      │  │
│  │ • Public URL     │  │ • Start command  │  │ • Auto-scale    │  │
│  │ • Free tier      │  │ • Env vars       │  │ • Add-ons       │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐                       │
│  │ Vercel           │  │ Docker           │                       │
│  │ (React)          │  │ (Everything)     │                       │
│  │                  │  │                  │                       │
│  │ • Static site    │  │ • Dockerfile     │                       │
│  │ • vercel --prod  │  │ • docker-compose │                       │
│  │ • CDN            │  │ • Multi-container│                       │
│  │ • Serverless     │  │ • Any platform   │                       │
│  └──────────────────┘  └──────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW EXAMPLE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. User Request:                                                   │
│     POST /simulate {"scenario": "3v3", "num_steps": 100}          │
│                                                                     │
│  2. Backend Processing:                                             │
│     ├─ Parse request (Pydantic validation)                         │
│     ├─ Check for trained model                                     │
│     │  ├─ If available: Load from MODEL_CACHE                      │
│     │  └─ If not: Use generate_example_replay()                    │
│     ├─ Run simulation:                                              │
│     │  ├─ Initialize ReplayWriter                                  │
│     │  ├─ For each timestep:                                       │
│     │  │  ├─ Get agent actions                                     │
│     │  │  ├─ Update environment                                    │
│     │  │  ├─ Collect rewards                                       │
│     │  │  └─ Write timestep                                        │
│     │  └─ Return replay_data                                       │
│     └─ Background: Save to replays/{id}.json                       │
│                                                                     │
│  3. Frontend Rendering:                                             │
│     ├─ Receive replay_data                                          │
│     ├─ Parse metadata & timesteps                                   │
│     ├─ Compute heatmap (20x20 grid)                                │
│     ├─ Compute pass network (adjacency matrix)                     │
│     ├─ Start playback loop:                                         │
│     │  ├─ Get timestep[currentStep]                                │
│     │  ├─ Render field, agents, ball                               │
│     │  ├─ Update UI panels                                         │
│     │  └─ Increment step (if playing)                              │
│     └─ User controls:                                               │
│        ├─ Play/Pause toggles playback                              │
│        ├─ Slider updates currentStep                               │
│        └─ Checkboxes toggle overlays                               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                      TESTING STRATEGY                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Unit Tests (tests/test_replay.py):                                │
│  ┌──────────────────────────────────────────────────┐             │
│  │ • test_replay_writer_basic()                      │             │
│  │   └─ Verify timestep recording                   │             │
│  │                                                    │             │
│  │ • test_replay_save_and_load()                     │             │
│  │   └─ Write to temp file, read back               │             │
│  │                                                    │             │
│  │ • test_generate_example_replay()                  │             │
│  │   └─ Check structure, metadata, agents           │             │
│  │                                                    │             │
│  │ • test_convert_numpy_types()                      │             │
│  │   └─ Verify JSON serialization                   │             │
│  │                                                    │             │
│  │ • test_full_replay_workflow()                     │             │
│  │   └─ End-to-end generation → save → load         │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                     │
│  Integration Tests (test_system.sh):                               │
│  ┌──────────────────────────────────────────────────┐             │
│  │ ✓ Replay generation                               │             │
│  │ ✓ Unit tests passing                              │             │
│  │ ✓ Python imports                                  │             │
│  │ ✓ Replay file integrity                           │             │
│  │ ✓ FastAPI server health                           │             │
│  │ ✓ React files present                             │             │
│  │ ✓ Documentation complete                          │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                     │
│  Smoke Tests:                                                       │
│  ┌──────────────────────────────────────────────────┐             │
│  │ • streamlit run streamlit_app.py (manual check)  │             │
│  │ • uvicorn backend.fastapi_server:app (API docs)  │             │
│  │ • curl http://localhost:8000/health               │             │
│  │ • cd frontend && npm start (visual check)         │             │
│  └──────────────────────────────────────────────────┘             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

```

## File Size Breakdown

```
Component               Lines    Purpose
─────────────────────────────────────────────────────────────
replay_schema.py         350     Replay system core
streamlit_app.py         600     Streamlit demo UI
fastapi_server.py        450     REST API backend
App.js                   400     React main component
FieldCanvas.jsx          250     Canvas renderer
App.css                  300     React styling
test_replay.py           180     Unit tests
README.md                450     Documentation
─────────────────────────────────────────────────────────────
TOTAL                   2,980    Production code
```

## Technology Stack

```
Backend                  Frontend                Deployment
────────────────────────────────────────────────────────────
Python 3.11              React 18               Docker
FastAPI                  HTML5 Canvas           docker-compose
Uvicorn (ASGI)           Axios                  Heroku (Procfile)
Pydantic                 ES6+                   Render
PyTorch                  CSS3                   Vercel
NumPy                                           HF Spaces
Streamlit
Plotly
imageio
```

## Performance Characteristics

```
Component          Performance       Notes
──────────────────────────────────────────────────────────
Streamlit          20-30 FPS        Plotly redraws
React Canvas       60 FPS           Direct Canvas API
API /simulate      ~100ms           Fallback mode
API /simulate      ~500ms           With trained model
Replay size        ~1 KB/timestep   JSON format
Heatmap compute    ~10ms            20x20 grid
Pass network       ~5ms             Adjacency matrix
```

## Security Model

```
Production Mode (ENVIRONMENT=production):
├─ API Key required (X-API-Key header)
├─ CORS origins restricted
├─ Rate limiting (future)
└─ HTTPS only (deployment platforms)

Development Mode:
├─ No API key
├─ CORS allow all origins
└─ HTTP allowed
```
