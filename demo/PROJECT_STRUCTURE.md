# Multi-Agent RL Football Demo - Complete File Tree
# ==================================================

demo/
│
├── README.md                          # Comprehensive documentation (3500+ words)
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker build configuration
├── docker-compose.yml                 # Multi-container orchestration
├── Procfile                          # Heroku deployment config
├── run_demo.sh                       # Quick start script (executable)
│
├── replay_schema.py                   # Core replay system (350 lines)
│   ├── ReplayWriter class            # Records episodes to JSON
│   ├── ReplayReader class            # Loads and parses replays
│   ├── generate_example_replay()     # Fallback simulation
│   └── convert_numpy_types()         # JSON serialization helper
│
├── streamlit_app.py                   # Streamlit demo app (600+ lines)
│   ├── Main visualization UI
│   ├── Play/pause/step controls
│   ├── Heatmap computation & rendering
│   ├── Pass network analysis
│   ├── Agent statistics tracking
│   ├── GIF export functionality
│   └── Plotly-based field rendering
│
├── backend/
│   └── fastapi_server.py             # REST API server (450+ lines)
│       ├── POST /simulate            # Run new simulation
│       ├── GET /replay/{id}          # Fetch saved replay
│       ├── GET /replays              # List all replays
│       ├── DELETE /replay/{id}       # Remove replay
│       ├── Model loading & caching
│       └── Fallback random policy
│
├── frontend/                          # React application
│   ├── package.json                  # Node dependencies & scripts
│   ├── public/
│   │   └── index.html               # HTML entry point
│   └── src/
│       ├── index.js                 # React entry point
│       ├── App.js                   # Main app component (400+ lines)
│       │   ├── Replay loading
│       │   ├── Playback controls
│       │   ├── Visualization toggles
│       │   ├── API integration
│       │   ├── Heatmap computation
│       │   ├── Pass network analysis
│       │   └── Agent statistics
│       ├── App.css                  # Complete styling (300+ lines)
│       └── components/
│           └── FieldCanvas.jsx      # Canvas renderer (250+ lines)
│               ├── Field drawing
│               ├── Agent rendering
│               ├── Heatmap overlay
│               ├── Pass network arrows
│               └── Trail visualization
│
├── replays/                          # Saved replay JSON files
│   ├── example_1v1.json             # 1v1 demo (auto-generated)
│   ├── example_2v2.json             # 2v2 demo (auto-generated)
│   └── example_3v3.json             # 3v3 demo (auto-generated)
│
└── tests/
    └── test_replay.py                # Unit tests (180 lines)
        ├── TestReplaySchema          # Writer/reader tests
        │   ├── test_replay_writer_basic
        │   ├── test_replay_save_and_load
        │   ├── test_generate_example_replay
        │   └── test_convert_numpy_types
        └── TestReplayIntegration     # End-to-end tests
            └── test_full_replay_workflow


# Key Features by File
# ====================

## replay_schema.py
- Complete replay JSON format definition
- Dataclass-based structure (AgentState, Timestep, ReplayMetadata)
- Writer: Records episodes step-by-step
- Reader: Loads and iterates through timesteps
- Example generator: Creates realistic football behavior
- NumPy type conversion for JSON compatibility

## streamlit_app.py
- Interactive field visualization with Plotly
- Real-time playback with speed control (0.1x - 3x)
- Agent trails (last 20 positions per agent)
- Position heatmaps (20x20 grid, normalized)
- Pass network with weighted arrows
- Per-agent statistics panel
- Reward breakdown display
- GIF export using imageio
- Scenario selector and replay dropdown

## backend/fastapi_server.py
- RESTful API with OpenAPI docs
- POST /simulate: Generate new episodes
- GET /replays: List all saved replays
- GET /replay/{id}: Fetch specific replay
- Trained model loading with caching
- Fallback to random policy if model missing
- Background task for replay saving
- CORS middleware for React frontend
- API key authentication (production mode)

## frontend/src/App.js
- React hooks for state management
- Async API calls with error handling
- Auto-replay loading on mount
- Playback loop with configurable FPS
- Trail history tracking
- Heatmap computation (client-side)
- Pass network analysis
- Agent statistics aggregation
- Responsive layout with sidebar

## frontend/src/components/FieldCanvas.jsx
- HTML5 Canvas rendering at 60 FPS
- Field graphics (goals, center line, circle)
- Agent circles with team colors
- Ball rendering
- Heatmap overlay (translucent red)
- Pass network arrows (width = pass count)
- Agent trails (gradient transparency)
- Scoreboard display

## tests/test_replay.py
- Unit tests for ReplayWriter
- Unit tests for ReplayReader
- Example replay generation test
- NumPy type conversion test
- Full workflow integration test
- All tests passing ✓


# Lines of Code Summary
# ======================
Total: ~2,500+ lines of production code

Python Backend:
- replay_schema.py:      ~350 lines
- streamlit_app.py:      ~600 lines
- fastapi_server.py:     ~450 lines
- test_replay.py:        ~180 lines

JavaScript Frontend:
- App.js:                ~400 lines
- FieldCanvas.jsx:       ~250 lines
- App.css:               ~300 lines

Documentation:
- README.md:             ~450 lines
- This file:             ~180 lines


# Technology Stack
# =================

Backend:
- Python 3.11
- FastAPI (REST API)
- Uvicorn (ASGI server)
- Pydantic (data validation)
- PyTorch (model loading)
- NumPy (data processing)

Visualization:
- Streamlit (prototype)
- Plotly (interactive plots)
- Pillow + imageio (GIF export)

Frontend:
- React 18
- HTML5 Canvas
- Axios (HTTP client)
- ES6+ JavaScript

Testing:
- Pytest
- unittest

Deployment:
- Docker + docker-compose
- Heroku (Procfile)
- Hugging Face Spaces
- Render / Vercel


# Quick Commands Reference
# =========================

# Development
streamlit run streamlit_app.py
uvicorn backend.fastapi_server:app --reload --port 8000
cd frontend && npm start

# Testing
python -m pytest tests/ -v
python replay_schema.py  # Generate examples

# Docker
docker-compose up --build
docker build -t football-rl-viz .

# Deployment
git push heroku main
vercel --prod
