#!/bin/bash
# Demo Quick Start Script
# =======================
# Runs the Streamlit demo with automatic setup

set -e

echo "🚀 Multi-Agent RL Football Demo - Quick Start"
echo "=============================================="
echo ""

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found at ../.venv"
    echo "Please create it first:"
    echo "  cd .. && python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

# Activate virtual environment
echo "✓ Activating virtual environment..."
source ../.venv/bin/activate

# Check if dependencies are installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -q -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Generate example replays if needed
if [ ! -f "replays/example_3v3.json" ]; then
    echo "📊 Generating example replays..."
    python replay_schema.py
    echo "✓ Example replays generated"
else
    echo "✓ Example replays found"
fi

# Run Streamlit
echo ""
echo "🎉 Starting Streamlit demo..."
echo "   Open your browser to: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run streamlit_app.py
