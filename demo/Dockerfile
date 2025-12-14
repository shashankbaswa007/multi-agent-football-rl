# Dockerfile for Multi-Agent RL Football Demo
# ============================================
# Multi-stage build for efficient production deployment

FROM python:3.11-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY replay_schema.py .
COPY backend/ backend/
COPY replays/ replays/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "backend.fastapi_server:app", "--host", "0.0.0.0", "--port", "8000"]
