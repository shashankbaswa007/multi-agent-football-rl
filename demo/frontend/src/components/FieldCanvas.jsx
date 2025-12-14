/**
 * FieldCanvas.jsx - Canvas-based football field renderer
 * 
 * Renders the football field, agents, ball, and overlays using HTML5 Canvas
 * for smooth 60 FPS playback.
 */

import React, { useRef, useEffect } from 'react';

const FIELD_WIDTH = 12;
const FIELD_HEIGHT = 8;
const CELL_SIZE = 60;
const AGENT_RADIUS = 20;
const BALL_RADIUS = 10;

const TEAM_COLORS = {
  0: '#FF4444',  // Red
  1: '#4444FF'   // Blue
};

const FieldCanvas = ({ 
  timestepData, 
  showTrails = false, 
  trailHistory = {}, 
  showHeatmap = false,
  heatmapData = null,
  heatmapTeam = 0,
  showPassNetwork = false,
  passNetworkData = {}
}) => {
  const canvasRef = useRef(null);
  
  useEffect(() => {
    if (!timestepData || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;
    
    // Clear canvas
    ctx.clearRect(0, 0, width, height);
    
    // Draw field background
    ctx.fillStyle = '#90EE90';
    ctx.fillRect(0, 0, width, height - 50);
    
    // Draw field border
    ctx.strokeStyle = '#2F4F2F';
    ctx.lineWidth = 3;
    ctx.strokeRect(0, 0, width, height - 50);
    
    // Draw center line
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 2;
    ctx.setLineDash([10, 5]);
    ctx.beginPath();
    ctx.moveTo(width / 2, 0);
    ctx.lineTo(width / 2, height - 50);
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw center circle
    ctx.beginPath();
    ctx.arc(width / 2, (height - 50) / 2, 50, 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw goals
    const goalWidth = 20;
    const goalHeight = (height - 50) / 3;
    const goalY = ((height - 50) - goalHeight) / 2;
    
    // Left goal
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 3;
    ctx.strokeRect(-goalWidth, goalY, goalWidth, goalHeight);
    
    // Right goal
    ctx.strokeRect(width, goalY, goalWidth, goalHeight);
    
    // Draw heatmap overlay if requested
    if (showHeatmap && heatmapData && heatmapData[heatmapTeam]) {
      const heatmap = heatmapData[heatmapTeam];
      const gridSize = heatmap.length;
      const cellWidth = width / gridSize;
      const cellHeight = (height - 50) / gridSize;
      
      for (let i = 0; i < gridSize; i++) {
        for (let j = 0; j < gridSize; j++) {
          const intensity = heatmap[i][j];
          if (intensity > 0) {
            ctx.fillStyle = `rgba(255, 0, 0, ${intensity * 0.5})`;
            ctx.fillRect(j * cellWidth, i * cellHeight, cellWidth, cellHeight);
          }
        }
      }
    }
    
    // Draw trails if requested
    if (showTrails) {
      Object.entries(trailHistory).forEach(([agentId, positions]) => {
        if (positions.length < 2) return;
        
        ctx.strokeStyle = 'rgba(128, 128, 128, 0.4)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(positions[0][0] * CELL_SIZE, positions[0][1] * CELL_SIZE);
        
        for (let i = 1; i < positions.length; i++) {
          ctx.lineTo(positions[i][0] * CELL_SIZE, positions[i][1] * CELL_SIZE);
        }
        ctx.stroke();
      });
    }
    
    // Draw pass network if requested
    if (showPassNetwork && Object.keys(passNetworkData).length > 0) {
      const agentPositions = {};
      timestepData.agents.forEach(agent => {
        agentPositions[agent.agent_id] = agent.position;
      });
      
      Object.entries(passNetworkData).forEach(([key, count]) => {
        const [agentA, agentB] = key.split(',');
        
        if (agentPositions[agentA] && agentPositions[agentB]) {
          const posA = agentPositions[agentA];
          const posB = agentPositions[agentB];
          
          // Draw arrow
          ctx.strokeStyle = 'rgba(255, 255, 0, 0.6)';
          ctx.lineWidth = Math.min(5, 1 + count * 0.5);
          
          const x1 = posA[0] * CELL_SIZE;
          const y1 = posA[1] * CELL_SIZE;
          const x2 = posB[0] * CELL_SIZE;
          const y2 = posB[1] * CELL_SIZE;
          
          drawArrow(ctx, x1, y1, x2, y2);
        }
      });
    }
    
    // Draw agents
    timestepData.agents.forEach(agent => {
      const x = agent.position[0] * CELL_SIZE;
      const y = agent.position[1] * CELL_SIZE;
      const team = agent.team;
      const hasBall = agent.has_ball;
      
      // Agent circle
      ctx.fillStyle = TEAM_COLORS[team];
      ctx.beginPath();
      ctx.arc(x, y, AGENT_RADIUS, 0, Math.PI * 2);
      ctx.fill();
      
      // Border (yellow if has ball)
      ctx.strokeStyle = hasBall ? 'yellow' : 'white';
      ctx.lineWidth = hasBall ? 3 : 1;
      ctx.stroke();
      
      // Agent label
      ctx.fillStyle = 'white';
      ctx.font = 'bold 14px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(agent.agent_id.split('_').pop(), x, y);
    });
    
    // Draw ball
    const ballX = timestepData.ball_position[0] * CELL_SIZE;
    const ballY = timestepData.ball_position[1] * CELL_SIZE;
    
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(ballX, ballY, BALL_RADIUS, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw scoreboard
    ctx.fillStyle = 'white';
    ctx.fillRect(0, height - 50, width, 50);
    
    ctx.fillStyle = 'black';
    ctx.font = 'bold 24px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const scoreText = `Team 0: ${timestepData.score[0]}  |  Team 1: ${timestepData.score[1]}`;
    ctx.fillText(scoreText, width / 2, height - 25);
    
  }, [timestepData, showTrails, trailHistory, showHeatmap, heatmapData, heatmapTeam, showPassNetwork, passNetworkData]);
  
  return (
    <canvas
      ref={canvasRef}
      width={FIELD_WIDTH * CELL_SIZE}
      height={FIELD_HEIGHT * CELL_SIZE + 50}
      style={{
        border: '2px solid #333',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)'
      }}
    />
  );
};

// Helper function to draw arrow
function drawArrow(ctx, x1, y1, x2, y2) {
  const headlen = 10;
  const angle = Math.atan2(y2 - y1, x2 - x1);
  
  // Draw line
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  
  // Draw arrowhead
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - headlen * Math.cos(angle - Math.PI / 6),
    y2 - headlen * Math.sin(angle - Math.PI / 6)
  );
  ctx.lineTo(
    x2 - headlen * Math.cos(angle + Math.PI / 6),
    y2 - headlen * Math.sin(angle + Math.PI / 6)
  );
  ctx.closePath();
  ctx.fill();
}

export default FieldCanvas;
