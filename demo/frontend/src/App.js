/**
 * App.js - Main React application for Multi-Agent RL Football Visualization
 * 
 * Features:
 * - Load replays from API or use cached data
 * - Play/pause/step controls with speed slider
 * - Heatmap and pass network overlays
 * - Agent statistics and reward breakdown
 * - Scenario selector and simulation trigger
 */

import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import FieldCanvas from './components/FieldCanvas';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // Replay data
  const [replayData, setReplayData] = useState(null);
  const [replays, setReplays] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  
  // Playback controls
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1.0);
  
  // Visualization options
  const [showTrails, setShowTrails] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [heatmapTeam, setHeatmapTeam] = useState(0);
  const [showPassNetwork, setShowPassNetwork] = useState(false);
  
  // Computed data
  const [trailHistory, setTrailHistory] = useState({});
  const [heatmapData, setHeatmapData] = useState(null);
  const [passNetworkData, setPassNetworkData] = useState({});
  const [agentStats, setAgentStats] = useState({});
  
  // Simulation controls
  const [scenario, setScenario] = useState('3v3');
  const [loading, setLoading] = useState(false);
  
  // Load available replays on mount
  useEffect(() => {
    fetchReplays();
  }, []);
  
  // Fetch list of replays from API
  const fetchReplays = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/replays`);
      setReplays(response.data.replays);
      
      // Auto-load first replay if available
      if (response.data.replays.length > 0 && !replayData) {
        loadReplay(response.data.replays[0].replay_id);
      }
    } catch (error) {
      console.error('Error fetching replays:', error);
    }
  };
  
  // Load a specific replay
  const loadReplay = async (replayId) => {
    try {
      const response = await axios.get(`${API_BASE_URL}/replay/${replayId}`);
      setReplayData(response.data);
      setCurrentStep(0);
      setPlaying(false);
      setTrailHistory({});
      
      // Compute heatmap and pass network
      computeHeatmap(response.data);
      computePassNetwork(response.data);
      computeAgentStats(response.data);
    } catch (error) {
      console.error('Error loading replay:', error);
      alert('Failed to load replay');
    }
  };
  
  // Run new simulation
  const runSimulation = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/simulate`, {
        scenario: scenario,
        num_steps: 100,
        seed: Math.floor(Math.random() * 10000),
        use_trained_model: false
      });
      
      setReplayData(response.data.replay_data);
      setCurrentStep(0);
      setPlaying(false);
      setTrailHistory({});
      
      // Compute visualization data
      computeHeatmap(response.data.replay_data);
      computePassNetwork(response.data.replay_data);
      computeAgentStats(response.data.replay_data);
      
      // Refresh replay list
      fetchReplays();
      
      alert(`Simulation complete! Final score: ${response.data.final_score[0]} - ${response.data.final_score[1]}`);
    } catch (error) {
      console.error('Error running simulation:', error);
      alert('Failed to run simulation');
    } finally {
      setLoading(false);
    }
  };
  
  // Compute heatmap data
  const computeHeatmap = (data) => {
    const gridSize = 20;
    const heatmaps = {
      0: Array(gridSize).fill(0).map(() => Array(gridSize).fill(0)),
      1: Array(gridSize).fill(0).map(() => Array(gridSize).fill(0))
    };
    
    data.timesteps.forEach(ts => {
      ts.agents.forEach(agent => {
        const team = agent.team;
        const pos = agent.position;
        
        const xIdx = Math.floor((pos[0] / 12) * gridSize);
        const yIdx = Math.floor((pos[1] / 8) * gridSize);
        
        const xClamped = Math.max(0, Math.min(gridSize - 1, xIdx));
        const yClamped = Math.max(0, Math.min(gridSize - 1, yIdx));
        
        heatmaps[team][yClamped][xClamped] += 1;
      });
    });
    
    // Normalize
    [0, 1].forEach(team => {
      const maxVal = Math.max(...heatmaps[team].flat());
      if (maxVal > 0) {
        heatmaps[team] = heatmaps[team].map(row => 
          row.map(val => val / maxVal)
        );
      }
    });
    
    setHeatmapData(heatmaps);
  };
  
  // Compute pass network
  const computePassNetwork = (data) => {
    const passCounts = {};
    
    for (let i = 0; i < data.timesteps.length - 1; i++) {
      const ts = data.timesteps[i];
      
      ts.agents.forEach(agent => {
        if (agent.action_name === 'Pass ⚽') {
          const nextTs = data.timesteps[i + 1];
          const passerTeam = agent.team;
          const passerId = agent.agent_id;
          
          nextTs.agents.forEach(nextAgent => {
            if (nextAgent.team === passerTeam && nextAgent.has_ball && nextAgent.agent_id !== passerId) {
              const key = `${passerId},${nextAgent.agent_id}`;
              passCounts[key] = (passCounts[key] || 0) + 1;
            }
          });
        }
      });
    }
    
    setPassNetworkData(passCounts);
  };
  
  // Compute agent statistics
  const computeAgentStats = (data) => {
    const stats = {};
    
    data.timesteps.forEach(ts => {
      ts.agents.forEach(agent => {
        const id = agent.agent_id;
        
        if (!stats[id]) {
          stats[id] = {
            totalReward: 0,
            passCount: 0,
            shotCount: 0,
            possessionTime: 0
          };
        }
        
        stats[id].totalReward += agent.reward;
        if (agent.action_name === 'Pass ⚽') stats[id].passCount += 1;
        if (agent.action_name === 'Shoot 🎯') stats[id].shotCount += 1;
        if (agent.has_ball) stats[id].possessionTime += 1;
      });
    });
    
    setAgentStats(stats);
  };
  
  // Playback logic
  useEffect(() => {
    if (!playing || !replayData) return;
    
    const interval = setInterval(() => {
      setCurrentStep(step => {
        if (step >= replayData.timesteps.length - 1) {
          setPlaying(false);
          return step;
        }
        return step + 1;
      });
    }, 1000 / (10 * speed));
    
    return () => clearInterval(interval);
  }, [playing, speed, replayData]);
  
  // Update trail history
  useEffect(() => {
    if (!replayData || !showTrails) return;
    
    const timestep = replayData.timesteps[currentStep];
    const newTrails = { ...trailHistory };
    
    timestep.agents.forEach(agent => {
      if (!newTrails[agent.agent_id]) {
        newTrails[agent.agent_id] = [];
      }
      
      newTrails[agent.agent_id].push(agent.position);
      
      // Keep only last 20 positions
      if (newTrails[agent.agent_id].length > 20) {
        newTrails[agent.agent_id].shift();
      }
    });
    
    setTrailHistory(newTrails);
  }, [currentStep, showTrails, replayData]);
  
  if (!replayData) {
    return (
      <div className="App">
        <div className="loading">
          <h2>Loading replays...</h2>
          <p>Make sure the FastAPI server is running on port 8000</p>
        </div>
      </div>
    );
  }
  
  const metadata = replayData.metadata;
  const timestep = replayData.timesteps[currentStep];
  
  return (
    <div className="App">
      <header className="App-header">
        <h1>⚽ Multi-Agent RL Football</h1>
        <p>Watch AI agents learn to play football together</p>
      </header>
      
      <div className="container">
        {/* Sidebar */}
        <aside className="sidebar">
          <section className="controls-section">
            <h3>🎮 Controls</h3>
            
            <div className="replay-selector">
              <label>Replay:</label>
              <select 
                value={metadata.replay_id} 
                onChange={(e) => loadReplay(e.target.value)}
              >
                {replays.map(replay => (
                  <option key={replay.replay_id} value={replay.replay_id}>
                    {replay.scenario} - {replay.final_score[0]}:{replay.final_score[1]}
                  </option>
                ))}
              </select>
            </div>
            
            <div className="button-group">
              <button onClick={() => { setCurrentStep(0); setPlaying(false); setTrailHistory({}); }}>
                ⏮️ Reset
              </button>
              <button onClick={() => setPlaying(!playing)}>
                {playing ? '⏸️ Pause' : '▶️ Play'}
              </button>
              <button onClick={() => { setCurrentStep(Math.min(currentStep + 1, replayData.timesteps.length - 1)); setPlaying(false); }}>
                ⏭️ Step
              </button>
            </div>
            
            <div className="slider-control">
              <label>Speed: {speed.toFixed(1)}x</label>
              <input 
                type="range" 
                min="0.1" 
                max="3" 
                step="0.1" 
                value={speed}
                onChange={(e) => setSpeed(parseFloat(e.target.value))}
              />
            </div>
            
            <div className="slider-control">
              <label>Step: {currentStep} / {replayData.timesteps.length - 1}</label>
              <input 
                type="range" 
                min="0" 
                max={replayData.timesteps.length - 1} 
                value={currentStep}
                onChange={(e) => { setCurrentStep(parseInt(e.target.value)); setPlaying(false); }}
              />
            </div>
          </section>
          
          <section className="viz-section">
            <h3>🎨 Visualization</h3>
            
            <label>
              <input 
                type="checkbox" 
                checked={showTrails}
                onChange={(e) => setShowTrails(e.target.checked)}
              />
              Show Trails
            </label>
            
            <label>
              <input 
                type="checkbox" 
                checked={showHeatmap}
                onChange={(e) => setShowHeatmap(e.target.checked)}
              />
              Show Heatmap
            </label>
            
            {showHeatmap && (
              <div className="radio-group">
                <label>
                  <input 
                    type="radio" 
                    name="heatmap-team"
                    checked={heatmapTeam === 0}
                    onChange={() => setHeatmapTeam(0)}
                  />
                  Team 0
                </label>
                <label>
                  <input 
                    type="radio" 
                    name="heatmap-team"
                    checked={heatmapTeam === 1}
                    onChange={() => setHeatmapTeam(1)}
                  />
                  Team 1
                </label>
              </div>
            )}
            
            <label>
              <input 
                type="checkbox" 
                checked={showPassNetwork}
                onChange={(e) => setShowPassNetwork(e.target.checked)}
              />
              Show Pass Network
            </label>
          </section>
          
          <section className="sim-section">
            <h3>🎲 New Simulation</h3>
            
            <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
              <option value="1v1">1v1</option>
              <option value="2v2">2v2</option>
              <option value="3v3">3v3</option>
            </select>
            
            <button onClick={runSimulation} disabled={loading}>
              {loading ? '⏳ Running...' : '▶️ Run Simulation'}
            </button>
          </section>
          
          <section className="info-section">
            <h3>📊 Replay Info</h3>
            <p><strong>Scenario:</strong> {metadata.scenario}</p>
            <p><strong>Total Steps:</strong> {metadata.total_steps}</p>
            <p><strong>Final Score:</strong> {metadata.final_score[0]} - {metadata.final_score[1]}</p>
            <p><strong>Winner:</strong> {metadata.winner !== null ? `Team ${metadata.winner} 🏆` : 'Draw'}</p>
          </section>
        </aside>
        
        {/* Main content */}
        <main className="main-content">
          <FieldCanvas 
            timestepData={timestep}
            showTrails={showTrails}
            trailHistory={trailHistory}
            showHeatmap={showHeatmap}
            heatmapData={heatmapData}
            heatmapTeam={heatmapTeam}
            showPassNetwork={showPassNetwork}
            passNetworkData={passNetworkData}
          />
          
          <div className="info-panels">
            <div className="panel">
              <h3>👥 Agent States</h3>
              {timestep.agents.map(agent => (
                <div key={agent.agent_id} className="agent-info">
                  <span className={`team-badge team-${agent.team}`}>
                    {agent.team === 0 ? '🔴' : '🔵'} {agent.agent_id}
                  </span>
                  {agent.has_ball && <span className="ball-badge">⚽</span>}
                  <br />
                  <small>
                    Action: {agent.action_name} | Reward: {agent.reward.toFixed(3)}
                  </small>
                </div>
              ))}
            </div>
            
            <div className="panel">
              <h3>📈 Reward Breakdown</h3>
              {Object.entries(timestep.reward_breakdown || {}).map(([key, value]) => (
                <p key={key}>
                  <strong>{key}:</strong> {value.toFixed(2)}
                </p>
              ))}
            </div>
          </div>
          
          <div className="stats-table">
            <h3>📊 Agent Statistics</h3>
            <table>
              <thead>
                <tr>
                  <th>Agent</th>
                  <th>Total Reward</th>
                  <th>Passes</th>
                  <th>Shots</th>
                  <th>Possession</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(agentStats).map(([agentId, stats]) => (
                  <tr key={agentId}>
                    <td>{agentId}</td>
                    <td>{stats.totalReward.toFixed(2)}</td>
                    <td>{stats.passCount}</td>
                    <td>{stats.shotCount}</td>
                    <td>{stats.possessionTime}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
