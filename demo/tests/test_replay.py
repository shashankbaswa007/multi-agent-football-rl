"""
Unit Tests for Replay Writer and Reader
========================================

Tests the replay JSON generation and parsing functionality.
"""

import unittest
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from replay_schema import (
    ReplayWriter, ReplayReader, generate_example_replay,
    convert_numpy_types
)


class TestReplaySchema(unittest.TestCase):
    """Test replay schema functionality."""
    
    def test_replay_writer_basic(self):
        """Test basic replay writing."""
        writer = ReplayWriter(scenario="3v3", seed=42)
        
        # Add a timestep
        agents_data = [
            {
                'agent_id': 'team0_agent0',
                'team': 0,
                'position': [5.0, 4.0],
                'action': 1,
                'action_name': 'Up',
                'reward': 0.1,
                'has_ball': True
            }
        ]
        
        writer.add_timestep(
            step=0,
            agents_data=agents_data,
            ball_position=[5.0, 4.0],
            score=[0, 0],
            episode_done=False,
            reward_breakdown={'team0_total': 0.1, 'team1_total': 0.0}
        )
        
        self.assertEqual(len(writer.timesteps), 1)
        self.assertEqual(writer.scenario, "3v3")
        self.assertEqual(writer.seed, 42)
    
    def test_replay_save_and_load(self):
        """Test saving and loading replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_replay.json"
            
            # Create and save replay
            writer = ReplayWriter(scenario="2v2", seed=123)
            
            for step in range(5):
                agents_data = [
                    {
                        'agent_id': f'agent{i}',
                        'team': i % 2,
                        'position': [float(i), float(step)],
                        'action': step % 7,
                        'action_name': 'Hold',
                        'reward': 0.05,
                        'has_ball': i == 0
                    }
                    for i in range(4)
                ]
                
                writer.add_timestep(
                    step=step,
                    agents_data=agents_data,
                    ball_position=[5.0, 4.0],
                    score=[0, 0],
                    episode_done=(step == 4),
                    reward_breakdown={'team0_total': 0.1, 'team1_total': 0.1}
                )
            
            writer.save(str(filepath))
            
            # Load and verify
            reader = ReplayReader(str(filepath))
            
            self.assertEqual(reader.get_num_timesteps(), 5)
            
            metadata = reader.get_metadata()
            self.assertEqual(metadata['scenario'], "2v2")
            self.assertEqual(metadata['seed'], 123)
            self.assertEqual(metadata['total_steps'], 5)
            
            # Check first timestep
            ts0 = reader.get_timestep(0)
            self.assertEqual(ts0['step'], 0)
            self.assertEqual(len(ts0['agents']), 4)
    
    def test_generate_example_replay(self):
        """Test example replay generation."""
        replay = generate_example_replay(scenario="3v3", num_steps=20, seed=42)
        
        self.assertIn('metadata', replay)
        self.assertIn('timesteps', replay)
        
        metadata = replay['metadata']
        self.assertEqual(metadata['scenario'], "3v3")
        self.assertEqual(metadata['num_agents'], 6)
        self.assertLessEqual(metadata['total_steps'], 20)
        
        # Check timesteps
        timesteps = replay['timesteps']
        self.assertGreater(len(timesteps), 0)
        
        ts = timesteps[0]
        self.assertIn('step', ts)
        self.assertIn('agents', ts)
        self.assertIn('ball_position', ts)
        self.assertIn('score', ts)
        
        # Verify all agents present
        self.assertEqual(len(ts['agents']), 6)
    
    def test_convert_numpy_types(self):
        """Test numpy type conversion."""
        import numpy as np
        
        data = {
            'int': np.int64(42),
            'float': np.float64(3.14),
            'bool': np.bool_(True),
            'array': np.array([1, 2, 3]),
            'nested': {
                'value': np.int32(10)
            }
        }
        
        converted = convert_numpy_types(data)
        
        # Verify all types are JSON-serializable
        json_str = json.dumps(converted)
        self.assertIsInstance(json_str, str)
        
        # Verify values
        parsed = json.loads(json_str)
        self.assertEqual(parsed['int'], 42)
        self.assertAlmostEqual(parsed['float'], 3.14, places=2)
        self.assertEqual(parsed['bool'], True)
        self.assertEqual(parsed['array'], [1, 2, 3])
        self.assertEqual(parsed['nested']['value'], 10)


class TestReplayIntegration(unittest.TestCase):
    """Integration tests for replay system."""
    
    def test_full_replay_workflow(self):
        """Test complete workflow from generation to visualization."""
        # Generate replay
        replay = generate_example_replay(scenario="1v1", num_steps=30, seed=99)
        replay = convert_numpy_types(replay)  # Convert numpy types for JSON
        
        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "workflow_test.json"
            
            with open(filepath, 'w') as f:
                json.dump(replay, f)
            
            # Load with reader
            reader = ReplayReader(str(filepath))
            
            # Iterate through timesteps
            count = 0
            for ts in reader.iter_timesteps():
                count += 1
                # Verify structure
                self.assertIn('step', ts)
                self.assertIn('agents', ts)
                self.assertIn('ball_position', ts)
                
                # Verify agents have required fields
                for agent in ts['agents']:
                    self.assertIn('agent_id', agent)
                    self.assertIn('position', agent)
                    self.assertIn('action', agent)
                    self.assertIn('reward', agent)
            
            self.assertEqual(count, reader.get_num_timesteps())


if __name__ == '__main__':
    unittest.main()
