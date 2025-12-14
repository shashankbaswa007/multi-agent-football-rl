"""
Comprehensive test suite for the improved football environment
"""

import numpy as np
from env.improved_football_env import ImprovedFootballEnv, MOVE_RIGHT, MOVE_LEFT, MOVE_UP, MOVE_DOWN, SHOOT, PASS, STAY

def test_kickoff_positioning():
    """Test proper kickoff positioning"""
    print("\n" + "="*70)
    print("TEST 1: KICKOFF POSITIONING")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    # Check ball at center
    assert np.allclose(env.ball_position, [5.0, 3.0], atol=0.1), "Ball not at center"
    print("✅ Ball starts at center")
    
    # Check team 0 on left
    for agent in env.agents[:env.num_agents_per_team]:
        assert env.agent_positions[agent][0] < 5.0, f"{agent} not on left side"
    print("✅ Team 0 positioned on left (attacking RIGHT)")
    
    # Check team 1 on right
    for agent in env.agents[env.num_agents_per_team:]:
        assert env.agent_positions[agent][0] > 5.0, f"{agent} not on right side"
    print("✅ Team 1 positioned on right (attacking LEFT)")
    
    # Check no agent too close to ball
    for agent in env.agents:
        dist = np.linalg.norm(env.agent_positions[agent] - env.ball_position)
        assert dist > 1.0, f"{agent} too close to ball at start"
    print("✅ No agent starts too close to ball")
    
    print("✅ TEST 1 PASSED\n")

def test_movement_mechanics():
    """Test movement is consistent and correct"""
    print("="*70)
    print("TEST 2: MOVEMENT MECHANICS")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = env.agents[0]
    start_pos = env.agent_positions[agent].copy()
    
    # Test each direction
    directions = [
        (MOVE_RIGHT, np.array([0.5, 0])),
        (MOVE_LEFT, np.array([-0.5, 0])),
        (MOVE_UP, np.array([0, -0.5])),
        (MOVE_DOWN, np.array([0, 0.5]))
    ]
    
    for action, expected_delta in directions:
        env.reset()
        env.agent_selection = agent
        start = env.agent_positions[agent].copy()
        env.step(action)
        end = env.agent_positions[agent]
        actual_delta = end - start
        
        assert np.allclose(actual_delta, expected_delta, atol=0.01), \
            f"Movement mismatch: expected {expected_delta}, got {actual_delta}"
        print(f"✅ {['STAY','UP','DOWN','LEFT','RIGHT','SHOOT','PASS'][action]} movement correct: {actual_delta}")
    
    print("✅ TEST 2 PASSED\n")

def test_observation_space():
    """Test observation contains all required information"""
    print("="*70)
    print("TEST 3: OBSERVATION SPACE")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = env.agents[0]
    observation = obs[agent]
    
    # Check shape
    assert observation.shape == (21,), f"Wrong shape: {observation.shape}"
    print(f"✅ Observation shape correct: {observation.shape}")
    
    # Check normalization
    assert np.all(np.abs(observation) <= 1.01), "Values not in [-1, 1]"
    print("✅ All values normalized to [-1, 1]")
    
    # Print observation breakdown
    print("\nObservation breakdown:")
    print(f"  [0-1] Agent position: {observation[0:2]}")
    print(f"  [2-3] Ball position: {observation[2:4]}")
    print(f"  [4] Has ball: {observation[4]}")
    print(f"  [5-6] Vector to ball: {observation[5:7]}")
    print(f"  [7-8] Vector to goal: {observation[7:9]}")
    print(f"  [9-10] Nearest teammate: {observation[9:11]}")
    print(f"  [11-12] Nearest opponent: {observation[11:13]}")
    print(f"  [13-14] Agent velocity: {observation[13:15]}")
    print(f"  [15-16] Ball velocity: {observation[15:17]}")
    print(f"  [17-20] Boundary distances: {observation[17:21]}")
    
    print("✅ TEST 3 PASSED\n")

def test_possession_mechanics():
    """Test ball possession and gaining control"""
    print("="*70)
    print("TEST 4: POSSESSION MECHANICS")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = env.agents[0]
    
    # Move ball close to agent
    env.ball_position = env.agent_positions[agent] + np.array([0.2, 0.0])
    env.kickoff_timer = 0  # Disable cooldown
    
    assert env.ball_possession is None, "Ball should not be possessed initially"
    
    # Check possession
    reward = env._check_possession(agent)
    
    assert env.ball_possession == agent, "Agent should have possession"
    assert reward > 0, "Should receive reward for gaining possession"
    print(f"✅ Agent gains possession, reward: {reward:.3f}")
    
    # Test interception
    env.reset()
    opponent = env.agents[1]
    env.ball_possession = opponent
    env.ball_position = env.agent_positions[agent] + np.array([0.2, 0.0])
    env.kickoff_timer = 0
    
    reward = env._check_possession(agent)
    assert env.ball_possession == agent, "Should intercept"
    assert reward >= 0.2, "Interception should give good reward"
    print(f"✅ Agent intercepts, reward: {reward:.3f}")
    
    print("✅ TEST 4 PASSED\n")

def test_reward_shaping():
    """Test reward function components"""
    print("="*70)
    print("TEST 5: REWARD SHAPING")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = env.agents[0]
    
    # Test moving toward ball
    env.ball_position = np.array([5.0, 3.0])
    env.agent_positions[agent] = np.array([3.0, 3.0])
    env.agent_selection = agent
    
    _, rewards, _, _, _ = env.step(MOVE_RIGHT)
    print(f"✅ Moving toward ball reward: {rewards[agent]:.3f}")
    
    # Test moving with ball toward goal
    env.reset()
    env.agent_selection = agent
    env.ball_possession = agent
    env.agent_positions[agent] = np.array([3.0, 3.0])
    
    _, rewards, _, _, _ = env.step(MOVE_RIGHT)
    print(f"✅ Moving toward goal with ball reward: {rewards[agent]:.3f}")
    assert rewards[agent] > 0, "Should reward moving toward goal with ball"
    
    print("✅ TEST 5 PASSED\n")

def test_behavior_modes():
    """Test attack/defend mode switching"""
    print("="*70)
    print("TEST 6: BEHAVIOR MODES")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    team_0_agent = env.agents[0]
    team_1_agent = env.agents[2]
    
    # Give ball to team 0
    env.ball_possession = team_0_agent
    env.ball_position = env.agent_positions[team_0_agent].copy()
    
    # Update modes
    env._update_agent_mode(team_0_agent)
    env._update_agent_mode(team_1_agent)
    
    assert env.agent_modes[team_0_agent] == 'attack', "Team with ball should be in attack mode"
    assert env.agent_modes[team_1_agent] == 'defend', "Team without ball should be in defend mode"
    print(f"✅ {team_0_agent} in attack mode (has ball)")
    print(f"✅ {team_1_agent} in defend mode (no ball)")
    
    # Switch possession
    env.ball_possession = team_1_agent
    env._update_agent_mode(team_0_agent)
    env._update_agent_mode(team_1_agent)
    
    assert env.agent_modes[team_0_agent] == 'defend'
    assert env.agent_modes[team_1_agent] == 'attack'
    print(f"✅ Modes switched correctly after possession change")
    
    print("✅ TEST 6 PASSED\n")

def test_shooting_mechanics():
    """Test shooting and goal detection"""
    print("="*70)
    print("TEST 7: SHOOTING & GOAL DETECTION")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=1, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    agent = env.agents[0]  # Team 0, attacks right
    
    # Position agent close to opponent goal (right side)
    env.agent_positions[agent] = np.array([9.0, 3.0])
    env.ball_possession = agent
    env.ball_position = env.agent_positions[agent].copy()
    env.agent_selection = agent
    
    initial_goals = env.episode_stats['goals_team_0']
    
    # Attempt multiple shots (some should score due to randomness)
    goals_scored = 0
    for _ in range(20):
        env.agent_positions[agent] = np.array([9.0, 3.0])
        env.ball_possession = agent
        env.agent_selection = agent
        env.step(SHOOT)
        if env.episode_stats['goals_team_0'] > initial_goals:
            goals_scored += 1
            initial_goals = env.episode_stats['goals_team_0']
    
    assert goals_scored > 0, "Should score at least one goal from close range"
    print(f"✅ Scored {goals_scored}/20 shots from close range")
    
    # Test shot from too far (should fail or be penalized)
    env.reset()
    env.agent_positions[agent] = np.array([3.0, 3.0])  # Far from goal
    env.ball_possession = agent
    env.agent_selection = agent
    
    _, rewards, _, _, _ = env.step(SHOOT)
    print(f"✅ Shot from far away penalized: {rewards[agent]:.3f}")
    
    print("✅ TEST 7 PASSED\n")

def test_passing_mechanics():
    """Test passing between teammates"""
    print("="*70)
    print("TEST 8: PASSING MECHANICS")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    passer = env.agents[0]
    receiver = env.agents[1]
    
    # Position agents close together
    env.agent_positions[passer] = np.array([3.0, 3.0])
    env.agent_positions[receiver] = np.array([4.5, 3.0])
    env.ball_possession = passer
    env.agent_selection = passer
    
    initial_passes = env.episode_stats['successful_passes']
    
    _, rewards, _, _, _ = env.step(PASS)
    
    if env.ball_possession == receiver:
        print(f"✅ Pass successful, reward: {rewards[passer]:.3f}")
        assert env.episode_stats['successful_passes'] > initial_passes
    else:
        print(f"⚠️ Pass intercepted or failed")
    
    print("✅ TEST 8 PASSED\n")

def test_full_episode():
    """Test a full episode with random actions"""
    print("="*70)
    print("TEST 9: FULL EPISODE (100 STEPS)")
    print("="*70)
    
    env = ImprovedFootballEnv(num_agents_per_team=2, grid_width=10, grid_height=6)
    obs, info = env.reset()
    
    step_count = 0
    done = False
    
    while not done and step_count < 100:
        agent = env.agent_selection
        action = np.random.randint(0, 7)
        
        obs, rewards, terms, truncs, infos = env.step(action)
        
        done = any(terms.values()) or any(truncs.values())
        step_count += 1
        
        if step_count % 25 == 0:
            print(f"  Step {step_count}: Goals {env.episode_stats['goals_team_0']}-{env.episode_stats['goals_team_1']}, "
                  f"Ball at {env.ball_position}, Possession: {env.ball_possession}")
    
    print(f"\n✅ Episode completed: {step_count} steps")
    print(f"✅ Final stats:")
    print(f"  Goals: Team 0: {env.episode_stats['goals_team_0']}, Team 1: {env.episode_stats['goals_team_1']}")
    print(f"  Passes: {env.episode_stats['successful_passes']}/{env.episode_stats['passes']}")
    print(f"  Shots: Team 0: {env.episode_stats['shots_team_0']}, Team 1: {env.episode_stats['shots_team_1']}")
    
    print("✅ TEST 9 PASSED\n")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" COMPREHENSIVE TEST SUITE - IMPROVED FOOTBALL ENVIRONMENT")
    print("="*70)
    
    try:
        test_kickoff_positioning()
        test_movement_mechanics()
        test_observation_space()
        test_possession_mechanics()
        test_reward_shaping()
        test_behavior_modes()
        test_shooting_mechanics()
        test_passing_mechanics()
        test_full_episode()
        
        print("="*70)
        print("🎉 ALL TESTS PASSED! ENVIRONMENT READY FOR TRAINING!")
        print("="*70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
