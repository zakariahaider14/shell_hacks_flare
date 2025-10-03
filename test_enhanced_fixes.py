#!/usr/bin/env python3
"""
Test script to verify the enhanced EVCS system fixes
"""

import sys
import numpy as np
from enhanced_integrated_evcs_system import EnhancedIntegratedEVCSLLMRLSystem, MultiAgentRLEnvironment

def test_observation_space_fix():
    """Test that observation space is correctly sized"""
    print("🧪 Testing observation space fix...")
    
    try:
        # Create a mock federated manager
        class MockFederatedManager:
            def __init__(self):
                self.local_models = {1: MockPINNModel(), 2: MockPINNModel()}
                self.global_model = MockPINNModel()
        
        class MockPINNModel:
            def __init__(self):
                self.config = type('Config', (), {})()
        
        # Test MARL environment
        mock_manager = MockFederatedManager()
        env = MultiAgentRLEnvironment(mock_manager, num_systems=2)
        
        # Check observation space
        for i in range(2):
            agent_key = f'agent_{i}'
            obs_space = env.observation_space[agent_key]
            print(f"  Agent {i} observation space: {obs_space.shape}")
            
            if obs_space.shape[0] == 25:
                print(f"  ✅ Agent {i} has correct observation space (25,)")
            else:
                print(f"  ❌ Agent {i} has wrong observation space: {obs_space.shape}")
                return False
        
        # Test getting observations
        observations, _ = env.reset()
        for agent_key, obs in observations.items():
            print(f"  {agent_key} observation shape: {obs.shape}")
            if obs.shape[0] == 25:
                print(f"  ✅ {agent_key} observation has correct shape")
            else:
                print(f"  ❌ {agent_key} observation has wrong shape: {obs.shape}")
                return False
        
        print("✅ Observation space fix verified!")
        return True
        
    except Exception as e:
        print(f"❌ Observation space test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pinn_attack_simulation():
    """Test PINN attack simulation"""
    print("\n🧪 Testing PINN attack simulation...")
    
    try:
        # Create mock environment
        class MockFederatedManager:
            def __init__(self):
                self.local_models = {1: MockPINNModel()}
                self.anomaly_detectors = {1: MockAnomalyDetector()}
        
        class MockPINNModel:
            def __init__(self):
                self.config = type('Config', (), {})()
        
        class MockAnomalyDetector:
            def calculate_anomaly_score(self, result):
                return 0.3  # Low anomaly score
        
        mock_manager = MockFederatedManager()
        env = MultiAgentRLEnvironment(mock_manager, num_systems=1)
        
        # Test attack simulation
        attack_params = {
            'type': 'voltage_manipulation',
            'magnitude': 0.7,
            'duration': 30.0,
            'stealth_factor': 0.8,
            'target': 1
        }
        
        pinn_model = mock_manager.local_models[1]
        result = env._simulate_pinn_attack(pinn_model, attack_params)
        
        print(f"  Attack result: {result}")
        
        # Check required fields
        required_fields = ['success', 'impact', 'attack_type', 'magnitude', 'stealth_factor']
        for field in required_fields:
            if field in result:
                print(f"  ✅ {field}: {result[field]}")
            else:
                print(f"  ❌ Missing field: {field}")
                return False
        
        print("✅ PINN attack simulation verified!")
        return True
        
    except Exception as e:
        print(f"❌ PINN attack simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_system_initialization():
    """Test enhanced system initialization"""
    print("\n🧪 Testing enhanced system initialization...")
    
    try:
        config = {
            'hierarchical': {
                'use_enhanced_pinn': True,
                'use_dqn_sac_security': True,
                'total_duration': 60.0,
                'num_distribution_systems': 2  # Small test
            },
            'rl': {
                'num_systems': 2,
                'dqn_timesteps': 1000,  # Small for test
                'sac_timesteps': 1000,
                'coordination_training': True
            },
            'attack': {
                'max_episodes': 2,  # Small for test
                'coordination_type': 'simultaneous'
            }
        }
        
        # Initialize system
        system = EnhancedIntegratedEVCSLLMRLSystem(config)
        
        print(f"  ✅ System initialized successfully")
        print(f"  Config: {system.config['hierarchical']['num_distribution_systems']} systems")
        print(f"  Attack scenarios: {len(system.attack_scenarios)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced system initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced EVCS System Fixes")
    print("=" * 50)
    
    tests = [
        test_observation_space_fix,
        test_pinn_attack_simulation,
        test_enhanced_system_initialization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The fixes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
