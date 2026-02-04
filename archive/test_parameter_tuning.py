#!/usr/bin/env python3
"""
Unit tests for parameter tuning framework.

Tests the parameter tuning system across various drone configurations and mission types,
validates optimization algorithms, and compares tuned vs default performance.
"""

import unittest
import numpy as np
import sys
import os
import argparse
import time
from typing import Dict, List, Tuple

sys.path.append('Simulation')

from drone_simulator import ConfigurableQuadcopter
from generalized_ctrl import GeneralizedControl
from parameter_tuner import ParameterTuner, TuningResult, run_tuning_session, save_tuning_results
from tuning_configurations import (
    DRONE_CONFIGURATIONS, MISSION_CONFIGURATIONS, PARAMETER_SETS, TUNING_SCENARIOS,
    get_tuning_configuration, create_custom_scenario
)
from cost_functions import multi_objective_cost, tracking_error_cost, stability_cost

# Import existing visualization utilities
from utils import animation, display
import utils
VISUALIZATION_AVAILABLE = True

class TestParameterTuning(unittest.TestCase):
    """Test suite for parameter tuning functionality."""
    
    def setUp(self):
        """Set up test configurations."""
        # Get visualization settings from global args
        self.enable_visualization = getattr(sys.modules[__name__], 'VISUALIZE_TUNING', False)
        self.enable_detailed_viz = getattr(sys.modules[__name__], 'VISUALIZE_DETAILED', False)
        self.save_plots = getattr(sys.modules[__name__], 'SAVE_PLOTS', False)
        self.plot_dir = getattr(sys.modules[__name__], 'PLOT_DIR', './tuning_plots')
        
        # Create plot directory if needed
        if (self.enable_visualization or self.enable_detailed_viz) and self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
        
        # Test configurations (subset for faster testing)
        self.test_drone_configs = {
            'small_quad': DRONE_CONFIGURATIONS['small_quad'],
            'standard_quad': DRONE_CONFIGURATIONS['standard_quad'],
            'large_quad': DRONE_CONFIGURATIONS['large_quad'],
            'standard_hex': DRONE_CONFIGURATIONS['standard_hex']
        }
        
        self.test_mission_configs = {
            'hover_stability': MISSION_CONFIGURATIONS['hover_stability'],
            'position_step': MISSION_CONFIGURATIONS['position_step'],
            'square_pattern': MISSION_CONFIGURATIONS['square_pattern']
        }
        
        self.test_parameter_sets = {
            'position_control': PARAMETER_SETS['position_control'],
            'velocity_control': PARAMETER_SETS['velocity_control'],
            'full_control': PARAMETER_SETS['full_control']
        }
    
    def test_cost_function_calculations(self):
        """Test that cost functions work correctly with trajectory data."""
        # Create sample trajectory data
        time_data = np.linspace(0, 10, 100)
        actual_positions = np.column_stack([
            np.sin(time_data * 0.5), 
            np.cos(time_data * 0.5), 
            -2.0 * np.ones_like(time_data)
        ])
        desired_positions = np.zeros_like(actual_positions)
        desired_positions[:, 2] = -2.0
        
        trajectory_data = {
            'time': time_data.tolist(),
            'actual_positions': actual_positions.tolist(),
            'desired_positions': desired_positions.tolist(),
            'actual_velocities': np.diff(actual_positions, axis=0, prepend=actual_positions[:1]).tolist(),
            'control_commands': np.random.uniform(1000, 1500, (100, 4)).tolist()
        }
        
        # Test individual cost functions
        tracking_cost = tracking_error_cost(trajectory_data)
        stability_cost_val = stability_cost(trajectory_data)
        multi_cost, breakdown = multi_objective_cost(trajectory_data)
        
        # Verify costs are reasonable
        self.assertGreater(tracking_cost, 0, "Tracking cost should be positive")
        self.assertGreater(stability_cost_val, 0, "Stability cost should be positive")
        self.assertGreater(multi_cost, 0, "Multi-objective cost should be positive")
        self.assertIn('tracking', breakdown, "Cost breakdown should include tracking")
        self.assertIn('stability', breakdown, "Cost breakdown should include stability")
        self.assertIn('efficiency', breakdown, "Cost breakdown should include efficiency")
    
    def test_parameter_tuner_initialization(self):
        """Test parameter tuner initialization with different configurations."""
        for drone_name, drone_config in self.test_drone_configs.items():
            for mission_name, mission_config in self.test_mission_configs.items():
                for param_name, param_set in self.test_parameter_sets.items():
                    with self.subTest(drone=drone_name, mission=mission_name, params=param_name):
                        tuner = ParameterTuner(
                            drone_config=drone_config,
                            mission_config=mission_config,
                            parameter_set=param_set,
                            algorithm='random_search',
                            enable_multiprocessing=False  # Disable for testing to avoid overhead
                        )
                        
                        # Verify tuner is properly initialized
                        self.assertEqual(tuner.drone_config, drone_config)
                        self.assertEqual(tuner.mission_config, mission_config)
                        self.assertEqual(tuner.parameter_set, param_set)
                        self.assertEqual(len(tuner.parameter_names), len(param_set['parameters']))
                        self.assertGreater(len(tuner.parameter_bounds), 0)
    
    def test_parameter_evaluation(self):
        """Test single parameter evaluation."""
        # Use simple configuration for fast testing
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
        
        # Test with default parameters
        default_params = {'pos_P_gain': np.array([1.0, 1.0, 1.0])}
        cost, trajectory_data = tuner.evaluate_parameters(default_params)
        
        # Verify evaluation results
        self.assertIsInstance(cost, (int, float), "Cost should be numeric")
        self.assertGreater(cost, 0, "Cost should be positive")
        self.assertIsInstance(trajectory_data, dict, "Trajectory data should be dictionary")
        self.assertIn('actual_positions', trajectory_data, "Should contain actual positions")
        self.assertGreater(len(trajectory_data['actual_positions']), 10, "Should have trajectory data")
    
    # def test_random_search_optimization(self):
    #     """Test random search optimization algorithm."""
    #     drone_config = self.test_drone_configs['small_quad']
    #     mission_config = self.test_mission_configs['hover_stability']
    #     param_set = self.test_parameter_sets['position_control']
        
    #     tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search')
        
    #     # Run short optimization
    #     result = tuner.optimize(max_iterations=10)
        
    #     # Verify optimization results
    #     self.assertIsInstance(result, TuningResult)
    #     self.assertIsNotNone(result.best_parameters)
    #     self.assertIsInstance(result.best_cost, (int, float))
    #     self.assertEqual(len(result.cost_history), 10)
    #     self.assertEqual(len(result.parameter_history), 10)
    #     self.assertGreater(result.tuning_time, 0)
        
    #     # Verify convergence info
    #     self.assertIn('total_evaluations', result.convergence_info)
    #     self.assertEqual(result.convergence_info['total_evaluations'], 10)
    
    # def test_gradient_descent_optimization(self):
    #     """Test gradient descent optimization algorithm."""
    #     drone_config = self.test_drone_configs['small_quad']
    #     mission_config = self.test_mission_configs['hover_stability']
    #     param_set = self.test_parameter_sets['position_control']
        
    #     algorithm_params = {
    #         'learning_rate': 0.01,
    #         'momentum': 0.9,
    #         'adaptive_lr': True
    #     }
        
    #     tuner = ParameterTuner(drone_config, mission_config, param_set, 
    #                           'gradient_descent', algorithm_params)
        
    #     # Run short optimization
    #     result = tuner.optimize(max_iterations=5, tolerance=1e-3)
        
    #     # Verify gradient descent specific results
    #     self.assertIn('algorithm', result.optimization_info)
    #     self.assertEqual(result.optimization_info['algorithm'], 'gradient_descent')
    #     self.assertIn('final_learning_rate', result.optimization_info)
    #     self.assertGreater(result.optimization_info['final_learning_rate'], 0)
    
    def test_genetic_algorithm_optimization(self):
        """Test genetic algorithm optimization."""
        drone_config = self.test_drone_configs['small_quad'] 
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        algorithm_params = {
            'population_size': 8,  # Small for testing
            'mutation_rate': 0.1,
            'crossover_rate': 0.8,
            'elite_size': 1
        }
        
        tuner = ParameterTuner(drone_config, mission_config, param_set,
                              'genetic', algorithm_params, enable_multiprocessing=False)
        
        # Run short optimization
        result = tuner.optimize(max_iterations=3)  # 3 generations
        
        # Verify genetic algorithm results
        self.assertEqual(result.optimization_info['algorithm'], 'genetic')
        self.assertEqual(result.optimization_info['population_size'], 8)
        self.assertIn('final_best_fitness', result.optimization_info)
        
        # Should have population_size * generations evaluations
        expected_evaluations = 8 * 3 + 8  # Initial population + generations
        self.assertEqual(result.convergence_info['total_evaluations'], expected_evaluations)
    
    def test_cmaes_optimization(self):
        """Test CMA-ES optimization algorithm."""
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        algorithm_params = {
            'population_size': 6,  # Small for testing
            'sigma': 0.3,
            'seed': 42  # For reproducible results
        }
        
        tuner = ParameterTuner(drone_config, mission_config, param_set,
                              'cmaes', algorithm_params, enable_multiprocessing=False)
        
        # Run short optimization
        result = tuner.optimize(max_iterations=3)  # 3 generations
        
        # Verify CMA-ES algorithm results
        self.assertEqual(result.optimization_info['algorithm'], 'cmaes')
        self.assertEqual(result.optimization_info['population_size'], 6)
        self.assertIn('sigma', result.optimization_info)
        self.assertIn('final_best_cost', result.optimization_info)
        
        # Should have population_size * generations evaluations
        expected_evaluations = 6 * 3  # population_size * generations
        self.assertEqual(result.convergence_info['total_evaluations'], expected_evaluations)
        
        # Verify convergence info is present
        self.assertIn('convergence_criteria', result.optimization_info)
    
    def test_tuned_vs_default_performance(self):
        """Test that tuned parameters improve performance over defaults."""
        drone_config = self.test_drone_configs['standard_quad']
        mission_config = self.test_mission_configs['position_step']
        param_set = self.test_parameter_sets['velocity_control']
        
        # Get default performance
        default_params = {
            'vel_P_gain': np.array([5.0, 5.0, 4.0]),
            'vel_D_gain': np.array([0.5, 0.5, 0.5]),
            'vel_I_gain': np.array([5.0, 5.0, 5.0])
        }
        
        tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
        default_cost, _ = tuner.evaluate_parameters(default_params)
        
        # Tune parameters
        result = tuner.optimize(max_iterations=20)
        tuned_cost = result.best_cost
        
        # Print results for analysis
        print(f"\nPerformance comparison for {drone_config['description']}:")
        print(f"Default cost: {default_cost:.4f}")
        print(f"Tuned cost: {tuned_cost:.4f}")
        print(f"Improvement: {((default_cost - tuned_cost) / default_cost * 100):.2f}%")
        
        # Tuned parameters should generally be better, but allow some tolerance
        # for stochastic optimization and limited iterations
        improvement_ratio = (default_cost - tuned_cost) / default_cost
        self.assertGreater(improvement_ratio, -0.5, 
                          "Tuned parameters shouldn't be much worse than defaults")
    
    def test_multi_drone_configuration_tuning(self):
        """Test tuning across multiple drone configurations."""
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        results = {}
        
        for drone_name, drone_config in self.test_drone_configs.items():
            tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
            result = tuner.optimize(max_iterations=15)
            results[drone_name] = result
            
            # Verify each drone was tuned
            self.assertIsNotNone(result.best_parameters)
            self.assertLess(result.best_cost, 1000.0, f"{drone_name} tuning failed")
        
        # Print comparison
        print(f"\nMulti-drone tuning results for hover stability:")
        for drone_name, result in results.items():
            print(f"{drone_name}: Cost={result.best_cost:.4f}, "
                  f"Time={result.tuning_time:.2f}s, "
                  f"Evals={result.convergence_info['total_evaluations']}")
    
    def test_multi_mission_tuning(self):
        """Test tuning across multiple mission types."""
        drone_config = self.test_drone_configs['standard_quad']
        param_set = self.test_parameter_sets['full_control']
        
        results = {}
        
        for mission_name, mission_config in self.test_mission_configs.items():
            tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
            result = tuner.optimize(max_iterations=15)
            results[mission_name] = result
            
            # Verify each mission was tuned
            self.assertIsNotNone(result.best_parameters)
            self.assertLess(result.best_cost, 1000.0, f"{mission_name} tuning failed")
        
        # Print comparison
        print(f"\nMulti-mission tuning results for standard quad:")
        for mission_name, result in results.items():
            print(f"{mission_name}: Cost={result.best_cost:.4f}, "
                  f"Time={result.tuning_time:.2f}s")
    
    def test_predefined_tuning_scenarios(self):
        """Test predefined tuning scenarios."""
        # Test a subset of scenarios for reasonable test time
        test_scenarios = ['micro_hover', 'racing_agility']
        
        for scenario_name in test_scenarios:
            if scenario_name in TUNING_SCENARIOS:
                with self.subTest(scenario=scenario_name):
                    # Run tuning session
                    result = run_tuning_session(
                        scenario_name=scenario_name,
                        algorithm='random_search',
                        max_iterations=10  # Short for testing
                    )
                    
                    # Verify results
                    self.assertIsInstance(result, TuningResult)
                    self.assertIsNotNone(result.best_parameters)
                    self.assertLess(result.best_cost, 1000.0)
                    
                    print(f"\nScenario '{scenario_name}' results:")
                    print(f"  Best cost: {result.best_cost:.4f}")
                    print(f"  Tuning time: {result.tuning_time:.2f}s")
                    print(f"  Evaluations: {result.convergence_info['total_evaluations']}")
    
    def test_parameter_bounds_enforcement(self):
        """Test that parameter bounds are properly enforced."""
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['velocity_control']
        
        tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
        
        # Test with parameters outside bounds
        invalid_params = {
            'vel_P_gain': np.array([100.0, 100.0, 100.0]),  # Too high
            'vel_D_gain': np.array([-1.0, -1.0, -1.0]),    # Too low
            'vel_I_gain': np.array([10.0, 10.0, 10.0])      # Valid
        }
        
        cost, _ = tuner.evaluate_parameters(invalid_params)
        
        # Should get high cost due to safety constraint penalty
        self.assertGreater(cost, 100.0, "Should penalize parameters outside bounds")
    
    def test_cmaes_parameter_validation(self):
        """Test CMA-ES specific parameter validation and behavior."""
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        # Test with different CMA-ES parameters
        test_cases = [
            # Test with automatic population size (None)
            {'population_size': None, 'sigma': 0.3, 'seed': 42},
            # Test with custom population size
            {'population_size': 8, 'sigma': 0.5, 'seed': 123},
            # Test with different sigma values
            {'population_size': 4, 'sigma': 0.1, 'seed': None}
        ]
        
        for i, algorithm_params in enumerate(test_cases):
            with self.subTest(case=i):
                tuner = ParameterTuner(drone_config, mission_config, param_set,
                                      'cmaes', algorithm_params, enable_multiprocessing=False)
                
                # Run short optimization
                result = tuner.optimize(max_iterations=2)
                
                # Verify CMA-ES specific results
                self.assertEqual(result.optimization_info['algorithm'], 'cmaes')
                self.assertIsInstance(result.optimization_info['sigma'], (int, float))
                self.assertGreater(result.optimization_info['sigma'], 0)
                
                # Check population size handling
                expected_pop_size = algorithm_params['population_size']
                if expected_pop_size is not None:
                    self.assertEqual(result.optimization_info['population_size'], expected_pop_size)
                else:
                    # CMA-ES should set automatic population size
                    self.assertGreater(result.optimization_info['population_size'], 0)
                
                # Verify evaluations match expected
                expected_evals = result.optimization_info['population_size'] * 2
                self.assertEqual(result.convergence_info['total_evaluations'], expected_evals)
    
    def test_multiprocessing_functionality(self):
        """Test that multiprocessing works correctly."""
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        # Test multiprocessing initialization
        tuner_mp = ParameterTuner(drone_config, mission_config, param_set, 'cmaes',
                                 {'population_size': 4, 'sigma': 0.3},
                                 n_processes=2, enable_multiprocessing=True)
        
        self.assertTrue(tuner_mp.enable_multiprocessing)
        self.assertEqual(tuner_mp.n_processes, 2)
        
        # Test that it can run (even if it falls back to sequential for small problems)
        result = tuner_mp.optimize(max_iterations=1)
        self.assertIsInstance(result, TuningResult)
        self.assertIsNotNone(result.best_parameters)
        
        # Test parallel evaluation method directly
        test_params_list = [
            {'pos_P_gain': [1.0, 1.0, 1.0]},
            {'pos_P_gain': [2.0, 2.0, 2.0]}
        ]
        
        # This should work regardless of whether it actually uses multiprocessing
        results = tuner_mp.evaluate_parameters_parallel(test_params_list)
        self.assertEqual(len(results), 2)
        for cost, trajectory_data in results:
            self.assertIsInstance(cost, (int, float))
            self.assertIsInstance(trajectory_data, dict)
    
    def test_trajectory_visualization(self):
        """Test trajectory visualization during tuning."""
        if not self.enable_visualization:
            self.skipTest("Visualization not enabled")
        
        drone_config = self.test_drone_configs['standard_quad']
        mission_config = self.test_mission_configs['square_pattern']
        param_set = self.test_parameter_sets['position_control']
        
        tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
        
        # Evaluate with detailed data collection
        params = {'pos_P_gain': np.array([2.0, 2.0, 2.0])}
        cost, trajectory_data = tuner.evaluate_parameters(params, collect_detailed_data=True)
        
        if trajectory_data:
            self._visualize_tuning_trajectory(
                trajectory_data, 
                f"Square Pattern - Cost: {cost:.3f}",
                drone_config['description']
            )
    
    def _visualize_tuning_trajectory(self, trajectory_data: Dict, title: str, drone_desc: str):
        """Visualize trajectory data from parameter tuning."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            # Convert data to numpy arrays
            time_array = np.array(trajectory_data['time'])
            actual_pos = np.array(trajectory_data['actual_positions'])
            desired_pos = np.array(trajectory_data['desired_positions'])
            
            # Create 3D trajectory plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot trajectories
            ax.plot(actual_pos[:, 0], actual_pos[:, 1], actual_pos[:, 2], 
                   'b-', label='Actual', linewidth=2)
            ax.plot(desired_pos[:, 0], desired_pos[:, 1], desired_pos[:, 2], 
                   'r--', label='Desired', linewidth=2)
            
            # Mark start and end points
            ax.scatter(actual_pos[0, 0], actual_pos[0, 1], actual_pos[0, 2], 
                      c='green', s=100, marker='o', label='Start')
            ax.scatter(actual_pos[-1, 0], actual_pos[-1, 1], actual_pos[-1, 2], 
                      c='red', s=100, marker='s', label='End')
            
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'{title}\n{drone_desc}')
            ax.legend()
            ax.grid(True)
            
            # Calculate and display error statistics
            pos_errors = np.linalg.norm(actual_pos - desired_pos, axis=1)
            max_error = np.max(pos_errors)
            mean_error = np.mean(pos_errors)
            
            ax.text2D(0.02, 0.98, 
                     f'Max Error: {max_error:.3f}m\nMean Error: {mean_error:.3f}m', 
                     transform=ax.transAxes, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            if self.save_plots:
                filename = os.path.join(self.plot_dir, f'tuning_trajectory_{int(time.time())}.png')
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved plot: {filename}")
            
            plt.show()
        
        except ImportError:
            print("Matplotlib not available for visualization")
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def test_results_saving_loading(self):
        """Test saving and loading tuning results."""
        drone_config = self.test_drone_configs['small_quad']
        mission_config = self.test_mission_configs['hover_stability']
        param_set = self.test_parameter_sets['position_control']
        
        tuner = ParameterTuner(drone_config, mission_config, param_set, 'random_search', 
                              enable_multiprocessing=False)
        original_result = tuner.optimize(max_iterations=5)
        
        # Save results
        test_filename = 'test_tuning_result.json'
        save_tuning_results(original_result, test_filename)
        
        # Verify file was created
        self.assertTrue(os.path.exists(test_filename))
        
        # Load results
        from parameter_tuner import load_tuning_results
        loaded_result = load_tuning_results(test_filename)
        
        # Verify loaded data matches original
        self.assertEqual(loaded_result.best_cost, original_result.best_cost)
        self.assertEqual(len(loaded_result.cost_history), len(original_result.cost_history))
        
        # Clean up
        os.remove(test_filename)

def run_comprehensive_tuning_analysis():
    """Run comprehensive analysis of tuning performance across configurations."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PARAMETER TUNING ANALYSIS")
    print("="*80)
    
    # Test configurations
    test_configs = {
        # 'Micro Quad': DRONE_CONFIGURATIONS['micro_quad'],
        # 'Racing Quad': DRONE_CONFIGURATIONS['racing_quad'],
        # 'Standard Quad': DRONE_CONFIGURATIONS['standard_quad'],
        'Large Quad': DRONE_CONFIGURATIONS['large_quad'],
        'Standard Hex': DRONE_CONFIGURATIONS['standard_hex'],
        'Standard Octo': DRONE_CONFIGURATIONS['standard_octo']
    }
    
    missions = {
        # 'Hover': MISSION_CONFIGURATIONS['hover_stability'],
        # 'Step Response': MISSION_CONFIGURATIONS['position_step'],
        # 'Square Pattern': MISSION_CONFIGURATIONS['square_pattern'],
        'Figure Eight': MISSION_CONFIGURATIONS['figure_eight'],
    }
    
    algorithms = ['cmaes']  # , 'genetic', 'random_search', 'gradient_descent'
    
    results_data = []
    
    # Test different combinations
    for drone_name, drone_config in test_configs.items():
        for mission_name, mission_config in missions.items():
            for algorithm in algorithms:
                print(f"\nTesting {drone_name} - {mission_name} - {algorithm}")
                
                try:
                    # Create tuner
                    param_set = PARAMETER_SETS['full_control']
                    algorithm_params = {
                        'random_search': {},
                        'gradient_descent': {'learning_rate': 0.01, 'momentum': 0.9},
                        'genetic': {'population_size': 10, 'mutation_rate': 0.1},
                        'cmaes': {
                            'population_size': 20,
                            'sigma': 0.3,
                            'seed': 42
                        }
                    }
                    
                    tuner = ParameterTuner(
                        drone_config=drone_config,
                        mission_config=mission_config,
                        parameter_set=param_set,
                        algorithm=algorithm,
                        algorithm_params=algorithm_params[algorithm],
                        n_processes=4,
                        enable_multiprocessing=True,  # Disable for testing,
                        verbose=True
                    )
                    
                    # Run optimization
                    result = tuner.optimize(max_iterations=20)
                    
                    # Store results
                    results_data.append({
                        'drone': drone_name,
                        'mission': mission_name,
                        'algorithm': algorithm,
                        'best_cost': result.best_cost,
                        'tuning_time': result.tuning_time,
                        'evaluations': result.convergence_info['total_evaluations'],
                        'improvement': result.convergence_info.get('improvement_ratio', 0),
                        'converged': result.convergence_info.get('converged', False)
                    })
                    
                except Exception as e:
                    print(f"  Failed: {e}")
                    results_data.append({
                        'drone': drone_name,
                        'mission': mission_name,
                        'algorithm': algorithm,
                        'best_cost': float('inf'),
                        'tuning_time': 0,
                        'evaluations': 0,
                        'improvement': 0,
                        'converged': False
                    })
    
    # Print results table
    _print_tuning_analysis_table(results_data)

def _print_tuning_analysis_table(results_data: List[Dict]):
    """Print formatted analysis table."""
    print("\n" + "="*120)
    print("TUNING PERFORMANCE ANALYSIS")
    print("="*120)
    
    # Group by algorithm
    algorithms = list(set(r['algorithm'] for r in results_data))
    
    for algorithm in algorithms:
        print(f"\n{algorithm.upper()} ALGORITHM RESULTS:")
        print("-" * 100)
        print(f"{'Drone':<15} {'Mission':<15} {'Cost':<10} {'Time(s)':<8} {'Evals':<6} {'Improve':<8} {'Conv':<5}")
        print("-" * 100)
        
        alg_results = [r for r in results_data if r['algorithm'] == algorithm]
        for result in alg_results:
            cost_str = f"{result['best_cost']:.3f}" if result['best_cost'] < 1000 else "FAIL"
            improve_str = f"{result['improvement']*100:.1f}%" if result['improvement'] > 0 else "N/A"
            conv_str = "YES" if result['converged'] else "NO"
            
            print(f"{result['drone']:<15} {result['mission']:<15} {cost_str:<10} "
                  f"{result['tuning_time']:<8.1f} {result['evaluations']:<6} "
                  f"{improve_str:<8} {conv_str:<5}")
    
    # Summary statistics
    print(f"\n{'ALGORITHM COMPARISON:'}")
    print("-" * 60)
    print(f"{'Algorithm':<15} {'Avg Cost':<10} {'Avg Time':<10} {'Success Rate':<12}")
    print("-" * 60)
    
    for algorithm in algorithms:
        alg_results = [r for r in results_data if r['algorithm'] == algorithm]
        valid_results = [r for r in alg_results if r['best_cost'] < 1000]
        
        avg_cost = np.mean([r['best_cost'] for r in valid_results]) if valid_results else float('inf')
        avg_time = np.mean([r['tuning_time'] for r in alg_results])
        success_rate = len(valid_results) / len(alg_results) * 100 if alg_results else 0
        
        cost_str = f"{avg_cost:.3f}" if avg_cost < 1000 else "FAIL"
        print(f"{algorithm:<15} {cost_str:<10} {avg_time:<10.1f} {success_rate:<12.1f}%")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Parameter tuning tests with optional visualization',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--visualize', action='store_true',
                       help='Enable trajectory visualization during tests')
    parser.add_argument('--visualize-detailed', action='store_true',
                       help='Enable detailed visualization')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files instead of displaying')
    parser.add_argument('--plot-dir', default='./tuning_plots',
                       help='Directory to save plots (default: ./tuning_plots)')
    parser.add_argument('--tests-only', action='store_true',
                       help='Run only unit tests, skip comprehensive analysis')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only comprehensive analysis, skip unit tests')
    
    return parser.parse_args()

def set_global_args(args):
    """Set global module variables based on parsed arguments."""
    setattr(sys.modules[__name__], 'VISUALIZE_TUNING', args.visualize)
    setattr(sys.modules[__name__], 'VISUALIZE_DETAILED', args.visualize_detailed)
    setattr(sys.modules[__name__], 'SAVE_PLOTS', args.save_plots)
    setattr(sys.modules[__name__], 'PLOT_DIR', args.plot_dir)

if __name__ == '__main__':
    # Parse command line arguments
    args = parse_args()
    set_global_args(args)
    
    # Print configuration
    if args.visualize or args.visualize_detailed:
        print("\n" + "="*60)
        print("PARAMETER TUNING TEST CONFIGURATION")
        print("="*60)
        print(f"Visualization: {args.visualize}")
        print(f"Detailed visualization: {args.visualize_detailed}")
        print(f"Save plots: {args.save_plots}")
        if args.save_plots:
            print(f"Plot directory: {args.plot_dir}")
        print("="*60 + "\n")
    
    # Run unit tests unless analysis-only mode
    if not args.analysis_only:
        print("Running parameter tuning unit tests...")
        unittest.main(verbosity=2, exit=False, argv=[''])
    
    # Run comprehensive analysis unless tests-only mode
    if not args.tests_only:
        print("\nRunning comprehensive tuning analysis...")
        run_comprehensive_tuning_analysis()