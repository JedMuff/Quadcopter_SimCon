#!/usr/bin/env python3
"""
Parameter tuning framework for quadcopter controller optimization.

This module provides various optimization algorithms and utilities for tuning
controller parameters across different drone configurations and mission types.
"""

import numpy as np
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
import cmaes
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import functools

# Import local modules
from drone_simulator import ConfigurableQuadcopter
from Simulation.px4_based_ctrl import GeneralizedControl
from trajectory import Trajectory
from cost_functions import (
    multi_objective_cost, safety_constraint_penalty, 
    trajectory_specific_cost, robustness_cost
)
from tuning_configurations import (
    get_tuning_configuration, get_parameter_bounds, 
    get_default_parameters, generate_random_parameters
)

@dataclass
class TuningResult:
    """Container for tuning results."""
    best_parameters: Dict
    best_cost: float
    cost_history: List[float]
    parameter_history: List[Dict]
    optimization_info: Dict
    drone_config: Dict
    mission_config: Dict
    tuning_time: float
    convergence_info: Dict

def _evaluate_parameters_worker(args):
    """
    Worker function for multiprocessing parameter evaluation.
    
    Args:
        args: Tuple of (drone_config, mission_config, cost_weights, parameters, parameter_bounds, collect_detailed_data)
    
    Returns:
        Tuple of (cost, trajectory_data)
    """
    drone_config, mission_config, cost_weights, parameters, parameter_bounds, collect_detailed_data = args
    
    try:
        # Create drone and controller with given parameters
        # Filter out description field that's not part of ConfigurableQuadcopter parameters
        drone_params = {k: v for k, v in drone_config.items() if k != 'description'}
        quad = ConfigurableQuadcopter(0, **drone_params)
        controller = GeneralizedControl(quad, 1, auto_scale_gains=False)
        
        # Apply parameters to controller
        for param_name, param_value in parameters.items():
            if hasattr(controller, param_name):
                if isinstance(param_value, (list, tuple)):
                    setattr(controller, param_name, np.array(param_value))
                else:
                    setattr(controller, param_name, param_value)
        
        # Run simulation
        trajectory_data = _run_simulation_worker(quad, controller, mission_config, collect_detailed_data)
        
        # Calculate cost
        from cost_functions import multi_objective_cost, trajectory_specific_cost, safety_constraint_penalty
        
        if mission_config.get('trajectory_type') in ['hover', 'step', 'waypoint', 'smooth']:
            cost = trajectory_specific_cost(trajectory_data, mission_config['trajectory_type'])
        else:
            cost, _ = multi_objective_cost(trajectory_data, cost_weights)
        
        # Add safety constraint penalty
        safety_penalty = safety_constraint_penalty(parameters, parameter_bounds)
        total_cost = cost + safety_penalty
        
        return total_cost, trajectory_data
        
    except Exception as e:
        # Return high cost for failed simulations
        print(f"Simulation failed: {e}")
        return 1000.0, {}

def _run_simulation_worker(quad: ConfigurableQuadcopter, controller: GeneralizedControl, 
                          mission_config: Dict, collect_detailed_data: bool = False) -> Dict:
    """
    Worker function for running trajectory tracking simulation.
    
    Args:
        quad: Drone instance
        controller: Controller instance
        mission_config: Mission configuration
        collect_detailed_data: Whether to collect detailed trajectory data
    
    Returns:
        Dictionary containing simulation results
    """
    from trajectory import Trajectory
    
    waypoints = mission_config['waypoints']
    duration = mission_config['duration']
    dt = 0.005
    
    # Initialize simulation
    steps = int(duration / dt)
    waypoint_duration = duration / len(waypoints) if len(waypoints) > 1 else duration
    
    # Reset drone state
    initial_pos = waypoints[0] if len(waypoints) > 0 else [0, 0, -1]
    quad.drone_sim.set_state(position=initial_pos, velocity=[0, 0, 0],
                            attitude=[0, 0, 0], angular_velocity=[0, 0, 0])
    
    # Create trajectory
    traj = Trajectory(quad, "xyz_pos", [0, 0, 1.0])
    traj.ctrlType = "xyz_pos"
    
    # Data collection
    trajectory_data = {
        'time': [],
        'actual_positions': [],
        'desired_positions': [],
        'actual_velocities': [],
        'control_commands': []
    }
    
    if collect_detailed_data:
        trajectory_data.update({
            'actual_attitudes': [],
            'angular_velocities': [],
            'desired_states': []
        })
    
    # Run simulation
    for step in range(steps):
        current_time = step * dt
        
        # Determine current waypoint
        if len(waypoints) > 1:
            waypoint_idx = min(int(current_time / waypoint_duration), len(waypoints) - 1)
        else:
            waypoint_idx = 0
        
        # Set trajectory target
        traj.sDes = np.zeros(19)
        traj.sDes[0:3] = waypoints[waypoint_idx]
        
        # Run controller
        controller.controller(traj, quad, traj.sDes, dt)
        
        # Safety checks
        if not np.all(np.isfinite(controller.w_cmd)):
            break
        if np.any(controller.w_cmd < 0) or np.any(controller.w_cmd > 2000):
            break
        
        # Update drone
        quad.update(current_time, dt, controller.w_cmd, None)
        
        # Collect data
        trajectory_data['time'].append(current_time)
        trajectory_data['actual_positions'].append(quad.pos.copy())
        trajectory_data['desired_positions'].append(np.array(waypoints[waypoint_idx]).copy())
        trajectory_data['actual_velocities'].append(quad.vel.copy())
        trajectory_data['control_commands'].append(controller.w_cmd.copy())
        
        if collect_detailed_data:
            trajectory_data['actual_attitudes'].append(quad.euler.copy())
            trajectory_data['angular_velocities'].append(quad.omega.copy())
            trajectory_data['desired_states'].append(traj.sDes.copy())
        
        # Check for instability
        pos_error = np.linalg.norm(quad.pos - np.array(waypoints[waypoint_idx]))
        if pos_error > 20.0 or np.linalg.norm(quad.vel) > 30.0:
            break
    
    return trajectory_data

class ParameterTuner:
    """
    Main parameter tuning class supporting multiple optimization algorithms.
    """
    
    def __init__(self, drone_config: Dict, mission_config: Dict, parameter_set: Dict,
                 algorithm: str = 'gradient_descent', algorithm_params: Dict = None,
                 n_processes: Optional[int] = None, enable_multiprocessing: bool = True,
                 verbose: bool = False):
        """
        Initialize parameter tuner.
        
        Args:
            drone_config: Drone configuration dictionary
            mission_config: Mission configuration dictionary  
            parameter_set: Parameter set configuration dictionary
            algorithm: Optimization algorithm ('gradient_descent', 'genetic', 'bayesian', 'random_search', 'cmaes')
            algorithm_params: Algorithm-specific parameters
            n_processes: Number of processes for multiprocessing (None for auto-detection)
            enable_multiprocessing: Whether to enable multiprocessing for parallel evaluation
            verbose: Whether to print detailed progress information during optimization
        """
        self.drone_config = drone_config
        self.mission_config = mission_config
        self.parameter_set = parameter_set
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params or {}
        
        # Multiprocessing setup
        # Note: Multiprocessing is most beneficial for:
        # - Population-based algorithms (genetic, CMA-ES) with large populations
        # - Complex simulations that take significant time per evaluation
        # - Large parameter spaces requiring many evaluations
        # For small/fast simulations, sequential evaluation may be faster due to overhead
        self.enable_multiprocessing = enable_multiprocessing
        self.n_processes = n_processes or min(mp.cpu_count(), 8)  # Limit to 8 processes max
        self.verbose = verbose
        
        # Get parameter bounds
        self.parameter_bounds = parameter_set['bounds']
        self.parameter_names = parameter_set['parameters']
        
        # Initialize default parameters
        self.default_parameters = get_default_parameters()
        
        # Cost function setup
        self.cost_weights = mission_config.get('cost_weights', {
            'tracking_weight': 1.0,
            'stability_weight': 0.8,
            'efficiency_weight': 0.4
        })
        
        # Tuning history
        self.cost_history = []
        self.parameter_history = []
        self.evaluation_count = 0
        
        # Best result tracking
        self.best_cost = float('inf')
        self.best_parameters = None
        
    def evaluate_parameters(self, parameters: Dict, collect_detailed_data: bool = False) -> Tuple[float, Dict]:
        """
        Evaluate a set of controller parameters by running simulation.
        
        Args:
            parameters: Dictionary of parameter values to evaluate
            collect_detailed_data: Whether to collect detailed trajectory data
        
        Returns:
            Tuple of (cost, trajectory_data)
        """
        self.evaluation_count += 1
        
        try:
            # Create drone and controller with given parameters
            # Filter out description field that's not part of ConfigurableQuadcopter parameters
            drone_params = {k: v for k, v in self.drone_config.items() if k != 'description'}
            quad = ConfigurableQuadcopter(0, **drone_params)
            controller = GeneralizedControl(quad, 1, auto_scale_gains=False)
            
            # Apply parameters to controller
            self._apply_parameters_to_controller(controller, parameters)
            
            # Run simulation
            trajectory_data = self._run_simulation(quad, controller, collect_detailed_data)
            
            # Calculate cost
            if self.mission_config.get('trajectory_type') in ['hover', 'step', 'waypoint', 'smooth']:
                cost = trajectory_specific_cost(trajectory_data, self.mission_config['trajectory_type'])
            else:
                cost, _ = multi_objective_cost(trajectory_data, self.cost_weights)
            
            # Add safety constraint penalty
            safety_penalty = safety_constraint_penalty(parameters, self.parameter_bounds)
            total_cost = cost + safety_penalty
            
            # Update best result
            if total_cost < self.best_cost:
                self.best_cost = total_cost
                self.best_parameters = parameters.copy()
            
            # Record history
            self.cost_history.append(total_cost)
            self.parameter_history.append(parameters.copy())
            
            return total_cost, trajectory_data
            
        except Exception as e:
            # Return high cost for failed simulations
            print(f"Simulation failed: {e}")
            return 1000.0, {}
    
    def evaluate_parameters_parallel(self, parameter_list: List[Dict], collect_detailed_data: bool = False) -> List[Tuple[float, Dict]]:
        """
        Evaluate multiple parameter sets in parallel.
        
        Args:
            parameter_list: List of parameter dictionaries to evaluate
            collect_detailed_data: Whether to collect detailed trajectory data
        
        Returns:
            List of (cost, trajectory_data) tuples
        """
        if not self.enable_multiprocessing or len(parameter_list) == 1 or len(parameter_list) < self.n_processes:
            # Fall back to sequential evaluation for small batches
            results = []
            for params in parameter_list:
                cost, trajectory_data = self.evaluate_parameters(params, collect_detailed_data)
                results.append((cost, trajectory_data))
            return results
        
        # Prepare arguments for parallel processing
        args_list = []
        for params in parameter_list:
            args = (
                self.drone_config,
                self.mission_config, 
                self.cost_weights,
                params,
                self.parameter_bounds,
                collect_detailed_data
            )
            args_list.append(args)
        
        # Run parallel evaluation
        results = []
        try:
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                # Submit all jobs
                future_to_params = {
                    executor.submit(_evaluate_parameters_worker, args): params 
                    for args, params in zip(args_list, parameter_list)
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_params):
                    params = future_to_params[future]
                    try:
                        cost, trajectory_data = future.result()
                        results.append((cost, trajectory_data))
                        
                        # Update evaluation count and history
                        self.evaluation_count += 1
                        
                        # Update best result
                        if cost < self.best_cost:
                            self.best_cost = cost
                            self.best_parameters = params.copy()
                        
                        # Record history
                        self.cost_history.append(cost)
                        self.parameter_history.append(params.copy())
                        
                    except Exception as e:
                        print(f"Parameter evaluation failed: {e}")
                        results.append((1000.0, {}))
                        
        except Exception as e:
            print(f"Parallel processing failed, falling back to sequential: {e}")
            # Fall back to sequential processing
            results = []
            for params in parameter_list:
                cost, trajectory_data = self.evaluate_parameters(params, collect_detailed_data)
                results.append((cost, trajectory_data))
        
        return results
    
    def _apply_parameters_to_controller(self, controller: GeneralizedControl, parameters: Dict):
        """Apply parameter values to controller instance."""
        for param_name, param_value in parameters.items():
            if hasattr(controller, param_name):
                if isinstance(param_value, (list, tuple)):
                    setattr(controller, param_name, np.array(param_value))
                else:
                    setattr(controller, param_name, param_value)
    
    def _run_simulation(self, quad: ConfigurableQuadcopter, controller: GeneralizedControl,
                       collect_detailed_data: bool = False) -> Dict:
        """
        Run trajectory tracking simulation.
        
        Args:
            quad: Drone instance
            controller: Controller instance
            collect_detailed_data: Whether to collect detailed trajectory data
        
        Returns:
            Dictionary containing simulation results
        """
        waypoints = self.mission_config['waypoints']
        duration = self.mission_config['duration']
        dt = 0.005
        
        # Initialize simulation
        steps = int(duration / dt)
        waypoint_duration = duration / len(waypoints) if len(waypoints) > 1 else duration
        
        # Reset drone state
        initial_pos = waypoints[0] if len(waypoints) > 0 else [0, 0, -1]
        quad.drone_sim.set_state(position=initial_pos, velocity=[0, 0, 0],
                                attitude=[0, 0, 0], angular_velocity=[0, 0, 0])
        
        # Create trajectory
        traj = Trajectory(quad, "xyz_pos", [0, 0, 1.0])
        traj.ctrlType = "xyz_pos"
        
        # Data collection
        trajectory_data = {
            'time': [],
            'actual_positions': [],
            'desired_positions': [],
            'actual_velocities': [],
            'control_commands': []
        }
        
        if collect_detailed_data:
            trajectory_data.update({
                'actual_attitudes': [],
                'angular_velocities': [],
                'desired_states': []
            })
        
        # Run simulation
        for step in range(steps):
            current_time = step * dt
            
            # Determine current waypoint
            if len(waypoints) > 1:
                waypoint_idx = min(int(current_time / waypoint_duration), len(waypoints) - 1)
            else:
                waypoint_idx = 0
            
            # Set trajectory target
            traj.sDes = np.zeros(19)
            traj.sDes[0:3] = waypoints[waypoint_idx]
            
            # Run controller
            controller.controller(traj, quad, traj.sDes, dt)
            
            # Safety checks
            if not np.all(np.isfinite(controller.w_cmd)):
                break
            if np.any(controller.w_cmd < 0) or np.any(controller.w_cmd > 2000):
                break
            
            # Update drone
            quad.update(current_time, dt, controller.w_cmd, None)
            
            # Collect data
            trajectory_data['time'].append(current_time)
            trajectory_data['actual_positions'].append(quad.pos.copy())
            trajectory_data['desired_positions'].append(np.array(waypoints[waypoint_idx]).copy())
            trajectory_data['actual_velocities'].append(quad.vel.copy())
            trajectory_data['control_commands'].append(controller.w_cmd.copy())
            
            if collect_detailed_data:
                trajectory_data['actual_attitudes'].append(quad.euler.copy())
                trajectory_data['angular_velocities'].append(quad.omega.copy())
                trajectory_data['desired_states'].append(traj.sDes.copy())
            
            # Check for instability
            pos_error = np.linalg.norm(quad.pos - np.array(waypoints[waypoint_idx]))
            if pos_error > 20.0 or np.linalg.norm(quad.vel) > 30.0:
                break
        
        return trajectory_data
    
    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-4) -> TuningResult:
        """
        Run parameter optimization using selected algorithm.
        
        Args:
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance
        
        Returns:
            TuningResult containing optimization results
        """
        start_time = time.time()
        
        if self.algorithm == 'gradient_descent':
            result = self._optimize_gradient_descent(max_iterations, tolerance)
        elif self.algorithm == 'genetic':
            result = self._optimize_genetic(max_iterations)
        elif self.algorithm == 'bayesian':
            result = self._optimize_bayesian(max_iterations)
        elif self.algorithm == 'random_search':
            result = self._optimize_random_search(max_iterations)
        elif self.algorithm == 'cmaes':
            result = self._optimize_cmaes(max_iterations)
        else:
            raise ValueError(f"Unknown optimization algorithm: {self.algorithm}")
        
        tuning_time = time.time() - start_time
        
        return TuningResult(
            best_parameters=self.best_parameters,
            best_cost=self.best_cost,
            cost_history=self.cost_history,
            parameter_history=self.parameter_history,
            optimization_info=result,
            drone_config=self.drone_config,
            mission_config=self.mission_config,
            tuning_time=tuning_time,
            convergence_info=self._analyze_convergence()
        )
    
    def _optimize_gradient_descent(self, max_iterations: int, tolerance: float) -> Dict:
        """Gradient descent optimization."""
        learning_rate = self.algorithm_params.get('learning_rate', 0.01)
        momentum = self.algorithm_params.get('momentum', 0.9)
        adaptive_lr = self.algorithm_params.get('adaptive_lr', True)
        
        if self.verbose:
            print(f"\nStarting Gradient Descent optimization:")
            print(f"  Learning rate: {learning_rate}")
            print(f"  Momentum: {momentum}")
            print(f"  Adaptive LR: {adaptive_lr}")
            print(f"  Max iterations: {max_iterations}")
            print(f"  Tolerance: {tolerance}")
            print("-" * 60)
        
        # Initialize parameters
        current_params = self._initialize_parameters()
        velocity = {name: np.zeros_like(value) for name, value in current_params.items()}
        
        best_cost_improvement = 0
        lr_reduction_count = 0
        
        for iteration in range(max_iterations):
            # Evaluate current parameters
            iteration_start_time = time.time()
            current_cost, _ = self.evaluate_parameters(current_params)
            
            # Calculate gradients (numerical differentiation)
            gradients = self._calculate_gradients(current_params)
            iteration_time = time.time() - iteration_start_time
            
            if self.verbose:
                print(f"Iteration {iteration + 1}: cost={current_cost:.4f}, lr={learning_rate:.6f}, time={iteration_time:.2f}s")
            
            # Update parameters with momentum
            for param_name in current_params.keys():
                if param_name in gradients:
                    # Update velocity with momentum
                    velocity[param_name] = momentum * velocity[param_name] - learning_rate * gradients[param_name]
                    
                    # Update parameters
                    current_params[param_name] = current_params[param_name] + velocity[param_name]
                    
                    # Apply bounds
                    current_params[param_name] = self._apply_bounds(param_name, current_params[param_name])
            
            # Adaptive learning rate
            if adaptive_lr and iteration > 10:
                recent_improvement = min(self.cost_history[-10:]) - self.cost_history[-1]
                if recent_improvement < best_cost_improvement * 0.1:
                    learning_rate *= 0.8
                    lr_reduction_count += 1
                    if lr_reduction_count >= 3:
                        break
                else:
                    best_cost_improvement = recent_improvement
            
            # Check convergence
            if iteration > 5:
                cost_change = abs(self.cost_history[-1] - self.cost_history[-6])
                if cost_change < tolerance:
                    break
        
        return {
            'algorithm': 'gradient_descent',
            'iterations': iteration + 1,
            'final_learning_rate': learning_rate,
            'lr_reductions': lr_reduction_count
        }
    
    def _optimize_genetic(self, max_iterations: int) -> Dict:
        """Genetic algorithm optimization."""
        population_size = self.algorithm_params.get('population_size', 20)
        mutation_rate = self.algorithm_params.get('mutation_rate', 0.1)
        crossover_rate = self.algorithm_params.get('crossover_rate', 0.8)
        elite_size = self.algorithm_params.get('elite_size', 2)
        
        if self.verbose:
            print(f"\nStarting Genetic Algorithm optimization:")
            print(f"  Population size: {population_size}")
            print(f"  Mutation rate: {mutation_rate}")
            print(f"  Crossover rate: {crossover_rate}")
            print(f"  Elite size: {elite_size}")
            print(f"  Max generations: {max_iterations}")
            print(f"  Multiprocessing: {'enabled' if self.enable_multiprocessing else 'disabled'}")
            print("-" * 60)
        
        # Initialize population
        population = []
        initial_params = [self._initialize_parameters() for _ in range(population_size)]
        
        # Evaluate initial population in parallel
        generation_start_time = time.time()
        results = self.evaluate_parameters_parallel(initial_params, collect_detailed_data=False)
        costs = [result[0] for result in results]
        generation_time = time.time() - generation_start_time
        
        population = initial_params
        fitness_scores = [1.0 / (1.0 + cost) for cost in costs]  # Convert cost to fitness
        
        if self.verbose:
            min_cost = min(costs)
            avg_cost = np.mean(costs)
            max_cost = max(costs)
            print(f"Generation 0: min={min_cost:.4f}, avg={avg_cost:.4f}, max={max_cost:.4f}, time={generation_time:.2f}s")
        
        for generation in range(max_iterations):
            # Selection, crossover, and mutation
            new_population = []
            
            # Keep elite individuals
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate rest of population
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                if np.random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    child1 = self._mutate(child1)
                if np.random.random() < mutation_rate:
                    child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            # Trim to population size
            new_population = new_population[:population_size]
            
            # Evaluate new population in parallel
            generation_start_time = time.time()
            population = new_population
            results = self.evaluate_parameters_parallel(population, collect_detailed_data=False)
            costs = [result[0] for result in results]
            fitness_scores = [1.0 / (1.0 + cost) for cost in costs]
            generation_time = time.time() - generation_start_time
            
            if self.verbose:
                min_cost = min(costs)
                avg_cost = np.mean(costs)
                max_cost = max(costs)
                print(f"Generation {generation + 1}: min={min_cost:.4f}, avg={avg_cost:.4f}, max={max_cost:.4f}, time={generation_time:.2f}s")
        
        return {
            'algorithm': 'genetic',
            'generations': max_iterations,
            'population_size': population_size,
            'final_best_fitness': max(fitness_scores)
        }
    
    def _optimize_bayesian(self, max_iterations: int) -> Dict:
        """Bayesian optimization (simplified implementation)."""
        # This is a placeholder for Bayesian optimization
        # In practice, you would use libraries like scikit-optimize
        
        # For now, use random search with some intelligent sampling
        exploration_rate = self.algorithm_params.get('exploration_rate', 0.3)
        
        if self.verbose:
            print(f"\nStarting Bayesian optimization (simplified):")
            print(f"  Exploration rate: {exploration_rate}")
            print(f"  Max iterations: {max_iterations}")
            print("-" * 60)
        
        costs = []
        for iteration in range(max_iterations):
            iteration_start_time = time.time()
            if iteration < 5 or np.random.random() < exploration_rate:
                # Exploration: random sampling
                params = self._initialize_parameters()
                mode = "explore"
            else:
                # Exploitation: sample around best parameters
                params = self._sample_around_best()
                mode = "exploit"
            
            cost, _ = self.evaluate_parameters(params)
            costs.append(cost)
            iteration_time = time.time() - iteration_start_time
            
            if self.verbose:
                if iteration == 0:
                    print(f"Iteration {iteration + 1}: cost={cost:.4f}, mode={mode}, time={iteration_time:.2f}s")
                elif (iteration + 1) % 10 == 0 or iteration == max_iterations - 1:
                    min_cost = min(costs)
                    avg_cost = np.mean(costs)
                    max_cost = max(costs)
                    print(f"Iteration {iteration + 1}: current={cost:.4f}, min={min_cost:.4f}, avg={avg_cost:.4f}, max={max_cost:.4f}, mode={mode}, time={iteration_time:.2f}s")
        
        return {
            'algorithm': 'bayesian',
            'iterations': max_iterations,
            'exploration_rate': exploration_rate
        }
    
    def _optimize_random_search(self, max_iterations: int) -> Dict:
        """Random search optimization."""
        if self.verbose:
            print(f"\nStarting Random Search optimization:")
            print(f"  Max iterations: {max_iterations}")
            print("-" * 60)
        
        costs = []
        for iteration in range(max_iterations):
            iteration_start_time = time.time()
            params = self._initialize_parameters()
            cost, _ = self.evaluate_parameters(params)
            costs.append(cost)
            iteration_time = time.time() - iteration_start_time
            
            if self.verbose:
                if iteration == 0:
                    print(f"Iteration {iteration + 1}: cost={cost:.4f}, time={iteration_time:.2f}s")
                elif (iteration + 1) % 10 == 0 or iteration == max_iterations - 1:
                    min_cost = min(costs)
                    avg_cost = np.mean(costs)
                    max_cost = max(costs)
                    print(f"Iteration {iteration + 1}: current={cost:.4f}, min={min_cost:.4f}, avg={avg_cost:.4f}, max={max_cost:.4f}, time={iteration_time:.2f}s")
        
        return {
            'algorithm': 'random_search',
            'iterations': max_iterations,
            'evaluations': self.evaluation_count
        }
    
    def _optimize_cmaes(self, max_iterations: int) -> Dict:
        """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization."""
        # Get algorithm parameters
        population_size = self.algorithm_params.get('population_size', None)  # Let CMA-ES decide
        sigma = self.algorithm_params.get('sigma', 0.3)
        seed = self.algorithm_params.get('seed', None)
        
        # Prepare parameter bounds and initial values
        param_names = list(self.parameter_names)
        param_dimensions = []
        param_bounds = []
        initial_values = []
        
        # Calculate total dimensions and prepare bounds
        total_dims = 0
        for param_name in param_names:
            if param_name in self.parameter_bounds:
                bounds = self.parameter_bounds[param_name]
                min_vals = np.array(bounds['min'])
                max_vals = np.array(bounds['max'])
                
                if min_vals.ndim == 0:  # Scalar parameter
                    param_dimensions.append(1)
                    param_bounds.append((min_vals.item(), max_vals.item()))
                    # Use midpoint as initial value
                    initial_values.append((min_vals.item() + max_vals.item()) / 2)
                    total_dims += 1
                else:  # Vector parameter
                    param_dimensions.append(len(min_vals))
                    for i in range(len(min_vals)):
                        param_bounds.append((min_vals[i], max_vals[i]))
                        initial_values.append((min_vals[i] + max_vals[i]) / 2)
                    total_dims += len(min_vals)
        
        # Initialize CMA-ES optimizer
        optimizer = cmaes.CMA(
            mean=np.array(initial_values),
            sigma=sigma,
            bounds=np.array(param_bounds),
            population_size=population_size,
            seed=seed
        )
        
        # Optimization loop
        generation = 0
        best_cost_overall = float('inf')
        
        if self.verbose:
            print(f"\nStarting CMA-ES optimization:")
            print(f"  Population size: {optimizer.population_size}")
            print(f"  Initial sigma: {sigma}")
            print(f"  Max generations: {max_iterations}")
            print(f"  Multiprocessing: {'enabled' if self.enable_multiprocessing else 'disabled'}")
            print("-" * 60)
        
        for generation in range(max_iterations):
            solutions = []
            param_dicts = []
            
            # Generate population
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                
                # Convert flat parameter vector back to parameter dictionary
                param_dict = self._vector_to_params(x, param_names, param_dimensions)
                
                solutions.append(x)
                param_dicts.append(param_dict)
            
            # Evaluate population in parallel
            generation_start_time = time.time()
            results = self.evaluate_parameters_parallel(param_dicts, collect_detailed_data=False)
            costs = [result[0] for result in results]
            generation_time = time.time() - generation_start_time
            
            # Update best overall cost
            for cost in costs:
                if cost < best_cost_overall:
                    best_cost_overall = cost
            
            # Tell CMA-ES about the results
            optimizer.tell([(x, y) for x, y in zip(solutions, costs)])
            
            # Verbose logging
            if self.verbose:
                min_cost = min(costs)
                avg_cost = np.mean(costs)
                max_cost = max(costs)
                print(f"Generation {generation + 1}: min={min_cost:.4f}, avg={avg_cost:.4f}, max={max_cost:.4f}, time={generation_time:.2f}s")
            
            # Check for convergence
            if optimizer.should_stop():
                if self.verbose:
                    print("CMA-ES convergence criteria met.")
                break
        
        return {
            'algorithm': 'cmaes',
            'generations': generation + 1,
            'population_size': optimizer.population_size,
            'initial_sigma': sigma,
            'final_best_cost': best_cost_overall,
            'convergence_criteria': optimizer.should_stop()
        }
    
    def _vector_to_params(self, x: np.ndarray, param_names: List[str], param_dimensions: List[int]) -> Dict:
        """Convert flat parameter vector to parameter dictionary."""
        params = {}
        idx = 0
        
        for i, param_name in enumerate(param_names):
            dim = param_dimensions[i]
            if dim == 1:
                params[param_name] = x[idx]
                idx += 1
            else:
                params[param_name] = np.array(x[idx:idx+dim])
                idx += dim
        
        return params
    
    def _initialize_parameters(self) -> Dict:
        """Initialize random parameters within bounds."""
        params = {}
        
        for param_name in self.parameter_names:
            if param_name in self.parameter_set.get('initial_range', {}):
                # Use initial range if available
                min_val = np.array(self.parameter_set['initial_range'][param_name]['min'])
                max_val = np.array(self.parameter_set['initial_range'][param_name]['max'])
                random_vals = np.random.uniform(0, 1, size=min_val.shape)
                params[param_name] = min_val + random_vals * (max_val - min_val)
            elif param_name in self.parameter_bounds:
                # Use bounds
                min_val = np.array(self.parameter_bounds[param_name]['min'])
                max_val = np.array(self.parameter_bounds[param_name]['max'])
                random_vals = np.random.uniform(0, 1, size=min_val.shape)
                params[param_name] = min_val + random_vals * (max_val - min_val)
            else:
                # Use default
                if param_name in self.default_parameters:
                    params[param_name] = self.default_parameters[param_name].copy()
        
        return params
    
    def _calculate_gradients(self, parameters: Dict, epsilon: float = 1e-6) -> Dict:
        """Calculate numerical gradients."""
        gradients = {}
        base_cost, _ = self.evaluate_parameters(parameters)
        
        for param_name, param_value in parameters.items():
            if param_name in self.parameter_names:
                param_grad = np.zeros_like(param_value)
                
                if param_value.ndim == 0:  # Scalar parameter
                    # Forward difference
                    perturbed_params = parameters.copy()
                    perturbed_params[param_name] = param_value + epsilon
                    cost_plus, _ = self.evaluate_parameters(perturbed_params)
                    param_grad = (cost_plus - base_cost) / epsilon
                else:  # Vector parameter
                    for i in range(len(param_value)):
                        perturbed_params = parameters.copy()
                        perturbed_params[param_name] = param_value.copy()
                        perturbed_params[param_name][i] += epsilon
                        cost_plus, _ = self.evaluate_parameters(perturbed_params)
                        param_grad[i] = (cost_plus - base_cost) / epsilon
                
                gradients[param_name] = param_grad
        
        return gradients
    
    def _apply_bounds(self, param_name: str, param_value: np.ndarray) -> np.ndarray:
        """Apply parameter bounds."""
        if param_name in self.parameter_bounds:
            bounds = self.parameter_bounds[param_name]
            if 'min' in bounds:
                param_value = np.maximum(param_value, bounds['min'])
            if 'max' in bounds:
                param_value = np.minimum(param_value, bounds['max'])
        return param_value
    
    def _tournament_selection(self, population: List[Dict], fitness_scores: List[float], 
                            tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm."""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Single-point crossover for genetic algorithm."""
        child1, child2 = parent1.copy(), parent2.copy()
        
        for param_name in parent1.keys():
            if param_name in self.parameter_names:
                if np.random.random() < 0.5:
                    child1[param_name] = parent2[param_name].copy()
                    child2[param_name] = parent1[param_name].copy()
        
        return child1, child2
    
    def _mutate(self, individual: Dict, mutation_strength: float = 0.1) -> Dict:
        """Mutation for genetic algorithm."""
        mutated = individual.copy()
        
        for param_name in individual.keys():
            if param_name in self.parameter_names and np.random.random() < 0.1:
                param_value = individual[param_name]
                noise = np.random.normal(0, mutation_strength, size=param_value.shape)
                mutated[param_name] = param_value + noise * param_value
                mutated[param_name] = self._apply_bounds(param_name, mutated[param_name])
        
        return mutated
    
    def _sample_around_best(self, std_factor: float = 0.1) -> Dict:
        """Sample parameters around current best."""
        if self.best_parameters is None:
            return self._initialize_parameters()
        
        params = {}
        for param_name, param_value in self.best_parameters.items():
            noise = np.random.normal(0, std_factor, size=param_value.shape)
            params[param_name] = param_value + noise * param_value
            params[param_name] = self._apply_bounds(param_name, params[param_name])
        
        return params
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence characteristics."""
        if len(self.cost_history) < 3:
            return {
                'converged': False, 
                'reason': 'insufficient_data',
                'total_evaluations': self.evaluation_count,
                'improvement_ratio': 0
            }
        
        # Check for convergence based on cost improvement
        if len(self.cost_history) >= 10:
            recent_costs = self.cost_history[-10:]
            cost_std = np.std(recent_costs)
            cost_trend = np.polyfit(range(10), recent_costs, 1)[0]
            converged = cost_std < 0.01 and abs(cost_trend) < 0.001
        else:
            recent_costs = self.cost_history[-3:]
            cost_std = np.std(recent_costs)
            cost_trend = 0
            converged = False
        
        return {
            'converged': converged,
            'final_cost_std': cost_std,
            'cost_trend': cost_trend,
            'total_evaluations': self.evaluation_count,
            'improvement_ratio': (self.cost_history[0] - self.best_cost) / self.cost_history[0] if len(self.cost_history) > 0 and self.cost_history[0] > 0 else 0
        }

def save_tuning_results(result: TuningResult, filename: str):
    """Save tuning results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    result_dict = asdict(result)
    result_dict = convert_numpy(result_dict)
    result_dict['timestamp'] = datetime.now().isoformat()
    
    with open(filename, 'w') as f:
        json.dump(result_dict, f, indent=2)

def load_tuning_results(filename: str) -> TuningResult:
    """Load tuning results from JSON file."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Remove timestamp if present (not part of TuningResult)
    if 'timestamp' in data:
        del data['timestamp']
    
    # Convert lists back to numpy arrays
    def convert_lists(obj):
        if isinstance(obj, dict):
            converted = {}
            for k, v in obj.items():
                if k.endswith('_gain') or k.endswith('_max') and isinstance(v, list):
                    converted[k] = np.array(v)
                else:
                    converted[k] = convert_lists(v)
            return converted
        elif isinstance(obj, list):
            return [convert_lists(item) for item in obj]
        else:
            return obj
    
    data = convert_lists(data)
    return TuningResult(**data)

def run_tuning_session(scenario_name: str, algorithm: str = 'gradient_descent',
                      max_iterations: int = 50, algorithm_params: Dict = None,
                      n_processes: Optional[int] = None, enable_multiprocessing: bool = True) -> TuningResult:
    """
    Run a complete tuning session for a predefined scenario.
    
    Args:
        scenario_name: Name of tuning scenario
        algorithm: Optimization algorithm to use
        max_iterations: Maximum optimization iterations
        algorithm_params: Algorithm-specific parameters
        n_processes: Number of processes for multiprocessing (None for auto-detection)
        enable_multiprocessing: Whether to enable multiprocessing for parallel evaluation
    
    Returns:
        TuningResult containing optimization results
    """
    # Get configuration
    config = get_tuning_configuration(scenario_name)
    
    # Create tuner
    tuner = ParameterTuner(
        drone_config=config['drone_config'],
        mission_config=config['mission_config'],
        parameter_set=config['parameter_set'],
        algorithm=algorithm,
        algorithm_params=algorithm_params,
        n_processes=n_processes,
        enable_multiprocessing=enable_multiprocessing
    )
    
    # Run optimization
    print(f"Starting tuning session: {scenario_name}")
    print(f"Algorithm: {algorithm}, Max iterations: {max_iterations}")
    
    result = tuner.optimize(max_iterations=max_iterations)
    
    print(f"Tuning completed in {result.tuning_time:.2f} seconds")
    print(f"Best cost: {result.best_cost:.4f}")
    print(f"Total evaluations: {result.convergence_info['total_evaluations']}")
    
    return result