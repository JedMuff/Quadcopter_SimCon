#!/usr/bin/env python3
"""
Cost functions for controller parameter tuning.

This module provides various cost function implementations for evaluating
controller performance during parameter optimization.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

def tracking_error_cost(trajectory_data: Dict, weights: Dict = None) -> float:
    """
    Calculate cost based on position tracking error.
    
    Args:
        trajectory_data: Dictionary containing simulation results with keys:
            - 'actual_positions': List of [x, y, z] positions
            - 'desired_positions': List of [x, y, z] desired positions
            - 'time': List of time stamps
        weights: Weight factors for different error components
            - 'rms_weight': Weight for RMS error (default: 1.0)
            - 'max_weight': Weight for maximum error (default: 0.5)
            - 'settling_weight': Weight for settling time penalty (default: 0.3)
    
    Returns:
        Cost value (lower is better)
    """
    if weights is None:
        weights = {'rms_weight': 1.0, 'max_weight': 0.5, 'settling_weight': 0.3}
    
    actual_pos = np.array(trajectory_data['actual_positions'])
    desired_pos = np.array(trajectory_data['desired_positions'])
    time_array = np.array(trajectory_data['time'])
    
    # Calculate position errors
    position_errors = np.linalg.norm(actual_pos - desired_pos, axis=1)
    
    # RMS error component
    rms_error = np.sqrt(np.mean(position_errors**2))
    
    # Maximum error component
    max_error = np.max(position_errors)
    
    # Settling time component (time to reach within 5% of final error)
    settling_cost = calculate_settling_time_cost(position_errors, time_array)
    
    # Combine costs
    total_cost = (weights['rms_weight'] * rms_error + 
                  weights['max_weight'] * max_error + 
                  weights['settling_weight'] * settling_cost)
    
    return total_cost

def stability_cost(trajectory_data: Dict, weights: Dict = None) -> float:
    """
    Calculate cost based on flight stability metrics.
    
    Args:
        trajectory_data: Dictionary containing simulation results
        weights: Weight factors for stability components
    
    Returns:
        Stability cost (lower is better)
    """
    if weights is None:
        weights = {'oscillation_weight': 1.0, 'overshoot_weight': 0.8, 'attitude_weight': 0.6}
    
    actual_pos = np.array(trajectory_data['actual_positions'])
    desired_pos = np.array(trajectory_data['desired_positions'])
    time_array = np.array(trajectory_data['time'])
    
    # Oscillation detection
    oscillation_cost = detect_oscillations(actual_pos, time_array)
    
    # Overshoot penalty
    overshoot_cost = calculate_overshoot(actual_pos, desired_pos)
    
    # Attitude stability (if available)
    attitude_cost = 0.0
    if 'actual_attitudes' in trajectory_data:
        attitudes = np.array(trajectory_data['actual_attitudes'])
        attitude_cost = np.mean(np.linalg.norm(attitudes, axis=1))
    
    total_cost = (weights['oscillation_weight'] * oscillation_cost +
                  weights['overshoot_weight'] * overshoot_cost +
                  weights['attitude_weight'] * attitude_cost)
    
    return total_cost

def control_effort_cost(trajectory_data: Dict, weights: Dict = None) -> float:
    """
    Calculate cost based on control effort efficiency.
    
    Args:
        trajectory_data: Dictionary containing simulation results
        weights: Weight factors for control effort components
    
    Returns:
        Control effort cost (lower is better)
    """
    if weights is None:
        weights = {'command_weight': 1.0, 'smoothness_weight': 0.5}
    
    if 'control_commands' not in trajectory_data:
        return 0.0
    
    commands = np.array(trajectory_data['control_commands'])
    
    # Normalize motor commands (typical range 1000-2000, normalize to 0-1)
    normalized_commands = (commands - 1000.0) / 1000.0
    normalized_commands = np.clip(normalized_commands, 0, 2)  # Allow some overhead
    
    # Total control effort (normalized)
    effort_cost = np.mean(np.sum(normalized_commands**2, axis=1))
    
    # Control smoothness (rate of change of normalized commands)
    smoothness_cost = 0.0
    if len(normalized_commands) > 1:
        command_derivatives = np.diff(normalized_commands, axis=0)
        smoothness_cost = np.mean(np.sum(command_derivatives**2, axis=1))
    
    total_cost = (weights['command_weight'] * effort_cost +
                  weights['smoothness_weight'] * smoothness_cost)
    
    return total_cost

def multi_objective_cost(trajectory_data: Dict, 
                        cost_weights: Dict = None,
                        component_weights: Dict = None) -> Tuple[float, Dict]:
    """
    Calculate multi-objective cost combining tracking, stability, and efficiency.
    
    Args:
        trajectory_data: Dictionary containing simulation results
        cost_weights: Weights for different cost categories
        component_weights: Detailed weights for cost components
    
    Returns:
        Tuple of (total_cost, cost_breakdown)
    """
    if cost_weights is None:
        cost_weights = {
            'tracking_weight': 1.0,
            'stability_weight': 0.6,
            'efficiency_weight': 0.3
        }
    
    # Calculate individual cost components
    tracking_cost = tracking_error_cost(trajectory_data, 
                                      component_weights.get('tracking', None) if component_weights else None)
    stability_cost_val = stability_cost(trajectory_data,
                                      component_weights.get('stability', None) if component_weights else None)
    efficiency_cost = control_effort_cost(trajectory_data,
                                        component_weights.get('efficiency', None) if component_weights else None)
    
    # Combine costs
    total_cost = (cost_weights['tracking_weight'] * tracking_cost +
                  cost_weights['stability_weight'] * stability_cost_val +
                  cost_weights['efficiency_weight'] * efficiency_cost)
    
    cost_breakdown = {
        'total': total_cost,
        'tracking': tracking_cost,
        'stability': stability_cost_val,
        'efficiency': efficiency_cost
    }
    
    return total_cost, cost_breakdown

def safety_constraint_penalty(parameters: Dict, bounds: Dict) -> float:
    """
    Calculate penalty for parameters that exceed safety bounds.
    
    Args:
        parameters: Dictionary of controller parameters
        bounds: Dictionary of parameter bounds (min, max values)
    
    Returns:
        Penalty cost (0 if within bounds, positive otherwise)
    """
    penalty = 0.0
    
    for param_name, param_value in parameters.items():
        if param_name in bounds:
            param_bounds = bounds[param_name]
            param_array = np.array(param_value) if isinstance(param_value, (list, tuple)) else np.array([param_value])
            
            # Check lower bounds
            if 'min' in param_bounds:
                min_val = param_bounds['min']
                min_violations = np.maximum(0, min_val - param_array)
                penalty += np.sum(min_violations**2) * 1000  # Heavy penalty
            
            # Check upper bounds
            if 'max' in param_bounds:
                max_val = param_bounds['max']
                max_violations = np.maximum(0, param_array - max_val)
                penalty += np.sum(max_violations**2) * 1000  # Heavy penalty
    
    return penalty

def calculate_settling_time_cost(errors: np.ndarray, time: np.ndarray, 
                               tolerance: float = 0.05) -> float:
    """
    Calculate settling time cost based on how long it takes to reach steady state.
    
    Args:
        errors: Array of position errors over time
        time: Array of time stamps
        tolerance: Tolerance for steady-state (fraction of final error)
    
    Returns:
        Settling time cost
    """
    if len(errors) < 10:
        return 100.0  # Heavy penalty for very short simulations
    
    # Estimate steady-state error from last 20% of trajectory
    steady_state_start = int(0.8 * len(errors))
    steady_state_error = np.mean(errors[steady_state_start:])
    
    # Find settling time (last time error exceeded tolerance)
    threshold = steady_state_error * (1 + tolerance)
    settling_indices = np.where(errors > threshold)[0]
    
    if len(settling_indices) == 0:
        return 0.0  # Already settled
    
    settling_time = time[settling_indices[-1]]
    total_time = time[-1]
    
    # Normalize settling time as fraction of total time
    settling_ratio = settling_time / total_time if total_time > 0 else 1.0
    
    return settling_ratio * 10.0  # Scale to reasonable cost range

def detect_oscillations(positions: np.ndarray, time: np.ndarray, 
                       min_period: float = 0.5) -> float:
    """
    Detect oscillatory behavior in position data.
    
    Args:
        positions: Array of positions [N, 3]
        time: Array of time stamps
        min_period: Minimum period to consider as oscillation
    
    Returns:
        Oscillation cost (higher for more oscillatory behavior)
    """
    if len(positions) < 20:
        return 0.0
    
    oscillation_cost = 0.0
    
    # Analyze each axis separately
    for axis in range(3):
        pos_axis = positions[:, axis]
        
        # Calculate second derivative (acceleration proxy)
        if len(pos_axis) >= 3:
            acceleration = np.gradient(np.gradient(pos_axis))
            
            # Count sign changes in acceleration (indicating oscillations)
            sign_changes = np.sum(np.diff(np.sign(acceleration)) != 0)
            
            # Normalize by trajectory length
            dt = time[1] - time[0] if len(time) > 1 else 0.01
            trajectory_duration = time[-1] - time[0]
            
            if trajectory_duration > min_period:
                oscillation_frequency = sign_changes / (2 * trajectory_duration)
                oscillation_cost += oscillation_frequency * np.std(pos_axis)
    
    return oscillation_cost

def calculate_overshoot(actual_pos: np.ndarray, desired_pos: np.ndarray) -> float:
    """
    Calculate overshoot penalty for trajectory tracking.
    
    Args:
        actual_pos: Actual positions [N, 3]
        desired_pos: Desired positions [N, 3]
    
    Returns:
        Overshoot cost
    """
    if len(actual_pos) < 2:
        return 0.0
    
    # Find trajectory segments where desired position changes significantly
    desired_changes = np.linalg.norm(np.diff(desired_pos, axis=0), axis=1)
    significant_changes = desired_changes > 0.1  # 10cm threshold
    
    if not np.any(significant_changes):
        return 0.0  # No significant trajectory changes
    
    overshoot_cost = 0.0
    change_indices = np.where(significant_changes)[0]
    
    for change_idx in change_indices:
        if change_idx + 10 < len(actual_pos):  # Ensure enough data after change
            # Look at response after trajectory change
            start_idx = change_idx
            end_idx = min(change_idx + 20, len(actual_pos))
            
            response_segment = actual_pos[start_idx:end_idx]
            desired_segment = desired_pos[start_idx:end_idx]
            
            # Calculate maximum deviation during response
            deviations = np.linalg.norm(response_segment - desired_segment, axis=1)
            max_deviation = np.max(deviations)
            final_deviation = deviations[-1]
            
            # Overshoot is deviation beyond final steady-state value
            if final_deviation > 0:
                overshoot = max(0, max_deviation - final_deviation * 1.5)
                overshoot_cost += overshoot
    
    return overshoot_cost

def trajectory_specific_cost(trajectory_data: Dict, trajectory_type: str) -> float:
    """
    Calculate cost specific to trajectory type.
    
    Args:
        trajectory_data: Dictionary containing simulation results
        trajectory_type: Type of trajectory ('hover', 'step', 'waypoint', 'smooth')
    
    Returns:
        Trajectory-specific cost
    """
    base_cost, _ = multi_objective_cost(trajectory_data)
    
    if trajectory_type == 'hover':
        # For hover, penalize any movement
        actual_pos = np.array(trajectory_data['actual_positions'])
        if len(actual_pos) > 10:
            position_variance = np.var(actual_pos, axis=0)
            hover_penalty = np.sum(position_variance) * 100
            return base_cost + hover_penalty
    
    elif trajectory_type == 'step':
        # For step response, focus on settling time and overshoot
        weights = {
            'tracking_weight': 1.0,
            'stability_weight': 1.2,  # Higher weight on stability
            'efficiency_weight': 0.2
        }
        step_cost, _ = multi_objective_cost(trajectory_data, weights)
        return step_cost
    
    elif trajectory_type == 'waypoint':
        # For waypoint following, balance tracking and efficiency
        weights = {
            'tracking_weight': 1.2,
            'stability_weight': 0.8,
            'efficiency_weight': 0.4
        }
        waypoint_cost, _ = multi_objective_cost(trajectory_data, weights)
        return waypoint_cost
    
    elif trajectory_type == 'smooth':
        # For smooth trajectories, prioritize smoothness and tracking
        weights = {
            'tracking_weight': 1.0,
            'stability_weight': 0.6,
            'efficiency_weight': 0.8  # Higher weight on smooth control
        }
        smooth_cost, _ = multi_objective_cost(trajectory_data, weights)
        return smooth_cost
    
    return base_cost

def robustness_cost(parameter_set: Dict, drone_configs: List[Dict], 
                   trajectory_types: List[str]) -> float:
    """
    Evaluate parameter robustness across multiple configurations and trajectories.
    
    This is a placeholder for robustness evaluation that would run multiple
    simulations with the same parameters across different conditions.
    
    Args:
        parameter_set: Controller parameters to evaluate
        drone_configs: List of drone configurations to test
        trajectory_types: List of trajectory types to test
    
    Returns:
        Robustness cost (lower means more robust)
    """
    # This would typically run multiple simulations and return variance in performance
    # For now, return 0 as a placeholder
    return 0.0