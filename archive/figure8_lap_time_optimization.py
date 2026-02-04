#!/usr/bin/env python3
"""
Figure-8 lap time optimization using CMA-ES.

This example optimizes controller gains/limits and trajectory parameters 
to minimize the time to complete figure-8 laps while staying within 0.5m 
of the intended trajectory.

Key features:
- Intended track: Ideal figure-8 trajectory from parameters
- Taken track: Actual flight path from simulation  
- Cost: Average lap completion time (penalty if >0.5m from intended track)
- Optimizes: Controller gains + trajectory parameters from pos_figure8()
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import cmaes
from typing import Dict, List, Tuple
import multiprocessing as mp
from functools import partial

# Import simulation components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'Simulation'))

from run_3D_simulation_configurable import create_drone_from_config, quad_sim
from generalized_ctrl import GeneralizedControl
from trajectory import Trajectory
from utils.windModel import Wind
from drone_simulator import ConfigurableQuadcopter
from utils.animation import sameAxisAnimation
import utils

def generate_intended_figure8_track(period=6.0, amplitude_x=3.0, amplitude_y=2.0, 
                                   center_z=-1.0, phase_offset=0.0, duration=30.0, dt=0.005):
    """
    Generate the intended figure-8 track based on trajectory parameters.
    
    Args:
        period: Total time for one complete figure-8 (seconds)
        amplitude_x: Half-width of figure-8 (meters)
        amplitude_y: Half-height of figure-8 (meters)
        center_z: Height of figure-8 (meters, negative for NED)
        phase_offset: Phase offset to start at different point on track
        duration: Total duration to generate (seconds)
        dt: Time step (seconds)
    
    Returns:
        Dictionary with 'time', 'positions', 'velocities'
    """
    times = np.arange(0, duration, dt)
    positions = []
    velocities = []
    
    for t in times:
        # Figure-8 parametric equations (Lissajous curve)
        omega = 2 * np.pi / period  # Angular frequency
        
        # Position
        x = amplitude_x * np.sin(omega * t + phase_offset)
        y = amplitude_y * np.sin(2 * (omega * t + phase_offset))
        z = center_z
        
        # Velocity (first derivative)
        vx = amplitude_x * omega * np.cos(omega * t + phase_offset)
        vy = amplitude_y * 2 * omega * np.cos(2 * (omega * t + phase_offset))
        vz = 0.0
        
        positions.append([x, y, z])
        velocities.append([vx, vy, vz])
    
    return {
        'time': times,
        'positions': np.array(positions),
        'velocities': np.array(velocities)
    }


def calculate_distance_to_intended_track(position, intended_track):
    """
    Calculate distance from current position to intended track at given time.
    
    Args:
        position: [x, y, z] current position
        intended_track: Dictionary with intended trajectory data
    
    Returns:
        Distance to intended track (meters)
    """
    # Calculate distance from current position to all points in intended track
    # axis=1 calculates distance for each 3D point (x,y,z differences)
    distances = np.linalg.norm(np.array(position) - intended_track['positions'], axis=1)
    return np.min(distances)


def figure8_lap_time_cost(taken_track_data: Dict, intended_track: Dict, 
                         max_track_distance: float = 0.5) -> float:
    """
    Calculate cost based on lap completion time with track adherence constraint.
    
    Args:
        taken_track_data: Actual flight data from simulation
        intended_track: Ideal figure-8 reference track
        max_track_distance: Maximum allowed distance from intended track (m)
    
    Returns:
        Cost value: lap time in seconds, or penalty if constraints violated
    """
    if not taken_track_data or 'actual_positions' not in taken_track_data:
        print("Error: No valid track data provided.")
        return 10.0  # High penalty for failed simulation
    
    taken_positions = np.array(taken_track_data['actual_positions'])
    taken_times = np.array(taken_track_data['time'])
    
    # Check track adherence throughout the flight
    track_violations = 0
    start_pos = intended_track['positions'][0]
    pass_through_start_latch = False
    laps = 0
    time_at_starts = []

    for i, pos in enumerate(taken_positions):        
        # Calculate distance to intended track  
        track_distance = calculate_distance_to_intended_track(pos, intended_track)
        distance_to_start = np.linalg.norm(pos - start_pos)

        # Use larger distance thresholds for lap detection (not track adherence)
        lap_detection_threshold = 0.5  # Distance to consider "at start"
        lap_reset_threshold = 1.5      # Distance to consider "away from start"
        
        if distance_to_start < lap_detection_threshold and not pass_through_start_latch:
            pass_through_start_latch = True
            laps += 1  # Count passing through start area as a lap
            time_at_starts.append(taken_times[i])

        elif distance_to_start >= lap_reset_threshold and pass_through_start_latch:
            pass_through_start_latch = False

        if track_distance > max_track_distance:
            track_violations += 1
            return 10
    
    lap_times = np.diff(time_at_starts)
    avr_lap_times = np.mean(lap_times) if len(time_at_starts) > 1 else np.inf

    if avr_lap_times == np.inf:
        return 10.0
    
    return avr_lap_times


def run_figure8_lap_simulation(controller_params: Dict, trajectory_params: Dict, 
                              duration: float = 25.0, early_termination: bool = True, 
                              max_track_distance: float = 0.5) -> Dict:
    """
    Run figure-8 simulation with given controller and trajectory parameters.
    
    Args:
        controller_params: Controller gains and limits
        trajectory_params: Figure-8 trajectory parameters
        duration: Maximum simulation duration
        early_termination: Stop simulation early if track violations occur
        max_track_distance: Maximum allowed distance from intended track
    
    Returns:
        Dictionary with simulation results
    """
    # Create drone
    # drone_config = {"type": "matched"}
    # quad = create_drone_from_config(drone_config, Ti=0)
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    quad = ConfigurableQuadcopter(Ti=0, propellers=propellers)

    # Create controller with custom parameters
    ctrl = GeneralizedControl(quad, yawType=3)  # yaw follows velocity direction
    
    # Apply controller parameters
    for param_name, param_value in controller_params.items():
        if hasattr(ctrl, param_name):
            if isinstance(param_value, (list, tuple, np.ndarray)):
                setattr(ctrl, param_name, np.array(param_value))
            else:
                setattr(ctrl, param_name, param_value)
    
    # Create trajectory with figure-8 (xyzType=14)
    traj = Trajectory(quad, "xyz_pos", [14, 3, 1.0])
    
    # Custom figure-8 function with optimized parameters
    def custom_pos_figure8():
        """Custom figure-8 with optimized parameters."""
        period = trajectory_params.get('period', 6.0)
        amplitude_x = trajectory_params.get('amplitude_x', 3.0)
        amplitude_y = trajectory_params.get('amplitude_y', 2.0)
        center_z = trajectory_params.get('center_z', -1.0)
        phase_offset = trajectory_params.get('phase_offset', 0.0)
        ramp_time = trajectory_params.get('ramp_time', 3.0)
        
        traj.t_idx = 0
        
        # Mark as initialized (position already set in main function)
        if not hasattr(traj, '_figure8_initialized') or not traj._figure8_initialized:
            traj._figure8_initialized = True
        
        # Get current simulation time from quad_sim time parameter
        current_time = getattr(traj, '_sim_time', 0)
        
        # Sigmoid speed scaling for smooth startup
        if current_time < ramp_time:
            speed_scale = 1 / (1 + np.exp(-5 * (2 * current_time / ramp_time - 1)))
        else:
            speed_scale = 1.0
        
        # Figure-8 parametric equations
        omega = 2 * np.pi / period * speed_scale
        
        # Position
        traj.desPos[0] = amplitude_x * np.sin(omega * current_time + phase_offset)
        traj.desPos[1] = amplitude_y * np.sin(2 * (omega * current_time + phase_offset))
        traj.desPos[2] = center_z
        
        # Velocity
        traj.desVel[0] = amplitude_x * omega * np.cos(omega * current_time + phase_offset)
        traj.desVel[1] = amplitude_y * 2 * omega * np.cos(2 * (omega * current_time + phase_offset))
        traj.desVel[2] = 0.0
        
        # Acceleration
        traj.desAcc[0] = -amplitude_x * omega**2 * np.sin(omega * current_time + phase_offset)
        traj.desAcc[1] = -amplitude_y * 4 * omega**2 * np.sin(2 * (omega * current_time + phase_offset))
        traj.desAcc[2] = 0.0
    
    # Directly override trajectory parameters by modifying the trajectory module
    # This is a simpler approach than monkey-patching methods
    def override_trajectory_figure8_params():
        """Override the hardcoded parameters in trajectory.py pos_figure8 function"""
        # We'll inject our parameters into the trajectory module's namespace
        import trajectory
        
        # Store original values
        if not hasattr(trajectory, '_original_figure8_params'):
            trajectory._original_figure8_params = {}
        
        # Override the hardcoded values by monkey-patching at module level
        trajectory._custom_figure8_params = {
            'period': trajectory_params.get('period', 6.0),
            'amplitude_x': trajectory_params.get('amplitude_x', 3.0),
            'amplitude_y': trajectory_params.get('amplitude_y', 2.0),
            'center_z': trajectory_params.get('center_z', -1.0),
            'phase_offset': trajectory_params.get('phase_offset', 0.0),
            'ramp_time': trajectory_params.get('ramp_time', 3.0)
        }
    
    override_trajectory_figure8_params()
    
    # Initialize figure-8 state
    traj._figure8_initialized = False
    traj._sim_time = 0
    
    # Set initial drone position to match figure-8 starting point
    initial_x = trajectory_params.get('amplitude_x', 3.0) * np.sin(trajectory_params.get('phase_offset', 0.0))
    initial_y = trajectory_params.get('amplitude_y', 2.0) * np.sin(2 * trajectory_params.get('phase_offset', 0.0))
    initial_z = trajectory_params.get('center_z', -1.0)
    quad.drone_sim.state[0:3] = [initial_x, initial_y, initial_z]
    quad._update_state_variables()
    
    # Wind model
    wind = Wind('None', 2.0, 90, -15)
    
    # Simulation parameters
    Ts = 0.005
    steps = int(duration / Ts)
    
    # Initialize result storage
    simulation_data = {
        'time': [],
        'actual_positions': [],
        'desired_positions': [],
        'actual_velocities': [],
        'desired_velocities': [],
        'control_commands': []
    }
    
    # Initialize trajectory and controller properly (like run_3D_simulation_configurable.py)
    sDes = traj.desiredState(0, Ts, quad)        
    
    # Force correct position since trajectory.py might override it
    if abs(quad.pos[2] - initial_z) > 0.1:  # If Z position was changed
        quad.drone_sim.state[0:3] = [initial_x, initial_y, initial_z]
        quad._update_state_variables()
    
    ctrl.controller(traj, quad, sDes, Ts)

    intended_track = generate_intended_figure8_track(
        period=trajectory_params.get('period', 6.0),
        amplitude_x=trajectory_params.get('amplitude_x', 3.0),
        amplitude_y=trajectory_params.get('amplitude_y', 2.0),
        center_z=trajectory_params.get('center_z', -1.0),
        phase_offset=trajectory_params.get('phase_offset', 0.0),
        duration=duration,  # Generate track up to current time + buffer
        dt=0.005
    )
    # Run simulation
    t = 0
    for step in range(steps):
        # Update trajectory time
        traj._sim_time = t
        
        # Generate custom figure-8 trajectory
        custom_pos_figure8()
        
        # Run simulation step
        t = quad_sim(t, Ts, quad, ctrl, wind, traj)
        
        # Store data
        simulation_data['time'].append(t)
        simulation_data['actual_positions'].append(quad.pos.copy())
        simulation_data['desired_positions'].append(traj.desPos.copy())
        simulation_data['actual_velocities'].append(quad.vel.copy())
        simulation_data['desired_velocities'].append(traj.desVel.copy())
        simulation_data['control_commands'].append(ctrl.w_cmd.copy())
        
        # Safety checks
        if not np.all(np.isfinite(quad.pos)) or np.linalg.norm(quad.pos) > 20:
            break
        if not np.all(np.isfinite(ctrl.w_cmd)) or np.any(ctrl.w_cmd < 0) or np.any(ctrl.w_cmd > 3000):
            break
            
        # Early termination on track violations (for optimization efficiency)
        if early_termination and step > 100:  # Allow some initial settling time
            # Generate intended track for distance calculation
            
            # Use same distance calculation as cost function
            track_distance = calculate_distance_to_intended_track(quad.pos, intended_track)
            
            if track_distance > max_track_distance:
                # Early termination due to track violation
                break
        
    
    return simulation_data


def evaluate_parameter_set(x, param_names, generation_num, individual_num):
    """
    Evaluate a single parameter set for parallel processing.
    
    Args:
        x: Parameter vector
        param_names: List of parameter names
        generation_num: Current generation number  
        individual_num: Individual number within generation
        
    Returns:
        Tuple of (cost, individual_num)
    """
    try:
        # Convert parameter vector to dictionaries
        controller_params = {
            'pos_P_gain': np.array([x[0], x[1], x[2]]),
            'vel_P_gain': np.array([x[3], x[4], x[5]]),
            'vel_D_gain': np.array([x[6], x[7], x[8]]),
            'vel_I_gain': np.array([x[9], x[10], x[11]]),
            'att_P_gain': np.array([x[12], x[13], x[14]]),
            'rate_P_gain': np.array([x[15], x[16], x[17]]),
            'rate_D_gain': np.array([x[18], x[19], x[20]]),
            'vel_max': np.array([x[21], x[22], x[23]]),
            'vel_max_all': x[24],
            'tilt_max': np.radians(x[25]),  # Convert degrees to radians
            'rate_max': np.array([np.radians(x[26]), np.radians(x[27]), np.radians(x[28])])
        }
        
        trajectory_params = {
            'period': x[29],
            'amplitude_x': x[30], 
            'amplitude_y': x[31],
            'ramp_time': x[32],
            'center_z': -1.5,  # Fixed height
            'phase_offset': 0.0  # Fixed phase
        }
        
        # Run simulation
        simulation_data = run_figure8_lap_simulation(
            controller_params, trajectory_params, duration=20.0)
        
        # Generate intended track with current trajectory parameters
        intended_track = generate_intended_figure8_track(
            period=trajectory_params['period'],
            amplitude_x=trajectory_params['amplitude_x'],
            amplitude_y=trajectory_params['amplitude_y'],
            center_z=trajectory_params['center_z'],
            phase_offset=trajectory_params['phase_offset'],
            duration=20.0
        )
        
        cost = figure8_lap_time_cost(simulation_data, intended_track)
        return (cost, individual_num, controller_params, trajectory_params)
        
    except Exception as e:
        print(f"    Gen {generation_num}, Individual {individual_num+1} failed: {str(e)[:40]}...")
        return (10.0, individual_num, None, None)


def optimize_figure8_lap_time():
    """
    Main optimization function using CMA-ES to minimize figure-8 lap completion time.
    """
    print("="*70)
    print("FIGURE-8 LAP TIME OPTIMIZATION")
    print("="*70)
    print("Objective: Minimize time to complete figure-8 lap")
    print("Constraint: Stay within 0.5m of intended trajectory")
    print("Optimizing: Controller gains/limits + trajectory parameters")
    print("-"*70)
    
    # Define parameters to optimize (comprehensive set from generalized_ctrl.py)
    param_names = [
        # Position gains
        'pos_P_gain_x', 'pos_P_gain_y', 'pos_P_gain_z',
        # Velocity gains
        'vel_P_gain_x', 'vel_P_gain_y', 'vel_P_gain_z', 
        'vel_D_gain_x', 'vel_D_gain_y', 'vel_D_gain_z',
        'vel_I_gain_x', 'vel_I_gain_y', 'vel_I_gain_z',
        # Attitude gains
        'att_P_gain_roll', 'att_P_gain_pitch', 'att_P_gain_yaw',
        # Rate gains
        'rate_P_gain_x', 'rate_P_gain_y', 'rate_P_gain_z',
        'rate_D_gain_x', 'rate_D_gain_y', 'rate_D_gain_z',
        # Velocity limits
        'vel_max_x', 'vel_max_y', 'vel_max_z', 'vel_max_all',
        # Angle and rate limits
        'tilt_max', 'rate_max_x', 'rate_max_y', 'rate_max_z',
        
        # Trajectory parameters (from pos_figure8)
        'period', 'amplitude_x', 'amplitude_y', 'ramp_time'
    ]
    
    # Parameter bounds [min, max] based on clip bounds in generalized_ctrl.py
    param_bounds = [
        # Position gains (0.1 to 20.0 from clip bounds)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Velocity P gains (0.5 to 50.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Velocity D gains (0.01 to 5.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Velocity I gains (0.1 to 20.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Attitude gains (0.5 to 50.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Rate P gains (0.1 to 10.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Rate D gains (0.001 to 1.0)
        (0.001, 10.0), (0.001, 10.0), (0.001, 10.0),
        # Velocity limits (0.5 to 50.0)
        (0.01, 50.0), (0.01, 50.0), (0.01, 50.0), (0.01, 50.0),
        # Tilt limit (10° to 70°) and rate limits (50°/s to 800°/s)
        (0.0, 800.0), (0.0, 800.0), (0.0, 800.0), (0.0, 800.0),
        
        # Trajectory parameters
        (1.0, 6.0),    # period: faster figure-8 for quicker laps
        (1.0, 5.0),    # amplitude_x: reasonable size
        (1.0, 5.0),    # amplitude_y: reasonable size
        (0.1, 5.0)     # ramp_time: extended range for startup
    ]
    
    # Initial parameter values (using defaults from generalized_ctrl.py __init__)
    initial_params = np.array([
        # Position gains (defaults: [1.0, 1.0, 1.0])
        1.0, 1.0, 1.0,
        # Velocity P gains (defaults: [5.0, 5.0, 4.0])
        5.0, 5.0, 4.0,
        # Velocity D gains (defaults: [0.5, 0.5, 0.5])
        0.5, 0.5, 0.5,
        # Velocity I gains (defaults: [5.0, 5.0, 5.0])
        5.0, 5.0, 5.0,
        # Attitude gains (defaults: [8.0, 8.0, 1.5])
        8.0, 8.0, 1.5,
        # Rate P gains (defaults: [1.5, 1.5, 1.0])
        1.5, 1.5, 1.0,
        # Rate D gains (defaults: [0.04, 0.04, 0.1])
        0.04, 0.04, 0.1,
        # Velocity limits (defaults: [5.0, 5.0, 5.0], vel_max_all=5.0)
        5.0, 5.0, 5.0, 5.0,
        # Tilt limit (default: 50°) and rate limits (defaults: [200°, 200°, 150°])
        50.0, 200.0, 200.0, 150.0,
        
        # Trajectory parameters (reasonable defaults)
        6.0,    # period
        3.0,    # amplitude_x
        2.0,    # amplitude_y  
        3.0     # ramp_time
    ])
    
    print(f"Optimizing {len(param_names)} parameters:")
    for i, name in enumerate(param_names):
        print(f"  {name}: [{param_bounds[i][0]:.1f}, {param_bounds[i][1]:.1f}] (init: {initial_params[i]:.1f})")
    print("-"*70)
    
    # Initialize CMA-ES optimizer
    optimizer = cmaes.CMA(
        mean=initial_params,
        sigma=0.2,  # 20% initial step size (reduced for more parameters)
        bounds=np.array(param_bounds),
        population_size=40,  # Larger population for higher dimensional space
        seed=42
    )
    
    # Determine number of processes to use
    num_processes = min(mp.cpu_count(), optimizer.population_size)
    print(f"CMA-ES Configuration:")
    print(f"  Population size: {optimizer.population_size}")
    print(f"  Parallel processes: {num_processes}")
    print(f"  Initial sigma: 0.2")
    print(f"  Max generations: 1000")
    print("-"*70)
    
    # Optimization tracking
    best_cost = float('inf')
    best_params = None
    cost_history = []
    successful_laps = []
    
    # Optimization loop
    for generation in range(1000):
        generation_start_time = time.time()
        solutions = []
        costs = []
        
        # Generate population
        for i in range(optimizer.population_size):
            x = optimizer.ask()
            solutions.append(x)
        
        # Evaluate population in parallel
        with mp.Pool(processes=num_processes) as pool:
            # Map evaluations to processes with all arguments
            eval_args = [(solutions[i], param_names, generation + 1, i) for i in range(len(solutions))]
            results = pool.starmap(evaluate_parameter_set, eval_args)
        
        # Process results and sort by individual number
        results.sort(key=lambda x: x[1])  # Sort by individual_num
        costs = []
        
        for cost, individual_num, controller_params, trajectory_params in results:
            costs.append(cost)
            
            # Track successful laps (cost < 10 means completed lap)
            if cost < 9.99:
                successful_laps.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    if controller_params and trajectory_params:
                        best_params = {**controller_params, **trajectory_params}
        
        # Update CMA-ES
        optimizer.tell([(x, cost) for x, cost in zip(solutions, costs)])
        cost_history.extend(costs)
        
        generation_time = time.time() - generation_start_time
        min_cost = min(costs)
        avg_cost = np.mean(costs)
        successful_runs = sum(1 for c in costs if c < 9.99)
        
        print(f"G:{generation+1} min={min_cost:.2f}s, avg={avg_cost:.1f}, success={successful_runs}/{len(costs)}, time={generation_time:.1f}s")
        
        # Check convergence
        if optimizer.should_stop():
            print("  CMA-ES convergence criteria met.")
            reason = optimizer.stop_reason()
            print(f"  Stopping reason: {reason}")
            break
    
    print("="*70)
    print("OPTIMIZATION COMPLETED")
    print("="*70)
    
    if best_cost < 10:
        print(f"Best lap time: {best_cost:.2f} seconds")
    else:
        print(f"Best cost: {best_cost:.1f} (no successful laps)")
    
    print(f"Total successful laps: {len(successful_laps)}")
    print(f"Total evaluations: {len(cost_history)}")
    
    if best_params:
        print("\nBest Parameters:")
        print("Controller:")
        for key, value in best_params.items():
            if key in ['pos_P_gain', 'vel_P_gain', 'vel_D_gain', 'vel_I_gain', 'att_P_gain', 'rate_P_gain', 'rate_D_gain', 'vel_max', 'rate_max']:
                if isinstance(value, np.ndarray):
                    print(f"  {key}: [{', '.join(f'{v:.3f}' for v in value)}]")
            elif key in ['vel_max_all']:
                print(f"  {key}: {value:.3f}")
            elif key in ['tilt_max']:
                print(f"  {key}: {np.degrees(value):.1f}°")
            elif key in ['period', 'amplitude_x', 'amplitude_y', 'ramp_time', 'center_z']:
                print(f"  {key}: {value:.2f}")
    
    # Run final verification
    if best_params and best_cost < 10:
        print("\nRunning final verification...")
        controller_keys = ['pos_P_gain', 'vel_P_gain', 'vel_D_gain', 'vel_I_gain', 'att_P_gain', 
                          'rate_P_gain', 'rate_D_gain', 'vel_max', 'vel_max_all', 'tilt_max', 'rate_max']
        trajectory_keys = ['period', 'amplitude_x', 'amplitude_y', 'ramp_time', 'center_z', 'phase_offset']
        
        final_simulation = run_figure8_lap_simulation(
            {k: v for k, v in best_params.items() if k in controller_keys},
            {k: v for k, v in best_params.items() if k in trajectory_keys},
            duration=25.0
        )
        
        final_intended = generate_intended_figure8_track(
            period=best_params['period'],
            amplitude_x=best_params['amplitude_x'],
            amplitude_y=best_params['amplitude_y'], 
            center_z=best_params['center_z'],
            duration=25.0
        )
        
        final_cost = figure8_lap_time_cost(final_simulation, final_intended)
        print(f"Final verification: {final_cost:.2f}s")
        
        # Plot results
        plot_lap_results(final_simulation, final_intended, cost_history)
    
    return best_params, best_cost


def plot_lap_results(simulation_data: Dict, intended_track: Dict, cost_history: List):
    """Plot lap optimization results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: 3D trajectory comparison
    actual_pos = np.array(simulation_data['actual_positions'])
    intended_pos = intended_track['positions']
    times = np.array(simulation_data['time'])
    
    ax1.plot(intended_pos[:, 0], intended_pos[:, 1], 'r-', linewidth=3, 
             label='Intended Track', alpha=0.8)
    ax1.plot(actual_pos[:, 0], actual_pos[:, 1], 'b-', linewidth=2, 
             label='Taken Track', alpha=0.7)
    ax1.plot(actual_pos[0, 0], actual_pos[0, 1], 'go', markersize=10, label='Start')
    ax1.plot(actual_pos[-1, 0], actual_pos[-1, 1], 'ro', markersize=10, label='Finish')
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Figure-8 Lap: Intended vs Taken Track')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Plot 2: Distance from intended track over time
    track_distances = []
    for i, pos in enumerate(actual_pos):
        distance = np.min(np.linalg.norm(pos - intended_pos, axis=1))
        track_distances.append(distance)
    
    ax2.plot(times[:len(track_distances)], track_distances, 'b-', linewidth=1)
    ax2.axhline(y=0.5, color='r', linestyle='--', linewidth=2, label='Max Distance (0.5m)')
    ax2.fill_between(times[:len(track_distances)], 0, track_distances, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance from Intended Track (m)')
    ax2.set_title('Track Adherence')
    ax2.legend()
    ax2.grid(True)
    
    # Plot 3: Speed profile
    actual_vel = np.array(simulation_data['actual_velocities'])
    intended_vel = intended_track['velocities']
    
    actual_speed = np.linalg.norm(actual_vel, axis=1)
    intended_speed = np.linalg.norm(intended_vel[:len(times)], axis=1)
    
    ax3.plot(times[:len(intended_speed)], intended_speed, 'r-', linewidth=2, 
             label='Intended Speed', alpha=0.8)
    ax3.plot(times[:len(actual_speed)], actual_speed, 'b-', linewidth=1, 
             label='Actual Speed', alpha=0.7)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Speed (m/s)')
    ax3.set_title('Speed Profile Comparison')
    ax3.legend()
    ax3.grid(True)
    
    # Plot 4: Optimization progress (focus on successful laps)
    if cost_history:
        successful_costs = [c for c in cost_history if c < 10]
        all_costs = cost_history.copy()
        
        ax4.plot(all_costs, 'o', markersize=3, alpha=0.3, color='gray', label='All evaluations')
        if successful_costs:
            # Find indices of successful runs
            success_indices = [i for i, c in enumerate(all_costs) if c < 10]
            ax4.plot(success_indices, successful_costs, 'go', markersize=4, label='Successful laps')
            
            # Show best lap time
            best_time = min(successful_costs)
            best_idx = all_costs.index(best_time)
            ax4.plot(best_idx, best_time, 'ro', markersize=8, label=f'Best: {best_time:.2f}s')
        
        ax4.set_xlabel('Evaluation')
        ax4.set_ylabel('Lap Time (s)')
        ax4.set_title('Optimization Progress')
        ax4.legend()
        ax4.grid(True)
        ax4.set_ylim(bottom=0, top=min(50, np.percentile(all_costs, 90)))
    
    plt.tight_layout()
    plt.show()


def debug_test_figure8_trajectory():
    """
    Debug test function to verify figure-8 trajectory with default drone and parameters.
    This creates an animation to visualize the trajectory.
    """
    print("="*60)
    print("DEBUG TEST: Figure-8 Trajectory with Default Parameters")
    print("="*60)
    
    # Default controller parameters (from GeneralizedControl __init__)
    controller_params = {
        'pos_P_gain': np.array([1.0, 1.0, 1.0]),
        'vel_P_gain': np.array([5.0, 5.0, 4.0]),
        'vel_D_gain': np.array([0.5, 0.5, 0.5]),
        'vel_I_gain': np.array([5.0, 5.0, 5.0]),
        'att_P_gain': np.array([8.0, 8.0, 1.5]),
        'rate_P_gain': np.array([1.5, 1.5, 1.0]),
        'rate_D_gain': np.array([0.04, 0.04, 0.1]),
        'vel_max': np.array([5.0, 5.0, 5.0]),
        'vel_max_all': 5.0,
        'tilt_max': np.radians(50.0),
        'rate_max': np.array([np.radians(200.0), np.radians(200.0), np.radians(150.0)])
    }
    
    # Default trajectory parameters
    trajectory_params = {
        'period': 8.0,
        'amplitude_x': 3.0,
        'amplitude_y': 2.0,
        'center_z': -1.5,
        'phase_offset': 0.0,
        'ramp_time': 3.0
    }
    
    print("Controller Parameters:")
    for key, value in controller_params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: [{', '.join(f'{v:.3f}' for v in value)}]")
        else:
            print(f"  {key}: {value:.3f}")
    
    print("\nTrajectory Parameters:")
    for key, value in trajectory_params.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nRunning simulation...")
    
    # Run simulation with default parameters
    simulation_data = run_figure8_lap_simulation(
        controller_params, trajectory_params, duration=15.0)
    
    # Convert simulation data to format expected by animation
    t_all = np.array(simulation_data['time'])
    pos_all = np.array(simulation_data['actual_positions'])
    sDes_traj_all = np.array(simulation_data['desired_positions'])
    
    # Create dummy quaternions (assume level flight for visualization)
    quat_all = np.zeros((len(t_all), 4))
    quat_all[:, 3] = 1.0  # w component = 1 for identity quaternion
    
    # Create waypoints for visualization using actual desired trajectory points
    # Sample key points from the desired trajectory to ensure correct coordinates
    sDes_array = np.array(sDes_traj_all)
    
    # Find indices for key points in the figure-8 (approximate)
    waypoint_indices = [
        0,                                   # Start point
        len(sDes_array) // 8,               # 1/8 through
        len(sDes_array) // 4,               # 1/4 through  
        3 * len(sDes_array) // 8,           # 3/8 through
        len(sDes_array) // 2,               # Halfway
        5 * len(sDes_array) // 8,           # 5/8 through
        3 * len(sDes_array) // 4,           # 3/4 through
        7 * len(sDes_array) // 8            # 7/8 through
    ]
    
    waypoints = np.array([sDes_array[i] for i in waypoint_indices if i < len(sDes_array)])
    
    # Create dummy drone parameters for animation
    class DummyParams:
        def __init__(self):
            self.dxm = 0.16
            self.dym = 0.16  
            self.dzm = 0.0
    
    params = DummyParams()
    params = {"dxm": 0.16, "dym": 0.16, "dzm": 0.0}
    
    print(f"Simulation completed: {len(t_all)} time steps over {t_all[-1]:.2f}s")
    
    # Generate intended track for comparison
    intended_track = generate_intended_figure8_track(
        period=trajectory_params['period'],
        amplitude_x=trajectory_params['amplitude_x'],
        amplitude_y=trajectory_params['amplitude_y'],
        center_z=trajectory_params['center_z'],
        phase_offset=trajectory_params['phase_offset'],
        duration=15.0
    )
    
    # Calculate cost for verification
    cost = figure8_lap_time_cost(simulation_data, intended_track, max_track_distance=0.5)
    print(f"Lap time cost: {cost:.2f}s")
    
    if cost < 10:
        print("✓ Simulation successful - lap completed!")
    else:
        print("✗ Simulation failed - no valid lap completed")
    
    print("\nCreating animation...")
    
    # Create animation using existing utility
    Ts = 0.005
    ani = sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_traj_all, 
                          Ts, params, xyzType=14, yawType=3, ifsave=False)
    
    print("Animation created. Close the plot window to continue.")
    plt.show()
    
    return simulation_data, intended_track


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('fork', force=True)
    
    import sys
    
    # Check for debug flag
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        print("Running debug test instead of optimization...")
        debug_test_figure8_trajectory()
        sys.exit(0)
    
    # Run the optimization
    best_params, best_cost = optimize_figure8_lap_time()
    
    print("\n" + "="*70)
    print("FIGURE-8 LAP TIME OPTIMIZATION COMPLETED")
    print("="*70)
    print("This example optimized:")
    print("1. Controller gains/limits from generalized_ctrl.py")
    print("2. Trajectory parameters from pos_figure8() function")
    print("3. Objective: Minimize lap completion time")
    print("4. Constraint: Stay within 0.5m of intended figure-8 trajectory")
    
    if best_cost < 10:
        print(f"\nBest lap time achieved: {best_cost:.2f} seconds")
    else:
        print(f"\nNo successful laps completed (best cost: {best_cost:.1f})")
    print("="*70)