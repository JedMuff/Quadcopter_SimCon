#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMA-ES Optimization for Lee Geometric Controller with Trajectory Tuning

This script uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
to find optimal controller gains AND trajectory parameters for the Lee
geometric controller.

CMA-ES is a state-of-the-art evolutionary algorithm that:
- Adapts the search distribution based on previous results
- Handles continuous optimization problems very well
- Works well even with noisy objectives
- Supports parallel evaluation for faster optimization

OPTIMIZATION PARAMETERS:
========================
By default, the script optimizes BOTH controller gains AND trajectory parameters:
- Controller gains (4 params): pos_P, vel_P, att_P, rate_P
- Trajectory params (4 params): period, amplitude_x, amplitude_y, ramp_time

This joint optimization finds the combination of controller and trajectory
characteristics that minimizes tracking error.

FITNESS FUNCTION:
================
The optimization uses a two-stage fitness function with an MSE threshold:

STAGE 1 - Accuracy Optimization (MSE > threshold):
- Focus on minimizing Mean Squared Error (MSE) of position tracking
- Fitness = MSE (lower is better)
- Goal: Achieve accurate trajectory following

STAGE 2 - Speed Optimization (MSE <= threshold):
- Once MSE is below threshold, optimize for faster average lap time
- Fitness = normalized_avg_lap_time + small_mse_term (lower is better)
- Goal: Complete laps in minimum average time while maintaining accuracy
- Uses average lap time across multiple laps (excluding first lap with ramp-up)
- If only one lap completed, uses that time (includes ramp-up)

- Crashed/unstable runs: Large penalty (1000.0)

This two-stage approach encourages the optimizer to:
1. First find configurations that track accurately (Stage 1)
2. Then optimize for faster lap times among accurate solutions (Stage 2)
3. Complete at least one lap without crashes

The MSE threshold can be adjusted with --mse-threshold (default: 0.05)

Additional metrics are computed for analysis but not used in fitness:
- Steady-State Error: Average error in final 25% of trajectory
- Error Growth Rate: Whether errors increase over time
- Oscillation Index: Variance in error changes (smoothness)
- Convergence Slope: Whether system is improving over time

The script runs simulations in headless mode (MPLBACKEND=Agg) to prevent
plot windows from opening and blocking the tuning process. All plots are
generated and saved automatically to the output directory.

Usage:
    python tune_lee_controller.py --max-evals 200              # Tune gains + trajectory (default)
    python tune_lee_controller.py --max-evals 500              # More evaluations for better results
    python tune_lee_controller.py --no-tune-trajectory         # Only tune gains (4 params)
    python tune_lee_controller.py --workers 1                  # Serial execution (no parallelization)
    python tune_lee_controller.py --mse-threshold 0.1          # Use higher MSE threshold before optimizing for speed

Note: CMA-ES requires the 'cma' package. Install with: pip install cma
"""

import subprocess
import numpy as np
import argparse
import re
import os
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Try to import CMA-ES
try:
    import cma
    CMA_AVAILABLE = True
except ImportError:
    CMA_AVAILABLE = False
    print("ERROR: 'cma' package not found. This script requires CMA-ES.")
    print("Install with: pip install cma")
    import sys
    sys.exit(1)

class ControllerTuner:
    """CMA-ES optimization for Lee Geometric Controller"""

    def __init__(self, sim_time=15.0, dt=0.005, output_dir="tuning_results", mse_threshold=5.0):
        self.sim_time = sim_time
        self.dt = dt
        self.output_dir = output_dir
        self.mse_threshold = mse_threshold  # MSE threshold for two-stage optimization
        self.results = []
        self.best_score = float('inf')
        self.best_gains = None

        # Create output directory
        Path(self.output_dir).mkdir(exist_ok=True)

    def run_simulation(self, pos_gain, vel_gain, att_gain, rate_gain,
                      traj_period=6.0, traj_amp_x=3.0, traj_amp_y=2.0, traj_ramp=3.0):
        """
        Run simulation with given gains and trajectory parameters, then capture output.

        Returns:
            dict with performance metrics or None if failed
        """
        gain_str = f"{pos_gain},{vel_gain},{att_gain},{rate_gain}"

        print(f"\n{'='*60}")
        print(f"Testing: pos={pos_gain:.2f}, vel={vel_gain:.2f}, "
              f"att={att_gain:.2f}, rate={rate_gain:.3f}")
        print(f"Trajectory: period={traj_period:.2f}s, amp_x={traj_amp_x:.2f}m, "
              f"amp_y={traj_amp_y:.2f}m, ramp={traj_ramp:.2f}s")
        print(f"{'='*60}")

        # Set environment to use non-interactive matplotlib backend
        # This prevents plot windows from opening and blocking the tuning
        env = os.environ.copy()
        env['MPLBACKEND'] = 'Agg'

        # Run simulation and capture output
        try:
            result = subprocess.run(
                [
                    "python", "examples/run_3D_simulation_lee_ctrl.py",
                    "--gains", gain_str,
                    "--time", str(self.sim_time),
                    "--dt", str(self.dt),
                    "--type", "matched",
                    "--traj-period", str(traj_period),
                    "--traj-amp-x", str(traj_amp_x),
                    "--traj-amp-y", str(traj_amp_y),
                    "--traj-ramp", str(traj_ramp)
                ],
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                env=env  # Pass modified environment
            )

            output = result.stdout + result.stderr

            # Parse performance metrics from output (pass ramp_time for lap duration calculation)
            metrics = self._parse_output(output, traj_ramp)

            if metrics:
                metrics['gains'] = {
                    'pos_P': pos_gain,
                    'vel_P': vel_gain,
                    'att_P': att_gain,
                    'rate_P': rate_gain
                }
                metrics['trajectory'] = {
                    'period': traj_period,
                    'amplitude_x': traj_amp_x,
                    'amplitude_y': traj_amp_y,
                    'ramp_time': traj_ramp
                }
                metrics['success'] = True
                return metrics
            else:
                return {
                    'success': False,
                    'gains': {'pos_P': pos_gain, 'vel_P': vel_gain,
                             'att_P': att_gain, 'rate_P': rate_gain},
                    'trajectory': {'period': traj_period, 'amplitude_x': traj_amp_x,
                                 'amplitude_y': traj_amp_y, 'ramp_time': traj_ramp},
                    'error': 'Failed to parse metrics'
                }

        except subprocess.TimeoutExpired:
            print("  ⚠️  Simulation timeout - likely unstable")
            return {
                'success': False,
                'gains': {'pos_P': pos_gain, 'vel_P': vel_gain,
                         'att_P': att_gain, 'rate_P': rate_gain},
                'trajectory': {'period': traj_period, 'amplitude_x': traj_amp_x,
                             'amplitude_y': traj_amp_y, 'ramp_time': traj_ramp},
                'error': 'Timeout (unstable)'
            }
        except Exception as e:
            print(f"  ⚠️  Error: {e}")
            return {
                'success': False,
                'gains': {'pos_P': pos_gain, 'vel_P': vel_gain,
                         'att_P': att_gain, 'rate_P': rate_gain},
                'trajectory': {'period': traj_period, 'amplitude_x': traj_amp_x,
                             'amplitude_y': traj_amp_y, 'ramp_time': traj_ramp},
                'error': str(e)
            }

    def _parse_output(self, output, ramp_time=3.0):
        """Parse simulation output to extract performance metrics.

        Args:
            output: Simulation output text
            ramp_time: Trajectory ramp-up time in seconds (used for lap duration calculation)
        """
        metrics = {
            'pos_errors': [],
            'vel_errors': [],
            'timestamps': [],
            'instability_count': 0,
            'completed': False,
            'early_termination': False,
            'flight_time': 0.0,
            'crash_time': None,
            'lap_completion_times': [],  # List of all lap completion times
            'lap_durations': [],  # Duration of each lap
            'avg_lap_time': None  # Average lap duration (excluding first lap)
        }

        # Extract debug information
        for line in output.split('\n'):
            # Extract timestamp and errors together for temporal analysis
            time_match = re.search(r't=([\d.]+)s', line)
            if time_match:
                current_time = float(time_match.group(1))

            # Position error
            match = re.search(r'Pos error:\s+([\d.]+)m', line)
            if match:
                pos_err = float(match.group(1))
                metrics['pos_errors'].append(pos_err)
                if time_match:
                    metrics['timestamps'].append(current_time)
                    metrics['flight_time'] = current_time  # Update flight time

            # Velocity error
            match = re.search(r'Vel error:\s+([\d.]+)m/s', line)
            if match:
                metrics['vel_errors'].append(float(match.group(1)))

            # Instability detection
            if 'INSTABILITY DETECTED' in line:
                metrics['instability_count'] += 1

            # Check if simulation completed normally
            if 'Simulated' in line and 's in' in line:
                match = re.search(r'Simulated\s+([\d.]+)s', line)
                if match:
                    metrics['flight_time'] = float(match.group(1))
                metrics['completed'] = True

            # Check for early termination
            if 'terminated early' in line or 'severe instability' in line:
                metrics['early_termination'] = True
                if time_match:
                    metrics['crash_time'] = current_time

            # Parse lap completion times
            if 'LAP' in line and 'COMPLETED' in line:
                match = re.search(r'LAP (\d+) COMPLETED at t=([\d.]+)s', line)
                if match:
                    lap_time = float(match.group(2))
                    metrics['lap_completion_times'].append(lap_time)
                    print(f"  DEBUG: Found lap completion at t={lap_time:.3f}s")

        # Calculate lap durations and average lap time
        print(f"  DEBUG: Total laps detected: {len(metrics['lap_completion_times'])}")
        if len(metrics['lap_completion_times']) > 0:
            print(f"  DEBUG: Lap completion times: {metrics['lap_completion_times']}")
            lap_times = metrics['lap_completion_times']

            # Calculate duration of each lap (difference between successive completion times)
            if len(lap_times) > 1:
                for i in range(1, len(lap_times)):
                    lap_duration = lap_times[i] - lap_times[i-1]
                    metrics['lap_durations'].append(lap_duration)

                # Average lap time uses laps 2, 3, 4, etc. (excluding first lap)
                # The first lap duration is from ramp-up end to first completion
                print(f"  DEBUG: Lap durations: {metrics['lap_durations']}")
                metrics['avg_lap_time'] = float(np.mean(metrics['lap_durations']))
                print(f"  DEBUG: Average lap time: {metrics['avg_lap_time']:.3f}s")
            else:
                # If only one lap completed, calculate duration from ramp-up end
                # lap_times[0] is absolute time, so subtract ramp_time to get lap duration
                metrics['avg_lap_time'] = lap_times[0] - ramp_time
                print(f"  DEBUG: Only one lap completed at t={lap_times[0]:.3f}s")
                print(f"  DEBUG: Lap duration (excluding ramp): {metrics['avg_lap_time']:.3f}s")
        else:
            print(f"  DEBUG: No laps completed during simulation")

        # Calculate summary statistics
        if len(metrics['pos_errors']) == 0:
            return None  # No valid data

        pos_errors = np.array(metrics['pos_errors'])
        vel_errors = np.array(metrics['vel_errors']) if len(metrics['vel_errors']) > 0 else np.zeros_like(pos_errors)

        # Basic statistics (convert to native Python types for JSON serialization)
        metrics['pos_error_mean'] = float(np.mean(pos_errors))
        metrics['pos_error_max'] = float(np.max(pos_errors))
        metrics['pos_error_std'] = float(np.std(pos_errors))
        metrics['pos_error_final'] = float(pos_errors[-1])

        metrics['vel_error_mean'] = float(np.mean(vel_errors))
        metrics['vel_error_max'] = float(np.max(vel_errors))
        metrics['vel_error_final'] = float(vel_errors[-1])

        # IMPROVED METRICS FOR STABILITY AND TRAJECTORY FOLLOWING

        # 1. Steady-state error (last 25% of trajectory) - crucial for stable flight
        steady_state_idx = int(len(pos_errors) * 0.75)
        if steady_state_idx < len(pos_errors):
            metrics['pos_error_steady_state'] = float(np.mean(pos_errors[steady_state_idx:]))
            metrics['vel_error_steady_state'] = float(np.mean(vel_errors[steady_state_idx:]))
        else:
            metrics['pos_error_steady_state'] = metrics['pos_error_mean']
            metrics['vel_error_steady_state'] = metrics['vel_error_mean']

        # 2. Error growth rate - detect diverging trajectories
        if len(pos_errors) > 10:
            # Compare first half to second half
            first_half_mean = np.mean(pos_errors[:len(pos_errors)//2])
            second_half_mean = np.mean(pos_errors[len(pos_errors)//2:])
            metrics['error_growth_rate'] = float((second_half_mean - first_half_mean) / first_half_mean if first_half_mean > 0 else 0)
        else:
            metrics['error_growth_rate'] = 0.0

        # 3. Oscillation index - detect oscillating/unstable behavior
        # Calculate rate of change of errors
        if len(pos_errors) > 1:
            error_deltas = np.diff(pos_errors)
            metrics['oscillation_index'] = float(np.std(error_deltas))
            # Count sign changes (oscillations)
            sign_changes = np.sum(np.diff(np.sign(error_deltas)) != 0)
            metrics['oscillation_count'] = int(sign_changes)
        else:
            metrics['oscillation_index'] = 0.0
            metrics['oscillation_count'] = 0

        # 4. Peak transient error (excluding initial startup)
        # Skip first 10% as startup transient
        startup_idx = max(1, int(len(pos_errors) * 0.1))
        metrics['pos_error_peak_transient'] = float(np.max(pos_errors[startup_idx:]))

        # 5. Convergence quality - is it getting better over time?
        if len(pos_errors) > 5:
            # Linear fit to see if error is decreasing
            time_indices = np.arange(len(pos_errors))
            slope = np.polyfit(time_indices, pos_errors, 1)[0]
            metrics['convergence_slope'] = float(slope)  # Negative is good (decreasing error)
        else:
            metrics['convergence_slope'] = 0.0

        # TWO-STAGE FITNESS FUNCTION (lower is better)
        # Stage 1: If MSE > threshold, optimize for accuracy (minimize MSE)
        # Stage 2: If MSE <= threshold, optimize for speed (minimize average lap time)
        # This encourages finding accurate solutions first, then faster lap completions

        if metrics['completed'] and not metrics['early_termination']:
            # Mean Squared Error of position tracking
            mse = float(np.mean(pos_errors ** 2))
            metrics['mse'] = mse  # Store MSE separately for analysis


            if mse > self.mse_threshold:
                laptime = 15
                score = laptime + mse 
            else:
                laptime = metrics['avg_lap_time'] if metrics['avg_lap_time'] is not None else self.sim_time
                print("Avg lap time:", metrics['avg_lap_time'])
                score = laptime  

            # score = laptime + mse 
            # if mse > self.mse_threshold:
            #     # Stage 1: Focus on achieving good tracking accuracy
            #     score = mse
            #     metrics['optimization_stage'] = 'accuracy'
            # else:
            #     # Stage 2: MSE is sufficient, now optimize for average lap completion time
            #     # Use avg_lap_time if available (for figure-8 trajectory), otherwise fall back to flight_time
            #     if metrics['avg_lap_time'] is not None:
            #         # Optimize for faster average lap completion
            #         # Expected lap time is approximately the trajectory period (e.g., 6 seconds)
            #         # We normalize by expected period to get a dimensionless ratio
            #         # Faster average laps = lower score
            #         time_ratio = metrics['avg_lap_time'] / self.sim_time
            #         score = self.mse_threshold * 0.9 * time_ratio + 0.001 * mse
            #         metrics['optimization_stage'] = 'speed_lap'
            #     else:
            #         # No lap time available, fall back to total flight time
            #         time_ratio = metrics['flight_time'] / self.sim_time
            #         score = self.mse_threshold * 0.9 * time_ratio + 0.001 * mse
            #         metrics['optimization_stage'] = 'speed'
        else:
            # Large penalty for crashes/instabilities or no lap completion
            # If lap wasn't completed within sim time, it gets a penalty
            score = 1000.0
            metrics['optimization_stage'] = 'failed'

        metrics['score'] = score

        # Clean up internal lists that don't need to be saved (saves space)
        del metrics['pos_errors']
        del metrics['vel_errors']
        del metrics['timestamps']

        return metrics

    def run_cmaes_tuning(self, max_evaluations=200, initial_guess=None, initial_std=None, num_workers=None,
                         tune_trajectory=True):
        """
        Run CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimization.

        CMA-ES is a state-of-the-art evolutionary algorithm that:
        - Adapts the search distribution based on previous results
        - Handles continuous optimization problems very well
        - Doesn't require a grid - samples intelligently
        - Works well even with noisy objectives

        Args:
            max_evaluations: Maximum number of evaluations (default: 200)
            initial_guess: Starting point [pos, vel, att, rate, period, amp_x, amp_y, ramp]. If None, uses baseline.
            initial_std: Initial standard deviation for search. If None, uses reasonable defaults.
            num_workers: Number of parallel workers. If None, uses CPU count / 2.
            tune_trajectory: If True, also tune trajectory parameters (8 total params). If False, only tune gains (4 params).
        """
        if not CMA_AVAILABLE:
            print("\n❌ ERROR: CMA-ES requires the 'cma' package.")
            print("Install with: pip install cma")
            return

        # Set up parallel workers
        if num_workers is None:
            num_workers = max(1, multiprocessing.cpu_count() // 2)

        print(f"\n{'='*70}")
        print(f"LEE CONTROLLER CMA-ES OPTIMIZATION (PARALLEL)")
        print(f"{'='*70}")
        print(f"Strategy: Evolutionary optimization with adaptive search distribution")
        print(f"Max evaluations: {max_evaluations}")
        print(f"Parallel workers: {num_workers}")
        print(f"Tuning trajectory: {tune_trajectory}")
        print(f"MSE threshold: {self.mse_threshold:.6f} (switch to speed optimization below this)")
        print(f"{'='*70}\n")

        # Set up initial guess and bounds
        if initial_guess is None:
            # Start from experimentally validated working region
            # Based on tuning results: pos~10, vel~9, att~4.4, rate~-0.3
            initial_guess = [10.0, 9.0, 4.4, -0.3]

            if tune_trajectory:
                # Add trajectory parameters: period, amp_x, amp_y, ramp_time
                initial_guess.extend([6.0, 3.0, 2.0, 3.0])

        if initial_std is None:
            # Standard deviations for each parameter
            initial_std = [2.0, 2.0, 0.5, 0.1]

            if tune_trajectory:
                # Add std for trajectory parameters
                initial_std.extend([1.5, 0.8, 0.6, 0.8])

        # Bounds: [min, max] for each parameter
        # Expanded based on experimental results showing higher gains work better
        bounds = [
            [5.0, 20.0],    # pos_P: expanded upward
            [5.0, 20.0],    # vel_P: expanded upward
            [2.0, 8.0],     # att_P: centered around 4.4
            [-1.0, 0.1]     # rate_P: NEGATIVE works best!
        ]

        if tune_trajectory:
            # Add bounds for trajectory parameters
            bounds.extend([
                [3.0, 12.0],    # period: 3-12 seconds
                [1.0, 5.0],     # amplitude_x: 1-5 meters
                [1.0, 4.0],     # amplitude_y: 1-4 meters
                [1.0, 5.0]      # ramp_time: 1-5 seconds
            ])

        print(f"Initial guess (gains): pos={initial_guess[0]:.2f}, vel={initial_guess[1]:.2f}, "
              f"att={initial_guess[2]:.2f}, rate={initial_guess[3]:.3f}")
        if tune_trajectory:
            print(f"Initial guess (traj): period={initial_guess[4]:.2f}, amp_x={initial_guess[5]:.2f}, "
                  f"amp_y={initial_guess[6]:.2f}, ramp={initial_guess[7]:.2f}")
        print(f"Search std: {initial_std}")
        print(f"Bounds (gains): pos=[{bounds[0][0]}, {bounds[0][1]}], vel=[{bounds[1][0]}, {bounds[1][1]}], "
              f"att=[{bounds[2][0]}, {bounds[2][1]}], rate=[{bounds[3][0]}, {bounds[3][1]}]")
        if tune_trajectory:
            print(f"Bounds (traj): period=[{bounds[4][0]}, {bounds[4][1]}], "
                  f"amp_x=[{bounds[5][0]}, {bounds[5][1]}], amp_y=[{bounds[6][0]}, {bounds[6][1]}], "
                  f"ramp=[{bounds[7][0]}, {bounds[7][1]}]")
        print()

        # Store results with metadata for reporting
        self.cmaes_results_cache = {}

        # Define objective function for CMA-ES (must return single scalar)
        def objective(x):
            """Objective function for CMA-ES. Returns MSE (lower is better)."""
            # Unpack parameters
            if tune_trajectory:
                pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp = x

                # Enforce bounds
                pos_g = np.clip(pos_g, bounds[0][0], bounds[0][1])
                vel_g = np.clip(vel_g, bounds[1][0], bounds[1][1])
                att_g = np.clip(att_g, bounds[2][0], bounds[2][1])
                rate_g = np.clip(rate_g, bounds[3][0], bounds[3][1])
                period = np.clip(period, bounds[4][0], bounds[4][1])
                amp_x = np.clip(amp_x, bounds[5][0], bounds[5][1])
                amp_y = np.clip(amp_y, bounds[6][0], bounds[6][1])
                ramp = np.clip(ramp, bounds[7][0], bounds[7][1])
            else:
                pos_g, vel_g, att_g, rate_g = x

                # Enforce bounds
                pos_g = np.clip(pos_g, bounds[0][0], bounds[0][1])
                vel_g = np.clip(vel_g, bounds[1][0], bounds[1][1])
                att_g = np.clip(att_g, bounds[2][0], bounds[2][1])
                rate_g = np.clip(rate_g, bounds[3][0], bounds[3][1])

                # Use default trajectory parameters
                period, amp_x, amp_y, ramp = 6.0, 3.0, 2.0, 3.0

            result = self.run_simulation(pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp)

            if result and result['success']:
                self.results.append(result)
                # Cache result for detailed reporting
                if tune_trajectory:
                    key = (pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp)
                else:
                    key = (pos_g, vel_g, att_g, rate_g)
                self.cmaes_results_cache[key] = result
                return result['score']
            else:
                # Return very high penalty for failed simulations
                if result:
                    self.results.append(result)
                    if tune_trajectory:
                        key = (pos_g, vel_g, att_g, rate_g, period, amp_x, amp_y, ramp)
                    else:
                        key = (pos_g, vel_g, att_g, rate_g)
                    self.cmaes_results_cache[key] = result
                return 1000.0

        # Set up CMA-ES options
        options = {
            'bounds': [list(b) for b in zip(*bounds)],  # Transpose bounds for CMA-ES format
            'maxfevals': max_evaluations,
            'verb_disp': 1,  # Display every iteration
            'verb_log': 0,   # No log file
            'tolx': 1e-4,    # Tolerance on parameter changes
            'tolfun': 1e-2,  # Tolerance on function value changes
        }

        # Run CMA-ES
        print("Starting CMA-ES optimization...\n")
        start_time = time.time()

        try:
            es = cma.CMAEvolutionStrategy(initial_guess, initial_std[0], options)

            iteration = 0
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                while not es.stop():
                    iteration += 1
                    solutions = es.ask()

                    # Parallel evaluation of solutions
                    if num_workers > 1:
                        # Submit all solutions to thread pool
                        future_to_sol = {executor.submit(objective, sol): sol for sol in solutions}

                        # Collect results and map back to original order
                        results_dict = {}
                        for future in as_completed(future_to_sol):
                            sol = future_to_sol[future]
                            try:
                                score = future.result()
                                results_dict[tuple(sol)] = score
                            except Exception as e:
                                print(f"Error evaluating solution: {e}")
                                results_dict[tuple(sol)] = 1000.0  # Penalty for errors

                        # Reorder to match solutions order
                        fitness_values = [results_dict[tuple(sol)] for sol in solutions]
                    else:
                        # Serial execution if only 1 worker
                        fitness_values = [objective(x) for x in solutions]

                    es.tell(solutions, fitness_values)

                    # Print progress with detailed metrics
                    best_idx = np.argmin(fitness_values)
                    best_solution = solutions[best_idx]
                    best_score = fitness_values[best_idx]

                    # Get detailed metrics for best in generation
                    best_key = tuple(best_solution)
                    best_gen_result = self.cmaes_results_cache.get(best_key, {})

                    # Get detailed metrics for best ever
                    best_ever_key = tuple(es.result.xbest)
                    best_ever_result = self.cmaes_results_cache.get(best_ever_key, {})

                    print(f"\n{'='*70}")
                    print(f"CMA-ES Iteration {iteration} | Evaluated: {es.result.evaluations}/{max_evaluations}")
                    print(f"{'='*70}")

                    # Best in this generation
                    flight_time_gen = best_gen_result.get('flight_time', 0)
                    completed_gen = best_gen_result.get('completed', False)
                    completion_str_gen = "✓" if completed_gen and not best_gen_result.get('early_termination', False) else "✗"

                    print(f"Best in generation: MSE={best_score:.6f} | Flight={flight_time_gen:.2f}s | {completion_str_gen}")
                    print(f"  Gains: pos={best_solution[0]:.3f}, vel={best_solution[1]:.3f}, "
                          f"att={best_solution[2]:.3f}, rate={best_solution[3]:.4f}")
                    if tune_trajectory and len(best_solution) >= 8:
                        print(f"  Trajectory: period={best_solution[4]:.2f}s, amp_x={best_solution[5]:.2f}m, "
                              f"amp_y={best_solution[6]:.2f}m, ramp={best_solution[7]:.2f}s")
                    if completed_gen:
                        pos_err = best_gen_result.get('pos_error_mean', 0)
                        print(f"  Mean position error: {pos_err:.4f}m")

                    # Best ever
                    flight_time_ever = best_ever_result.get('flight_time', 0)
                    completed_ever = best_ever_result.get('completed', False)
                    completion_str_ever = "✓" if completed_ever and not best_ever_result.get('early_termination', False) else "✗"

                    print(f"\nBest ever: MSE={es.result.fbest:.6f} | Flight={flight_time_ever:.2f}s | {completion_str_ever}")
                    print(f"  Gains: pos={es.result.xbest[0]:.3f}, vel={es.result.xbest[1]:.3f}, "
                          f"att={es.result.xbest[2]:.3f}, rate={es.result.xbest[3]:.4f}")
                    if tune_trajectory and len(es.result.xbest) >= 8:
                        print(f"  Trajectory: period={es.result.xbest[4]:.2f}s, amp_x={es.result.xbest[5]:.2f}m, "
                              f"amp_y={es.result.xbest[6]:.2f}m, ramp={es.result.xbest[7]:.2f}s")
                    if completed_ever:
                        pos_err_ever = best_ever_result.get('pos_error_mean', 0)
                        print(f"  Mean position error: {pos_err_ever:.4f}m")

                    print(f"{'='*70}")

                    # Save intermediate results
                    if iteration % 5 == 0:
                        self._save_results()

            end_time = time.time()
            elapsed = end_time - start_time

            # Print final results with detailed metrics
            best_ever_key = tuple(es.result.xbest)
            best_ever_result = self.cmaes_results_cache.get(best_ever_key, {})

            print(f"\n{'='*70}")
            print(f"CMA-ES OPTIMIZATION COMPLETE")
            print(f"{'='*70}")
            print(f"Time elapsed: {elapsed/60:.1f} minutes")
            print(f"Total evaluations: {es.result.evaluations}")
            print(f"Stop condition: {es.stop()}")

            flight_time_final = best_ever_result.get('flight_time', 0)
            completed_final = best_ever_result.get('completed', False)
            early_term = best_ever_result.get('early_termination', False)

            print(f"\n🎯 BEST SOLUTION FOUND:")
            print(f"  Mean Squared Error: {es.result.fbest:.6f}")
            print(f"  \nCONTROL GAINS:")
            print(f"    Position gain:      {es.result.xbest[0]:.3f}")
            print(f"    Velocity gain:      {es.result.xbest[1]:.3f}")
            print(f"    Attitude gain:      {es.result.xbest[2]:.3f}")
            print(f"    Rate gain:          {es.result.xbest[3]:.4f}")
            if tune_trajectory and len(es.result.xbest) >= 8:
                print(f"  \nTRAJECTORY PARAMETERS:")
                print(f"    Period:             {es.result.xbest[4]:.2f} s")
                print(f"    Amplitude X:        {es.result.xbest[5]:.2f} m")
                print(f"    Amplitude Y:        {es.result.xbest[6]:.2f} m")
                print(f"    Ramp time:          {es.result.xbest[7]:.2f} s")
            print(f"\n  PERFORMANCE:")
            print(f"    Flight time:         {flight_time_final:.2f}s")
            print(f"    Completed:           {'✓ Yes' if completed_final and not early_term else '✗ No'}")
            if completed_final:
                print(f"    Mean position error: {best_ever_result.get('pos_error_mean', 0):.4f}m")
                print(f"    RMSE (sqrt of MSE):  {np.sqrt(es.result.fbest):.4f}m")
            print(f"{'='*70}\n")

            # Save final results
            self._save_results()

            # Analyze all results
            self._analyze_results()

        except KeyboardInterrupt:
            print("\n\n⚠️  CMA-ES interrupted by user. Saving results...")
            self._save_results()
            self._analyze_results()
        except Exception as e:
            print(f"\n\n❌ ERROR during CMA-ES optimization: {e}")
            import traceback
            traceback.print_exc()
            self._save_results()

    def _save_results(self):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/tuning_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n💾 Results saved to: {filename}")

    def _analyze_results(self):
        """Analyze results and generate report."""
        # Filter successful results
        successful = [r for r in self.results if r.get('success', False)]

        if not successful:
            print("\n⚠️  No successful runs to analyze!")
            return

        print(f"\n{'='*70}")
        print(f"TUNING ANALYSIS")
        print(f"{'='*70}\n")

        # Sort by score (lower is better)
        successful.sort(key=lambda x: x['score'])

        # Top 10 configurations
        # Check if any result has trajectory parameters
        has_trajectory = any('trajectory' in r for r in successful[:10])

        if has_trajectory:
            print("🏆 TOP 10 CONFIGURATIONS (by fitness score) - WITH TRAJECTORY TUNING:\n")
            print(f"{'Rank':<6} {'Score':<12} {'MSE':<12} {'Stage':<11} {'AvgLap':<9} {'Laps':<6} {'Pos_P':<8} {'Vel_P':<8} {'Att_P':<8} {'Rate_P':<9} "
                  f"{'Period':<8} {'Amp_X':<8} {'Amp_Y':<8} {'Ramp':<7}")
            print(f"{'-'*140}")
        else:
            print("🏆 TOP 10 GAIN CONFIGURATIONS (by fitness score):\n")
            print(f"{'Rank':<6} {'Score':<12} {'MSE':<12} {'Stage':<11} {'AvgLap':<9} {'Laps':<6} {'Flight':<10} {'Pos_P':<8} {'Vel_P':<8} {'Att_P':<8} {'Rate_P':<8} {'Completed':<10}")
            print(f"{'-'*130}")

        for i, result in enumerate(successful[:10], 1):
            gains = result['gains']
            flight_time = result.get('flight_time', 0)
            avg_lap_time = result.get('avg_lap_time', None)
            lap_count = len(result.get('lap_completion_times', []))
            lap_str = f"{avg_lap_time:.2f}s" if avg_lap_time is not None else "N/A"
            laps_str = str(lap_count) if lap_count > 0 else "-"
            completed = "✓" if result['completed'] and not result.get('early_termination', False) else "✗"
            mse = result.get('mse', result['score'])  # Use stored MSE if available
            stage = result.get('optimization_stage', 'unknown')

            if has_trajectory and 'trajectory' in result:
                traj = result['trajectory']
                print(f"{i:<6} {result['score']:<12.6f} {mse:<12.6f} {stage:<11} {lap_str:<9} {laps_str:<6} "
                      f"{gains['pos_P']:<8.2f} {gains['vel_P']:<8.2f} "
                      f"{gains['att_P']:<8.2f} {gains['rate_P']:<9.3f} "
                      f"{traj['period']:<8.2f} {traj['amplitude_x']:<8.2f} "
                      f"{traj['amplitude_y']:<8.2f} {traj['ramp_time']:<7.2f}")
            else:
                print(f"{i:<6} {result['score']:<12.6f} {mse:<12.6f} {stage:<11} {lap_str:<9} {laps_str:<6} {flight_time:<10.2f}s "
                      f"{gains['pos_P']:<8.2f} {gains['vel_P']:<8.2f} "
                      f"{gains['att_P']:<8.2f} {gains['rate_P']:<8.3f} "
                      f"{completed:<10}")

        # Best configuration
        best = successful[0]
        best_mse = best.get('mse', best['score'])
        best_stage = best.get('optimization_stage', 'unknown')

        print(f"\n{'='*70}")
        print(f"🎯 RECOMMENDED CONFIGURATION (Best Fitness Score):")
        print(f"{'='*70}")
        print(f"  CONTROL GAINS:")
        print(f"    Position gain:  {best['gains']['pos_P']:.3f}")
        print(f"    Velocity gain:  {best['gains']['vel_P']:.3f}")
        print(f"    Attitude gain:  {best['gains']['att_P']:.3f}")
        print(f"    Rate gain:      {best['gains']['rate_P']:.4f}")

        if 'trajectory' in best:
            traj = best['trajectory']
            print(f"\n  TRAJECTORY PARAMETERS:")
            print(f"    Period:         {traj['period']:.2f} s")
            print(f"    Amplitude X:    {traj['amplitude_x']:.2f} m")
            print(f"    Amplitude Y:    {traj['amplitude_y']:.2f} m")
            print(f"    Ramp time:      {traj['ramp_time']:.2f} s")

        print(f"\n  FITNESS METRICS:")
        print(f"    Fitness score:       {best['score']:.6f}")
        print(f"    Mean Squared Error:  {best_mse:.6f}")
        print(f"    Optimization stage:  {best_stage}")
        if best_stage in ['speed', 'speed_lap']:
            print(f"    (MSE < {self.mse_threshold:.6f} threshold, optimized for speed)")
        print(f"\n  FLIGHT METRICS:")
        print(f"    Flight time:              {best.get('flight_time', 0):.2f} s")
        if best.get('avg_lap_time') is not None:
            lap_count = len(best.get('lap_completion_times', []))
            print(f"    Laps completed:           {lap_count}")
            print(f"    Average lap time:         {best.get('avg_lap_time'):.3f} s")
            if len(best.get('lap_durations', [])) > 0:
                print(f"    Lap durations:            {[f'{d:.3f}' for d in best.get('lap_durations', [])]}")
        print(f"    Completed trajectory:     {best['completed']}")
        print(f"    Early termination:        {best.get('early_termination', False)}")
        print(f"    Instability count:        {best['instability_count']}")
        if best['completed']:
            print(f"\n  ACCURACY METRICS (completed flights):")
            print(f"    Mean position error:      {best['pos_error_mean']:.3f} m")
            print(f"    Steady-state error:       {best.get('pos_error_steady_state', 0):.3f} m")
            print(f"    Final position error:     {best['pos_error_final']:.3f} m")
        print(f"{'='*70}")

        # Command to test best configuration
        print(f"\n📋 To test the best configuration, run:")
        cmd = f"python examples/run_3D_simulation_lee_ctrl.py --gains " \
              f"{best['gains']['pos_P']:.3f},{best['gains']['vel_P']:.3f}," \
              f"{best['gains']['att_P']:.3f},{best['gains']['rate_P']:.4f}"

        if 'trajectory' in best:
            traj = best['trajectory']
            cmd += f" --traj-period {traj['period']:.2f} " \
                   f"--traj-amp-x {traj['amplitude_x']:.2f} " \
                   f"--traj-amp-y {traj['amplitude_y']:.2f} " \
                   f"--traj-ramp {traj['ramp_time']:.2f}"

        print(cmd)

        # Generate plots
        self._generate_plots(successful)

        # Save best configuration
        self._save_best_config(best)

    def _generate_plots(self, successful_results):
        """Generate visualization plots."""
        print(f"\n📊 Generating plots...")

        # Extract data
        pos_gains = [r['gains']['pos_P'] for r in successful_results]
        vel_gains = [r['gains']['vel_P'] for r in successful_results]
        att_gains = [r['gains']['att_P'] for r in successful_results]
        rate_gains = [r['gains']['rate_P'] for r in successful_results]
        scores = [r['score'] for r in successful_results]
        pos_errors = [r['pos_error_mean'] for r in successful_results]
        vel_errors = [r['vel_error_mean'] for r in successful_results]

        # Extract new stability metrics
        ss_errors = [r.get('pos_error_steady_state', r['pos_error_mean']) for r in successful_results]
        oscillation_indices = [r.get('oscillation_index', 0) for r in successful_results]
        growth_rates = [r.get('error_growth_rate', 0) * 100 for r in successful_results]

        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Lee Controller CMA-ES Tuning Results - MSE Analysis',
                    fontsize=16, fontweight='bold')

        # Plot 1: MSE vs Position Gain
        ax = axes[0, 0]
        scatter = ax.scatter(pos_gains, scores, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.set_xlabel('Position Gain (pos_P)')
        ax.set_ylabel('MSE (lower is better)')
        ax.set_title('MSE vs Position Gain')
        ax.grid(True, alpha=0.3)

        # Plot 2: MSE vs Velocity Gain
        ax = axes[0, 1]
        ax.scatter(vel_gains, scores, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.set_xlabel('Velocity Gain (vel_P)')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Velocity Gain')
        ax.grid(True, alpha=0.3)

        # Plot 3: MSE vs Attitude Gain
        ax = axes[0, 2]
        ax.scatter(att_gains, scores, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.set_xlabel('Attitude Gain (att_P)')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Attitude Gain')
        ax.grid(True, alpha=0.3)

        # Plot 4: MSE vs Rate Gain
        ax = axes[1, 0]
        ax.scatter(rate_gains, scores, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.set_xlabel('Rate Gain (rate_P)')
        ax.set_ylabel('MSE')
        ax.set_title('MSE vs Rate Gain')
        ax.grid(True, alpha=0.3)

        # Plot 5: Steady-State Error vs Oscillation (KEY STABILITY PLOT)
        ax = axes[1, 1]
        scatter_stability = ax.scatter(ss_errors, oscillation_indices, c=scores,
                                       cmap='RdYlGn_r', alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
        ax.set_xlabel('Steady-State Error (m)')
        ax.set_ylabel('Oscillation Index')
        ax.set_title('Stability Map (lower-left is best)')
        ax.grid(True, alpha=0.3)
        # Mark best region
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax.axvline(x=0.2, color='green', linestyle='--', alpha=0.3, linewidth=1)

        # Plot 6: Error Growth Rate Distribution
        ax = axes[1, 2]
        colors = ['green' if g < 0 else 'red' for g in growth_rates]
        ax.scatter(range(len(growth_rates)), growth_rates, c=colors, alpha=0.6, s=40)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Unstable threshold')
        ax.set_xlabel('Configuration Index')
        ax.set_ylabel('Error Growth Rate (%)')
        ax.set_title('Error Growth (negative is good)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 7: Mean vs Steady-State Error
        ax = axes[2, 0]
        ax.scatter(pos_errors, ss_errors, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.plot([0, max(pos_errors)], [0, max(pos_errors)], 'k--', alpha=0.3, label='y=x')
        ax.set_xlabel('Mean Position Error (m)')
        ax.set_ylabel('Steady-State Error (m)')
        ax.set_title('Convergence Quality')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 8: Gain Ratio Analysis (Pos/Vel)
        ax = axes[2, 1]
        gain_ratios = [p/v if v > 0 else 0 for p, v in zip(pos_gains, vel_gains)]
        ax.scatter(gain_ratios, scores, c=scores, cmap='RdYlGn_r', alpha=0.6, s=50)
        ax.set_xlabel('Position/Velocity Gain Ratio')
        ax.set_ylabel('MSE')
        ax.set_title('Gain Ratio vs MSE')
        ax.grid(True, alpha=0.3)

        # Plot 9: MSE Distribution
        ax = axes[2, 2]
        ax.hist(scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('MSE')
        ax.set_ylabel('Frequency')
        ax.set_title('MSE Distribution')
        ax.axvline(np.median(scores), color='red', linestyle='--', linewidth=2, label='Median')
        ax.axvline(np.percentile(scores, 25), color='green', linestyle='--',
                  linewidth=2, label='Top 25%')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(scatter, ax=axes.ravel().tolist(), label='MSE',
                    shrink=0.6, pad=0.01)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"{self.output_dir}/tuning_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Plots saved to: {plot_file}")

        # Close the figure to free memory (don't show interactively during tuning)
        plt.close(fig)

    def _save_best_config(self, best_result):
        """Save best configuration to a file."""
        config = {
            'timestamp': datetime.now().isoformat(),
            'gains': best_result['gains'],
            'performance': {
                'mse': best_result['score'],
                'rmse': float(np.sqrt(best_result['score'])) if best_result['completed'] else None,
                'pos_error_mean': best_result['pos_error_mean'],
                'pos_error_max': best_result['pos_error_max'],
                'vel_error_mean': best_result['vel_error_mean'],
                'vel_error_max': best_result['vel_error_max'],
                'completed': best_result['completed']
            },
            'command': (
                f"python examples/run_3D_simulation_lee_ctrl.py --gains "
                f"{best_result['gains']['pos_P']:.3f},"
                f"{best_result['gains']['vel_P']:.3f},"
                f"{best_result['gains']['att_P']:.3f},"
                f"{best_result['gains']['rate_P']:.4f}"
            )
        }

        # Add trajectory parameters if present
        if 'trajectory' in best_result:
            config['trajectory'] = best_result['trajectory']
            traj = best_result['trajectory']
            config['command'] += (
                f" --traj-period {traj['period']:.2f} "
                f"--traj-amp-x {traj['amplitude_x']:.2f} "
                f"--traj-amp-y {traj['amplitude_y']:.2f} "
                f"--traj-ramp {traj['ramp_time']:.2f}"
            )

        config_file = f"{self.output_dir}/best_gains.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"  ✓ Best config saved to: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description='CMA-ES Optimization for Lee Geometric Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default CMA-ES optimization (200 evaluations, parallel)
  python tune_lee_controller.py

  # CMA-ES with more evaluations for better results
  python tune_lee_controller.py --max-evals 500

  # CMA-ES with custom worker count
  python tune_lee_controller.py --max-evals 200 --workers 8

  # CMA-ES with serial execution (no parallelization)
  python tune_lee_controller.py --max-evals 200 --workers 1

  # Custom simulation time and output directory
  python tune_lee_controller.py --time 20.0 --output my_results

Note: This script requires the 'cma' package. Install with: pip install cma
        """
    )

    parser.add_argument('--max-evals', type=int, default=200,
                       help='Maximum evaluations for CMA-ES (default: 200)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU_count/2)')
    parser.add_argument('--time', type=float, default=15.0,
                       help='Simulation time per test in seconds (default: 15.0)')
    parser.add_argument('--dt', type=float, default=0.005,
                       help='Simulation time step (default: 0.005)')
    parser.add_argument('--output', type=str, default='tuning_results',
                       help='Output directory for results (default: tuning_results)')
    parser.add_argument('--no-tune-trajectory', action='store_true',
                       help='Only tune controller gains, not trajectory parameters (default: tune both)')
    parser.add_argument('--mse-threshold', type=float, default=5.0,
                       help='MSE threshold for two-stage optimization: optimize for accuracy until MSE < threshold, then optimize for speed (default: 0.05)')

    args = parser.parse_args()

    # Create tuner
    tuner = ControllerTuner(sim_time=args.time, dt=args.dt, output_dir=args.output,
                           mse_threshold=args.mse_threshold)

    # Run CMA-ES optimization
    tuner.run_cmaes_tuning(max_evaluations=args.max_evals, num_workers=args.workers,
                          tune_trajectory=not args.no_tune_trajectory)


if __name__ == '__main__':
    main()
