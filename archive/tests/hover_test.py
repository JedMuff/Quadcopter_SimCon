#!/usr/bin/env python3
"""
Simple Hover Test

This script tests the configurable drone framework with a simple hovering task.
The drone should start at [0,0,0] and reach target altitude within 0.25m tolerance.

Usage:
    python hover_test.py --framework original    # Test original framework
    python hover_test.py --framework config      # Test configurable framework
    python hover_test.py --framework both        # Test both frameworks
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import time

# Import frameworks
from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from drone_simulator import ConfigurableQuadcopter
from utils.windModel import Wind
import utils
import config

def test_original_framework(target_pos, sim_time=10, plot=False):
    """Test hovering with original framework."""
    print("Testing Original Framework")
    print("-" * 30)
    
    # Initialize
    quad = Quadcopter(0)
    
    trajSelect = np.zeros(3)
    trajSelect[0] = 1  # pos_waypoint_timed
    trajSelect[1] = 0  # no yaw
    trajSelect[2] = 0  # waypoint time
    
    traj = Trajectory(quad, "xyz_pos", trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 0, 0, 0)
    
    # Set target position
    traj.wps = np.array([[0, 0, 0], target_pos])
    traj.t_wps = np.array([0, sim_time])
    
    # Print initial state
    print(f"Initial position: {quad.pos}")
    print(f"Target position: {target_pos}")
    print(f"Mass: {quad.params['mB']:.3f} kg")
    print(f"kTh: {quad.params['kTh']:.2e}")
    
    # Simulation arrays
    dt = 0.005
    num_steps = int(sim_time / dt)
    
    times = []
    positions = []
    errors = []
    
    # Run simulation
    t = 0
    for i in range(num_steps):
        # Get desired states
        sDes = traj.desiredState(t, dt, quad)
        
        # Run controller
        ctrl.controller(traj, quad, sDes, dt)
        
        # Update dynamics
        quad.update(t, dt, ctrl.w_cmd, wind)
        t += dt
        
        # Log data
        times.append(t)
        positions.append(quad.pos.copy())
        error = np.linalg.norm(quad.pos - target_pos)
        errors.append(error)
        
        # Progress update
        if i % int(2.0/dt) == 0:  # Every 2 seconds
            print(f"t={t:.1f}s: pos={quad.pos}, error={error:.3f}m")
    
    # Final results
    final_pos = quad.pos
    final_error = np.linalg.norm(final_pos - target_pos)
    success = final_error < 0.25
    
    print(f"\\nFinal position: {final_pos}")
    print(f"Final error: {final_error:.4f} m")
    print(f"Success (< 0.25m): {'✓' if success else '✗'}")
    
    if plot:
        plot_results(times, positions, errors, target_pos, "Original Framework")
    
    return success, final_error, times, positions, errors

def test_configurable_framework(target_pos, sim_time=10, plot=False):
    """Test hovering with configurable framework."""
    print("\\nTesting Configurable Framework")
    print("-" * 30)
    
    # Initialize with similar configuration to original
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    trajSelect = np.zeros(3)
    trajSelect[0] = 1  # pos_waypoint_timed
    trajSelect[1] = 0  # no yaw
    trajSelect[2] = 0  # waypoint time
    
    traj = Trajectory(quad, "xyz_pos", trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 0, 0, 0)
    
    # Set target position
    traj.wps = np.array([[0, 0, 0], target_pos])
    traj.t_wps = np.array([0, sim_time])
    
    # Print initial state
    print(f"Initial position: {quad.pos}")
    print(f"Target position: {target_pos}")
    print(f"Mass: {quad.params['mB']:.3f} kg")
    print(f"kTh: {quad.params['kTh']:.2e}")
    
    # Simulation arrays
    dt = 0.005
    num_steps = int(sim_time / dt)
    
    times = []
    positions = []
    errors = []
    
    # Run simulation
    t = 0
    for i in range(num_steps):
        # Get desired states
        sDes = traj.desiredState(t, dt, quad)
        
        # Run controller
        ctrl.controller(traj, quad, sDes, dt)
        
        # Update dynamics
        quad.update(t, dt, ctrl.w_cmd, wind)
        t += dt
        
        # Log data
        times.append(t)
        positions.append(quad.pos.copy())
        error = np.linalg.norm(quad.pos - target_pos)
        errors.append(error)
        
        # Progress update
        if i % int(2.0/dt) == 0:  # Every 2 seconds
            print(f"t={t:.1f}s: pos={quad.pos}, error={error:.3f}m")
    
    # Final results
    final_pos = quad.pos
    final_error = np.linalg.norm(final_pos - target_pos)
    success = final_error < 0.25
    
    print(f"\\nFinal position: {final_pos}")
    print(f"Final error: {final_error:.4f} m")
    print(f"Success (< 0.25m): {'✓' if success else '✗'}")
    
    if plot:
        plot_results(times, positions, errors, target_pos, "Configurable Framework")
    
    return success, final_error, times, positions, errors

def plot_results(times, positions, errors, target_pos, title):
    """Plot simulation results."""
    times = np.array(times)
    positions = np.array(positions)
    errors = np.array(errors)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Hover Test Results - {title}')
    
    # Position over time
    axes[0,0].plot(times, positions[:, 0], 'r-', label='X', linewidth=2)
    axes[0,0].plot(times, positions[:, 1], 'g-', label='Y', linewidth=2)
    axes[0,0].plot(times, positions[:, 2], 'b-', label='Z', linewidth=2)
    axes[0,0].axhline(y=target_pos[0], color='r', linestyle='--', alpha=0.5)
    axes[0,0].axhline(y=target_pos[1], color='g', linestyle='--', alpha=0.5)
    axes[0,0].axhline(y=target_pos[2], color='b', linestyle='--', alpha=0.5)
    axes[0,0].set_title('Position vs Time')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 3D trajectory
    axes[0,1] = fig.add_subplot(2, 2, 2, projection='3d')
    axes[0,1].plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    axes[0,1].scatter([0], [0], [0], color='g', s=100, label='Start')
    axes[0,1].scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='r', s=100, label='Target')
    axes[0,1].set_title('3D Trajectory')
    axes[0,1].set_xlabel('X (m)')
    axes[0,1].set_ylabel('Y (m)')
    axes[0,1].set_zlabel('Z (m)')
    axes[0,1].legend()
    
    # Error over time
    axes[1,0].plot(times, errors, 'b-', linewidth=2)
    axes[1,0].axhline(y=0.25, color='r', linestyle='--', label='Success threshold')
    axes[1,0].set_title('Position Error vs Time')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Error (m)')
    axes[1,0].set_yscale('log')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Error histogram
    axes[1,1].hist(errors, bins=50, alpha=0.7, edgecolor='black')
    axes[1,1].axvline(x=0.25, color='r', linestyle='--', label='Success threshold')
    axes[1,1].set_title('Error Distribution')
    axes[1,1].set_xlabel('Error (m)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def compare_frameworks(target_pos, sim_time=10):
    """Compare both frameworks side by side."""
    print("="*60)
    print("FRAMEWORK COMPARISON")
    print("="*60)
    
    # Test original
    orig_success, orig_error, orig_times, orig_pos, orig_errors = test_original_framework(target_pos, sim_time)
    
    # Test configurable
    config_success, config_error, config_times, config_pos, config_errors = test_configurable_framework(target_pos, sim_time)
    
    # Summary
    print("\\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Target: {target_pos}")
    print(f"Original Framework   - Success: {'✓' if orig_success else '✗'}, Error: {orig_error:.4f}m")
    print(f"Configurable Framework - Success: {'✓' if config_success else '✗'}, Error: {config_error:.4f}m")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Framework Comparison')
    
    orig_times = np.array(orig_times)
    orig_pos = np.array(orig_pos)
    orig_errors = np.array(orig_errors)
    
    config_times = np.array(config_times)
    config_pos = np.array(config_pos)
    config_errors = np.array(config_errors)
    
    # Position Z comparison
    axes[0,0].plot(orig_times, orig_pos[:, 2], 'b-', label='Original', linewidth=2)
    axes[0,0].plot(config_times, config_pos[:, 2], 'r--', label='Configurable', linewidth=2)
    axes[0,0].axhline(y=target_pos[2], color='g', linestyle=':', label='Target')
    axes[0,0].set_title('Z Position Comparison')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Z Position (m)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Error comparison
    axes[0,1].plot(orig_times, orig_errors, 'b-', label='Original', linewidth=2)
    axes[0,1].plot(config_times, config_errors, 'r--', label='Configurable', linewidth=2)
    axes[0,1].axhline(y=0.25, color='g', linestyle=':', label='Success threshold')
    axes[0,1].set_title('Error Comparison')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_yscale('log')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # 3D comparison
    axes[1,0] = fig.add_subplot(2, 2, 3, projection='3d')
    axes[1,0].plot(orig_pos[:, 0], orig_pos[:, 1], orig_pos[:, 2], 'b-', label='Original', linewidth=2)
    axes[1,0].plot(config_pos[:, 0], config_pos[:, 1], config_pos[:, 2], 'r--', label='Configurable', linewidth=2)
    axes[1,0].scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='g', s=100, label='Target')
    axes[1,0].set_title('3D Trajectory Comparison')
    axes[1,0].set_xlabel('X (m)')
    axes[1,0].set_ylabel('Y (m)')
    axes[1,0].set_zlabel('Z (m)')
    axes[1,0].legend()
    
    # Final error comparison
    categories = ['Original', 'Configurable']
    final_errors = [orig_error, config_error]
    colors = ['blue', 'red']
    
    bars = axes[1,1].bar(categories, final_errors, color=colors, alpha=0.7)
    axes[1,1].axhline(y=0.25, color='green', linestyle='--', label='Success threshold')
    axes[1,1].set_title('Final Error Comparison')
    axes[1,1].set_ylabel('Final Error (m)')
    axes[1,1].legend()
    axes[1,1].grid(True, axis='y')
    
    # Add value labels on bars
    for bar, error in zip(bars, final_errors):
        height = bar.get_height()
        axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{error:.3f}m', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simple hover test')
    parser.add_argument('--framework', choices=['original', 'config', 'both'], 
                       default='both', help='Which framework to test')
    parser.add_argument('--target', type=str, default='0,0,1', 
                       help='Target position x,y,z')
    parser.add_argument('--time', type=float, default=10, 
                       help='Simulation time (s)')
    parser.add_argument('--plot', action='store_true', 
                       help='Show plots')
    
    args = parser.parse_args()
    
    # Parse target position
    target_pos = np.array([float(x) for x in args.target.split(',')])
    
    # Run tests
    if args.framework == 'original':
        test_original_framework(target_pos, args.time, args.plot)
    elif args.framework == 'config':
        test_configurable_framework(target_pos, args.time, args.plot)
    elif args.framework == 'both':
        compare_frameworks(target_pos, args.time)

if __name__ == "__main__":
    main()