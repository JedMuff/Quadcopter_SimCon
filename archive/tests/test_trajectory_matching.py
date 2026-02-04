#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory Matching Test

Test if the improved force_original_parameters() method achieves better trajectory matching.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Import both controller types
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from trajectory import Trajectory
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from utils.windModel import Wind
import utils
import config

def run_trajectory_comparison():
    """Run trajectory comparison with improved parameter matching."""
    
    print("Testing trajectory matching with improved force_original_parameters()...")
    
    # Common parameters
    Ti = 0
    Tf = 10.0
    Ts = 0.01
    t_all = np.arange(Ti, Tf + Ts, Ts)
    
    # Create identical trajectory for both systems
    trajSelect = np.zeros(3)
    trajSelect[0] = 5  # minimum jerk trajectory
    trajSelect[1] = 3  # follow yaw
    trajSelect[2] = 1  # use average speed
    
    ctrlType = "xyz_pos"
    waypoints = np.array([
        [0, 0, 0, 0],
        [2, 0, -2, 0], 
        [2, 2, -2, np.pi/2],
        [0, 2, -2, np.pi],
        [0, 0, -2, 0]
    ])
    
    # Wind model (identical for both)
    wind = Wind("None", 0, 0, 0)
    
    # Setup original system
    print("Setting up original system...")
    quad_orig = OriginalQuadcopter(Ti)
    ctrl_orig = OriginalControl(quad_orig, yawType=1)
    traj = Trajectory(quad_orig, ctrlType, trajSelect)
    
    # Setup generalized system with improved matching
    print("Setting up generalized system with improved matching...")
    orig_temp = OriginalQuadcopter(0)
    arm_length = orig_temp.params["dxm"]
    
    propeller_config = create_standard_propeller_config(
        config_type="quad",
        arm_length=arm_length,
        prop_size="matched"
    )
    quad_gen = ConfigurableQuadcopter(Ti, propellers=propeller_config)
    
    # Apply improved parameter forcing
    quad_gen.force_original_parameters()
    
    ctrl_gen = GeneralizedControl(quad_gen, yawType=1)
    
    print("Running trajectory simulations...")
    
    # Storage for trajectories
    orig_trajectory = []
    gen_trajectory = []
    
    # Run original system
    print("  Running original controller...")
    for t in t_all:
        # Store position before update
        orig_trajectory.append(quad_orig.pos.copy())
        
        # Get desired state and run control
        sDes = traj.desiredState(t, Ts, quad_orig)
        ctrl_orig.controller(traj, quad_orig, sDes, Ts)
        quad_orig.update(t, Ts, ctrl_orig.w_cmd, wind)
    
    # Reset and run generalized system  
    print("  Running generalized controller...")
    quad_gen = ConfigurableQuadcopter(Ti, propellers=propeller_config)
    quad_gen.force_original_parameters()  # Apply parameter forcing again
    ctrl_gen = GeneralizedControl(quad_gen, yawType=1)
    
    for t in t_all:
        # Store position before update
        gen_trajectory.append(quad_gen.pos.copy())
        
        # Get desired state and run control
        sDes = traj.desiredState(t, Ts, quad_gen)
        ctrl_gen.controller(traj, quad_gen, sDes, Ts)
        quad_gen.update(t, Ts, ctrl_gen.w_cmd, wind)
    
    # Convert to numpy arrays
    orig_trajectory = np.array(orig_trajectory)
    gen_trajectory = np.array(gen_trajectory)
    
    # Calculate trajectory differences
    position_differences = np.linalg.norm(orig_trajectory - gen_trajectory, axis=1)
    max_error = np.max(position_differences)
    avg_error = np.mean(position_differences)
    final_error = position_differences[-1]
    
    print("\n" + "="*60)
    print("IMPROVED TRAJECTORY MATCHING RESULTS")
    print("="*60)
    print(f"Maximum trajectory error: {max_error:.6f} m")
    print(f"Average trajectory error: {avg_error:.6f} m")
    print(f"Final position error: {final_error:.6f} m")
    
    # Compare with previous results (158m error mentioned in the problem)
    previous_error = 158.0  # From the problem description
    improvement_factor = previous_error / final_error if final_error > 0 else float('inf')
    
    print(f"\nComparison with previous results:")
    print(f"Previous final error: {previous_error:.1f} m")
    print(f"Current final error: {final_error:.6f} m")
    print(f"Improvement factor: {improvement_factor:.1f}x better")
    
    if final_error < 0.1:
        print("✅ SUCCESS: Trajectory matching significantly improved!")
    elif final_error < 1.0:
        print("⚠️  PARTIAL SUCCESS: Good improvement but still some divergence")
    else:
        print("❌ ISSUE: Still significant trajectory divergence")
        
    # Detailed error analysis
    print(f"\nDetailed error analysis:")
    print(f"  Errors > 1m: {np.sum(position_differences > 1.0)} time steps")
    print(f"  Errors > 0.1m: {np.sum(position_differences > 0.1)} time steps")
    print(f"  Errors > 0.01m: {np.sum(position_differences > 0.01)} time steps")
    
    # Create diagnostic plot
    plt.figure(figsize=(15, 10))
    
    # 3D trajectory comparison
    plt.subplot(2, 3, 1)
    plt.plot(orig_trajectory[:, 0], orig_trajectory[:, 1], 'b-', label='Original', linewidth=2)
    plt.plot(gen_trajectory[:, 0], gen_trajectory[:, 1], 'r--', label='Generalized', linewidth=2)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('XY Trajectory Comparison')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    # Altitude comparison
    plt.subplot(2, 3, 2)
    plt.plot(t_all[:-1], orig_trajectory[:, 2], 'b-', label='Original', linewidth=2)
    plt.plot(t_all[:-1], gen_trajectory[:, 2], 'r--', label='Generalized', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Altitude Comparison')
    plt.legend()
    plt.grid(True)
    
    # Position error over time
    plt.subplot(2, 3, 3)
    plt.plot(t_all[:-1], position_differences, 'g-', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Position Error (m)')
    plt.title('Position Error Over Time')
    plt.grid(True)
    plt.yscale('log')
    
    # Error distribution
    plt.subplot(2, 3, 4)
    plt.hist(position_differences, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Position Error (m)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True)
    
    # Cumulative error
    plt.subplot(2, 3, 5)
    cumulative_error = np.cumsum(position_differences)
    plt.plot(t_all[:-1], cumulative_error, 'purple', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Cumulative Error (m)')
    plt.title('Cumulative Position Error')
    plt.grid(True)
    
    # Final positions comparison
    plt.subplot(2, 3, 6)
    final_positions = np.array([orig_trajectory[-1], gen_trajectory[-1]])
    labels = ['Original', 'Generalized']
    colors = ['blue', 'red']
    
    for i, (pos, label, color) in enumerate(zip(final_positions, labels, colors)):
        plt.scatter(pos[0], pos[1], c=color, s=100, label=f'{label}: ({pos[0]:.3f}, {pos[1]:.3f})')
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Final Positions')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.tight_layout()
    plt.savefig('improved_trajectory_matching.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Diagnostic plots saved as 'improved_trajectory_matching.png'")
    
    return {
        'max_error': max_error,
        'avg_error': avg_error,
        'final_error': final_error,
        'improvement_factor': improvement_factor,
        'trajectory_original': orig_trajectory,
        'trajectory_generalized': gen_trajectory,
        'position_differences': position_differences
    }

if __name__ == "__main__":
    results = run_trajectory_comparison()