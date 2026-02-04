#!/usr/bin/env python3
"""
Test Trajectory Tracking Performance

Compare trajectory tracking between original and configurable frameworks
with matched physical parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter
from trajectory import Trajectory
from ctrl import Control
from utils.windModel import Wind

def test_trajectory_tracking():
    """Test trajectory tracking performance for both frameworks."""
    print("TRAJECTORY TRACKING PERFORMANCE TEST")
    print("=" * 45)
    
    # Simulation parameters
    Ti = 0
    Ts = 0.005
    Tf = 10.0  # Shorter test
    
    # Trajectory settings
    trajSelect = np.array([5, 3, 1])  # minimum jerk, follow yaw, average speed
    ctrlType = "xyz_pos"
    
    # Initialize frameworks
    print("Initializing frameworks...")
    
    # Original framework
    quad_orig = OriginalQuadcopter(Ti)
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    wind = Wind('None', 2.0, 90, -15)
    
    # Configurable framework  
    quad_config = ConfigurableQuadcopter(Ti, drone_type="quad", arm_length=0.11, prop_size=5)
    traj_config = Trajectory(quad_config, ctrlType, trajSelect)
    ctrl_config = Control(quad_config, traj_config.yawType)
    
    print(f"Original quad mass: {quad_orig.params['mB']:.3f} kg")
    print(f"Configurable quad mass: {quad_config.params['mB']:.3f} kg")
    
    # Initialize desired states
    sDes_orig = traj_orig.desiredState(0, Ts, quad_orig)
    sDes_config = traj_config.desiredState(0, Ts, quad_config)
    
    ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
    ctrl_config.controller(traj_config, quad_config, sDes_config, Ts)
    
    # Run simulation
    print(f"\nRunning {Tf}s simulation...")
    
    # Storage arrays
    numSteps = int(Tf/Ts)
    t_all = np.zeros(numSteps)
    pos_orig = np.zeros((numSteps, 3))
    pos_config = np.zeros((numSteps, 3))
    pos_des = np.zeros((numSteps, 3))
    
    # Initial states
    pos_orig[0] = quad_orig.pos
    pos_config[0] = quad_config.pos
    pos_des[0] = sDes_orig[:3]
    
    # Simulation loop
    t = Ti
    for i in range(1, numSteps):
        # Original framework
        quad_orig.update(t, Ts, ctrl_orig.w_cmd, wind)
        sDes_orig = traj_orig.desiredState(t + Ts, Ts, quad_orig)
        ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
        
        # Configurable framework
        quad_config.update(t, Ts, ctrl_config.w_cmd, wind)
        sDes_config = traj_config.desiredState(t + Ts, Ts, quad_config)
        ctrl_config.controller(traj_config, quad_config, sDes_config, Ts)
        
        # Store results
        t += Ts
        t_all[i] = t
        pos_orig[i] = quad_orig.pos
        pos_config[i] = quad_config.pos
        pos_des[i] = sDes_orig[:3]
    
    # Calculate tracking errors
    error_orig = np.linalg.norm(pos_orig - pos_des, axis=1)
    error_config = np.linalg.norm(pos_config - pos_des, axis=1)
    
    # Results
    print(f"\nTrajectory Tracking Results:")
    print(f"Original framework:")
    print(f"  Final position: [{pos_orig[-1][0]:.3f}, {pos_orig[-1][1]:.3f}, {pos_orig[-1][2]:.3f}]")
    print(f"  Mean tracking error: {np.mean(error_orig):.4f} m")
    print(f"  Max tracking error: {np.max(error_orig):.4f} m")
    
    print(f"Configurable framework:")
    print(f"  Final position: [{pos_config[-1][0]:.3f}, {pos_config[-1][1]:.3f}, {pos_config[-1][2]:.3f}]")
    print(f"  Mean tracking error: {np.mean(error_config):.4f} m")
    print(f"  Max tracking error: {np.max(error_config):.4f} m")
    
    print(f"Desired final position: [{pos_des[-1][0]:.3f}, {pos_des[-1][1]:.3f}, {pos_des[-1][2]:.3f}]")
    
    # Compare performance
    mean_diff = abs(np.mean(error_orig) - np.mean(error_config))
    max_diff = abs(np.max(error_orig) - np.max(error_config))
    
    print(f"\nPerformance Comparison:")
    print(f"  Mean error difference: {mean_diff:.4f} m")
    print(f"  Max error difference: {max_diff:.4f} m")
    
    # Determine if tracking is similar
    similar_performance = mean_diff < 0.1 and max_diff < 0.2
    
    if similar_performance:
        print("✅ Both frameworks show similar trajectory tracking performance")
    else:
        print("❌ Frameworks show different trajectory tracking performance")
        print("   This may be due to different physical parameters or control tuning")
    
    return similar_performance, error_orig, error_config, t_all, pos_orig, pos_config, pos_des

def create_tracking_plot(t_all, pos_orig, pos_config, pos_des, error_orig, error_config):
    """Create comparison plots of trajectory tracking."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 3D trajectory plot
    ax1.plot(pos_des[:, 0], pos_des[:, 1], 'k--', label='Desired', linewidth=2)
    ax1.plot(pos_orig[:, 0], pos_orig[:, 1], 'b-', label='Original', linewidth=1.5)
    ax1.plot(pos_config[:, 0], pos_config[:, 1], 'r-', label='Configurable', linewidth=1.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')
    
    # Z trajectory
    ax2.plot(t_all, pos_des[:, 2], 'k--', label='Desired', linewidth=2)
    ax2.plot(t_all, pos_orig[:, 2], 'b-', label='Original', linewidth=1.5)
    ax2.plot(t_all, pos_config[:, 2], 'r-', label='Configurable', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Altitude Profile')
    ax2.legend()
    ax2.grid(True)
    
    # Tracking errors
    ax3.plot(t_all, error_orig, 'b-', label='Original', linewidth=1.5)
    ax3.plot(t_all, error_config, 'r-', label='Configurable', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (m)')
    ax3.set_title('Trajectory Tracking Error')
    ax3.legend()
    ax3.grid(True)
    
    # Position comparison
    for i, label in enumerate(['X', 'Y', 'Z']):
        ax4.plot(t_all, pos_orig[:, i] - pos_config[:, i], label=f'{label} diff')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Difference (m)')
    ax4.set_title('Original vs Configurable Position Difference')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('trajectory_tracking_comparison.png', dpi=150, bbox_inches='tight')
    print("\n📊 Tracking comparison plot saved as 'trajectory_tracking_comparison.png'")

def main():
    print("TRAJECTORY AND WAYPOINT SYSTEM VALIDATION")
    print("=" * 50)
    
    # Run trajectory tracking test
    success, error_orig, error_config, t_all, pos_orig, pos_config, pos_des = test_trajectory_tracking()
    
    # Create comparison plot
    create_tracking_plot(t_all, pos_orig, pos_config, pos_des, error_orig, error_config)
    
    print(f"\n\nCONCLUSION")
    print("=" * 15)
    
    if success:
        print("✅ TRAJECTORY SYSTEMS ARE EQUIVALENT")
        print("   Both frameworks use identical waypoints and trajectory generation")
        print("   Tracking performance is very similar between frameworks")
        print("   No corrections needed to trajectory or waypoint systems")
    else:
        print("⚠️  PERFORMANCE DIFFERENCES DETECTED")
        print("   While trajectory generation is identical, tracking performance differs")
        print("   This is likely due to different physical parameters or control tuning")
        print("   Consider matching physical parameters between frameworks")

if __name__ == "__main__":
    main()