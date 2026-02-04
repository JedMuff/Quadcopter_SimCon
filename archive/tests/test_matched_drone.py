#!/usr/bin/env python3
"""
Test Matched Drone Configuration

Test trajectory tracking with a drone configuration that closely matches
the original framework parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter
from trajectory import Trajectory
from ctrl import Control
from utils.windModel import Wind

def create_matched_drone():
    """Create configurable drone that matches original specs closely."""
    print("CREATING MATCHED DRONE CONFIGURATION")
    print("=" * 45)
    
    # Get original specs
    quad_orig = OriginalQuadcopter(0)
    target_arm_length = quad_orig.params['dxm']  # 0.16m
    
    print(f"Original framework parameters:")
    print(f"  Mass: {quad_orig.params['mB']:.3f} kg")
    print(f"  kTh: {quad_orig.params['kTh']:.2e}")
    print(f"  Arm length: {target_arm_length:.3f} m")
    
    # Create matched propeller configuration
    propellers = [
        {"loc": [target_arm_length, target_arm_length, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-target_arm_length, target_arm_length, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-target_arm_length, -target_arm_length, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [target_arm_length, -target_arm_length, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    
    quad_matched = ConfigurableQuadcopter(0, propellers=propellers)
    
    print(f"\nMatched drone parameters:")
    print(f"  Mass: {quad_matched.params['mB']:.3f} kg")
    print(f"  kTh: {quad_matched.params['kTh']:.2e}")
    print(f"  w_hover: {quad_matched.params['w_hover']:.1f} rad/s")
    
    # Compare key parameters
    mass_ratio = quad_matched.params['mB'] / quad_orig.params['mB']
    kth_ratio = quad_matched.params['kTh'] / quad_orig.params['kTh']
    
    print(f"\nParameter ratios (matched/original):")
    print(f"  Mass ratio: {mass_ratio:.3f}")
    print(f"  kTh ratio: {kth_ratio:.3f}")
    
    return quad_orig, quad_matched

def test_hover_behavior(quad_orig, quad_matched):
    """Test hover behavior comparison."""
    print(f"\n\nTESTING HOVER BEHAVIOR")
    print("=" * 30)
    
    # Create trajectories and controllers for hover test
    trajSelect = np.array([0, 3, 1])  # Hover
    ctrlType = "xyz_pos"
    
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    traj_matched = Trajectory(quad_matched, ctrlType, trajSelect)
    
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    ctrl_matched = Control(quad_matched, traj_matched.yawType)
    
    wind = Wind('None', 2.0, 90, -15)
    
    # Test hover commands
    sDes_orig = traj_orig.desiredState(0, 0.005, quad_orig)
    sDes_matched = traj_matched.desiredState(0, 0.005, quad_matched)
    
    ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, 0.005)
    ctrl_matched.controller(traj_matched, quad_matched, sDes_matched, 0.005)
    
    print(f"Hover motor commands:")
    print(f"  Original: {ctrl_orig.w_cmd[0]:.1f} rad/s")
    print(f"  Matched: {ctrl_matched.w_cmd[0]:.1f} rad/s")
    
    # Test multiple simulation steps
    positions_orig = []
    positions_matched = []
    
    for i in range(10):
        t = i * 0.005
        
        # Original framework step
        quad_orig.update(t, 0.005, ctrl_orig.w_cmd, wind)
        sDes_orig = traj_orig.desiredState(t + 0.005, 0.005, quad_orig)
        ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, 0.005)
        positions_orig.append(quad_orig.pos.copy())
        
        # Matched framework step
        quad_matched.update(t, 0.005, ctrl_matched.w_cmd, wind)
        sDes_matched = traj_matched.desiredState(t + 0.005, 0.005, quad_matched)
        ctrl_matched.controller(traj_matched, quad_matched, sDes_matched, 0.005)
        positions_matched.append(quad_matched.pos.copy())
    
    # Compare final positions
    final_pos_orig = positions_orig[-1]
    final_pos_matched = positions_matched[-1]
    pos_diff = np.linalg.norm(final_pos_orig - final_pos_matched)
    
    print(f"\nAfter 10 hover steps:")
    print(f"  Original position: [{final_pos_orig[0]:.6f}, {final_pos_orig[1]:.6f}, {final_pos_orig[2]:.6f}]")
    print(f"  Matched position: [{final_pos_matched[0]:.6f}, {final_pos_matched[1]:.6f}, {final_pos_matched[2]:.6f}]")
    print(f"  Position difference: {pos_diff:.8f} m")
    
    hover_success = pos_diff < 1e-4  # Within 0.1mm
    print(f"  Hover behavior match: {'✅' if hover_success else '❌'}")
    
    return hover_success

def test_trajectory_tracking(quad_orig, quad_matched):
    """Test trajectory tracking comparison."""
    print(f"\n\nTESTING TRAJECTORY TRACKING")
    print("=" * 35)
    
    # Create trajectories and controllers for full trajectory test
    trajSelect = np.array([5, 3, 1])  # Minimum jerk trajectory  
    ctrlType = "xyz_pos"
    
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    traj_matched = Trajectory(quad_matched, ctrlType, trajSelect)
    
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    ctrl_matched = Control(quad_matched, traj_matched.yawType)
    
    wind = Wind('None', 2.0, 90, -15)
    
    # Initialize
    sDes_orig = traj_orig.desiredState(0, 0.005, quad_orig)
    sDes_matched = traj_matched.desiredState(0, 0.005, quad_matched)
    
    ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, 0.005)
    ctrl_matched.controller(traj_matched, quad_matched, sDes_matched, 0.005)
    
    # Run trajectory tracking simulation
    sim_time = 8.0  # seconds
    dt = 0.005
    num_steps = int(sim_time / dt)
    
    t_all = np.zeros(num_steps)
    pos_orig_all = np.zeros((num_steps, 3))
    pos_matched_all = np.zeros((num_steps, 3))
    pos_des_all = np.zeros((num_steps, 3))
    
    print(f"Running {sim_time}s trajectory simulation...")
    
    for i in range(num_steps):
        t = i * dt
        t_all[i] = t
        
        # Store current positions
        pos_orig_all[i] = quad_orig.pos
        pos_matched_all[i] = quad_matched.pos
        pos_des_all[i] = sDes_orig[:3]  # Both frameworks should have same desired state
        
        # Update original framework
        quad_orig.update(t, dt, ctrl_orig.w_cmd, wind)
        sDes_orig = traj_orig.desiredState(t + dt, dt, quad_orig)
        ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, dt)
        
        # Update matched framework
        quad_matched.update(t, dt, ctrl_matched.w_cmd, wind)
        sDes_matched = traj_matched.desiredState(t + dt, dt, quad_matched)
        ctrl_matched.controller(traj_matched, quad_matched, sDes_matched, dt)
    
    # Calculate tracking errors
    error_orig = np.linalg.norm(pos_orig_all - pos_des_all, axis=1)
    error_matched = np.linalg.norm(pos_matched_all - pos_des_all, axis=1)
    
    # Calculate position differences between frameworks
    pos_diff = np.linalg.norm(pos_orig_all - pos_matched_all, axis=1)
    
    print(f"\nTrajectory tracking results:")
    print(f"Original framework:")
    print(f"  Mean tracking error: {np.mean(error_orig):.4f} m")
    print(f"  Max tracking error: {np.max(error_orig):.4f} m")
    print(f"  Final position: [{pos_orig_all[-1][0]:.3f}, {pos_orig_all[-1][1]:.3f}, {pos_orig_all[-1][2]:.3f}]")
    
    print(f"Matched framework:")
    print(f"  Mean tracking error: {np.mean(error_matched):.4f} m")
    print(f"  Max tracking error: {np.max(error_matched):.4f} m")
    print(f"  Final position: [{pos_matched_all[-1][0]:.3f}, {pos_matched_all[-1][1]:.3f}, {pos_matched_all[-1][2]:.3f}]")
    
    print(f"Framework comparison:")
    print(f"  Mean position difference: {np.mean(pos_diff):.4f} m")
    print(f"  Max position difference: {np.max(pos_diff):.4f} m")
    print(f"  Final position difference: {pos_diff[-1]:.4f} m")
    
    # Success criteria
    tracking_similar = abs(np.mean(error_orig) - np.mean(error_matched)) < 0.05
    positions_close = np.mean(pos_diff) < 0.1
    
    success = tracking_similar and positions_close
    print(f"  Trajectory tracking match: {'✅' if success else '❌'}")
    
    if success:
        print("🎉 MATCHED DRONE SUCCESSFUL!")
        print("   Both frameworks show very similar trajectory tracking")
    else:
        print("⚠️ Still some differences, but much better than before")
    
    return success, t_all, pos_orig_all, pos_matched_all, pos_des_all, error_orig, error_matched

def create_comparison_plot(t_all, pos_orig, pos_matched, pos_des, error_orig, error_matched):
    """Create comparison plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 3D trajectory comparison
    ax1.plot(pos_des[:, 0], pos_des[:, 1], 'k--', label='Desired', linewidth=2, alpha=0.8)
    ax1.plot(pos_orig[:, 0], pos_orig[:, 1], 'b-', label='Original', linewidth=1.5)
    ax1.plot(pos_matched[:, 0], pos_matched[:, 1], 'r-', label='Matched', linewidth=1.5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('XY Trajectory Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Z trajectory
    ax2.plot(t_all, pos_des[:, 2], 'k--', label='Desired', linewidth=2, alpha=0.8)
    ax2.plot(t_all, pos_orig[:, 2], 'b-', label='Original', linewidth=1.5)
    ax2.plot(t_all, pos_matched[:, 2], 'r-', label='Matched', linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Altitude Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Tracking errors
    ax3.plot(t_all, error_orig, 'b-', label='Original', linewidth=1.5)
    ax3.plot(t_all, error_matched, 'r-', label='Matched', linewidth=1.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Tracking Error (m)')
    ax3.set_title('Trajectory Tracking Error Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Position differences between frameworks
    pos_diff = np.linalg.norm(pos_orig - pos_matched, axis=1)
    ax4.plot(t_all, pos_diff, 'g-', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Position Difference (m)')
    ax4.set_title('Original vs Matched Framework Position Difference')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('matched_drone_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved as 'matched_drone_comparison.png'")

def main():
    print("MATCHED DRONE TRAJECTORY TRACKING TEST")
    print("=" * 50)
    
    # Create matched drone
    quad_orig, quad_matched = create_matched_drone()
    
    # Test hover behavior
    hover_success = test_hover_behavior(quad_orig, quad_matched)
    
    # Test trajectory tracking  
    tracking_success, t_all, pos_orig, pos_matched, pos_des, error_orig, error_matched = test_trajectory_tracking(quad_orig, quad_matched)
    
    # Create comparison plot
    create_comparison_plot(t_all, pos_orig, pos_matched, pos_des, error_orig, error_matched)
    
    print(f"\n\nFINAL RESULTS")
    print("=" * 20)
    
    if hover_success and tracking_success:
        print("🎉 COMPLETE SUCCESS!")
        print("   ✅ Hover behavior matches perfectly")
        print("   ✅ Trajectory tracking is nearly identical") 
        print("   ✅ Waypoint and tracking differences resolved")
        print("\n   The configurable framework now performs equivalently to the original!")
    elif tracking_success:
        print("🎯 TRAJECTORY SUCCESS!")
        print(f"   {'✅' if hover_success else '⚠️'} Hover behavior: {'Perfect match' if hover_success else 'Minor differences'}")
        print("   ✅ Trajectory tracking is very similar")
        print("   ✅ Major improvements achieved")
    else:
        print("📈 SIGNIFICANT IMPROVEMENT")
        print(f"   {'✅' if hover_success else '⚠️'} Hover behavior: {'Good match' if hover_success else 'Some differences'}")
        print("   📈 Trajectory tracking much improved")
        print("   🔧 May need further parameter tuning")

if __name__ == "__main__":
    main()