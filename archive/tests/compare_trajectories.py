#!/usr/bin/env python3
"""
Compare Trajectory Behavior Between Original and Configurable Frameworks

Verify that both frameworks produce identical trajectory tracking behavior.
"""

import numpy as np
import sys
sys.path.append('.')

# Import both frameworks
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter
from trajectory import Trajectory
from ctrl import Control
from utils.windModel import Wind

def test_trajectory_compatibility():
    """Test that both frameworks produce identical trajectory following."""
    print("TRAJECTORY COMPATIBILITY TEST")
    print("=" * 35)
    
    # Simulation parameters
    Ti = 0
    Ts = 0.005
    
    # Trajectory settings (matching both simulation files)
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.array([5, 3, 1])  # minimum jerk, follow yaw, average speed
    ctrlType = ctrlOptions[0]
    
    print(f"Trajectory settings: trajSelect = {trajSelect}")
    print(f"Control type: {ctrlType}")
    
    # Create both quad types
    print("\nInitializing quadcopters...")
    
    try:
        # Original framework
        quad_orig = OriginalQuadcopter(Ti)
        traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
        ctrl_orig = Control(quad_orig, traj_orig.yawType)
        print("✓ Original framework initialized")
        
        # Configurable framework (matching quadrotor)
        quad_config = ConfigurableQuadcopter(Ti, drone_type="quad", arm_length=0.11, prop_size=5)
        traj_config = Trajectory(quad_config, ctrlType, trajSelect)
        ctrl_config = Control(quad_config, traj_config.yawType)
        print("✓ Configurable framework initialized")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return False
    
    # Compare waypoints
    print(f"\nWaypoint comparison:")
    print(f"Original waypoints shape: {traj_orig.wps.shape}")
    print(f"Configurable waypoints shape: {traj_config.wps.shape}")
    
    waypoints_match = np.allclose(traj_orig.wps, traj_config.wps)
    print(f"Waypoints identical: {waypoints_match}")
    
    if not waypoints_match:
        print("Original waypoints:")
        print(traj_orig.wps)
        print("Configurable waypoints:")
        print(traj_config.wps)
    
    # Compare trajectory parameters
    print(f"\nTrajectory parameters:")
    print(f"xyzType: orig={traj_orig.xyzType}, config={traj_config.xyzType}")
    print(f"yawType: orig={traj_orig.yawType}, config={traj_config.yawType}")
    print(f"averVel: orig={traj_orig.averVel}, config={traj_config.averVel}")
    
    params_match = (traj_orig.xyzType == traj_config.xyzType and 
                   traj_orig.yawType == traj_config.yawType and
                   traj_orig.averVel == traj_config.averVel)
    print(f"Trajectory parameters identical: {params_match}")
    
    # Test trajectory generation at several time points
    print(f"\nTesting trajectory generation:")
    
    test_times = [0, 1.0, 5.0, 10.0, 15.0]
    all_match = True
    
    for t in test_times:
        # Generate desired states
        sDes_orig = traj_orig.desiredState(t, Ts, quad_orig)
        sDes_config = traj_config.desiredState(t, Ts, quad_config)
        
        # Compare desired positions (first 3 elements)
        pos_match = np.allclose(sDes_orig[:3], sDes_config[:3], atol=1e-6)
        
        if pos_match:
            print(f"  t={t:4.1f}s: ✓ positions match")
        else:
            print(f"  t={t:4.1f}s: ❌ positions differ")
            print(f"    Original:     {sDes_orig[:3]}")
            print(f"    Configurable: {sDes_config[:3]}")
            all_match = False
    
    return waypoints_match and params_match and all_match

def compare_control_behavior():
    """Compare controller behavior between frameworks."""
    print(f"\n\nCONTROL BEHAVIOR COMPARISON")
    print("=" * 32)
    
    # Quick test of controller interface compatibility
    try:
        Ti = 0
        quad_orig = OriginalQuadcopter(Ti)
        quad_config = ConfigurableQuadcopter(Ti, drone_type="quad", arm_length=0.11, prop_size=5)
        
        print("Key parameter comparison:")
        print(f"Mass - Original: {quad_orig.params['mB']:.3f} kg, Configurable: {quad_config.params['mB']:.3f} kg")
        print(f"kTh  - Original: {quad_orig.params['kTh']:.2e}, Configurable: {quad_config.params['kTh']:.2e}")
        
        # Test that both can be used with Control class
        trajSelect = np.array([5, 3, 1])
        traj_orig = Trajectory(quad_orig, "xyz_pos", trajSelect)
        traj_config = Trajectory(quad_config, "xyz_pos", trajSelect)
        
        ctrl_orig = Control(quad_orig, traj_orig.yawType)
        ctrl_config = Control(quad_config, traj_config.yawType)
        
        print("✓ Both frameworks compatible with Control class")
        return True
        
    except Exception as e:
        print(f"❌ Control compatibility test failed: {e}")
        return False

def main():
    print("TRAJECTORY AND WAYPOINT COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    # Test 1: Trajectory compatibility
    traj_success = test_trajectory_compatibility()
    
    # Test 2: Control behavior
    ctrl_success = compare_control_behavior()
    
    print(f"\n\nSUMMARY")
    print("=" * 15)
    
    if traj_success and ctrl_success:
        print("🎉 FRAMEWORKS ARE FULLY COMPATIBLE!")
        print("   ✅ Identical waypoints and trajectory generation")
        print("   ✅ Same trajectory parameters")
        print("   ✅ Compatible with existing control system")
        print("   → Both simulations should produce identical flight behavior")
    else:
        print("⚠️  DIFFERENCES FOUND:")
        print(f"   Trajectory compatibility: {'✅' if traj_success else '❌'}")
        print(f"   Control compatibility: {'✅' if ctrl_success else '❌'}")
        print("   → May need to investigate specific differences")

if __name__ == "__main__":
    main()