#!/usr/bin/env python3
"""
Debug Controller Integration Issue

Investigate why the configurable framework fails with trajectory tracking
while working fine with direct motor commands.
"""

import numpy as np
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter
from trajectory import Trajectory
from ctrl import Control

def debug_controller_integration():
    """Debug the integration issue between configurable framework and controllers."""
    print("CONTROLLER INTEGRATION DEBUG")
    print("=" * 35)
    
    # Create both frameworks
    Ti = 0
    Ts = 0.005
    
    quad_orig = OriginalQuadcopter(Ti)
    quad_config = ConfigurableQuadcopter(Ti, drone_type="quad", arm_length=0.11, prop_size=5)
    
    # Create trajectories and controllers
    trajSelect = np.array([0, 3, 1])  # HOVER trajectory to simplify debugging
    ctrlType = "xyz_pos"
    
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    traj_config = Trajectory(quad_config, ctrlType, trajSelect)
    
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    ctrl_config = Control(quad_config, traj_config.yawType)
    
    print("Framework comparison:")
    print(f"Original - Mass: {quad_orig.params['mB']:.3f} kg, kTh: {quad_orig.params['kTh']:.2e}")
    print(f"Config   - Mass: {quad_config.params['mB']:.3f} kg, kTh: {quad_config.params['kTh']:.2e}")
    
    # Test initial states
    print(f"\nInitial states:")
    print(f"Original position: {quad_orig.pos}")
    print(f"Config position: {quad_config.pos}")
    
    # Get desired states for hover (should be [0,0,0])
    sDes_orig = traj_orig.desiredState(0, Ts, quad_orig)
    sDes_config = traj_config.desiredState(0, Ts, quad_config)
    
    print(f"\nDesired states (should be same):")
    print(f"Original: pos={sDes_orig[:3]}, vel={sDes_orig[3:6]}")
    print(f"Config:   pos={sDes_config[:3]}, vel={sDes_config[3:6]}")
    
    # Generate control commands
    ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
    ctrl_config.controller(traj_config, quad_config, sDes_config, Ts)
    
    print(f"\nControl commands:")
    print(f"Original w_cmd: {ctrl_orig.w_cmd}")
    print(f"Config w_cmd: {ctrl_config.w_cmd}")
    
    # Test one simulation step
    print(f"\nAfter one simulation step:")
    
    # Store initial positions
    init_pos_orig = quad_orig.pos.copy()
    init_pos_config = quad_config.pos.copy()
    
    # Update dynamics
    from utils.windModel import Wind
    wind = Wind('None', 2.0, 90, -15)
    
    quad_orig.update(0, Ts, ctrl_orig.w_cmd, wind)
    quad_config.update(0, Ts, ctrl_config.w_cmd, wind)
    
    print(f"Original: {init_pos_orig} → {quad_orig.pos} (Δ={quad_orig.pos - init_pos_orig})")
    print(f"Config:   {init_pos_config} → {quad_config.pos} (Δ={quad_config.pos - init_pos_config})")
    
    # Check if one framework is falling while other hovers
    pos_change_orig = np.linalg.norm(quad_orig.pos - init_pos_orig)
    pos_change_config = np.linalg.norm(quad_config.pos - init_pos_config)
    
    print(f"\nPosition change magnitudes:")
    print(f"Original: {pos_change_orig:.6f} m")
    print(f"Config:   {pos_change_config:.6f} m")
    
    if pos_change_config > 10 * pos_change_orig:
        print("❌ ISSUE: Configurable framework showing excessive movement")
        print("   This suggests the quadratic fix is causing problems with control integration")
        return False
    else:
        print("✓ Position changes are reasonable")
        return True

def test_direct_vs_controller_commands():
    """Test difference between direct motor commands and controller-generated commands."""
    print(f"\n\nDIRECT VS CONTROLLER COMMAND TEST")
    print("=" * 40)
    
    quad_config = ConfigurableQuadcopter(0, drone_type="quad", arm_length=0.11, prop_size=5)
    
    # Test 1: Direct hover commands (this works)
    w_hover = quad_config.params["w_hover"]
    direct_cmds = np.ones(4) * w_hover
    
    print(f"Direct hover commands: {direct_cmds[0]:.1f} rad/s")
    
    # Store initial state
    initial_pos = quad_config.pos.copy()
    
    # Apply direct commands
    quad_config.update(0, 0.005, direct_cmds, None)
    
    pos_change_direct = np.linalg.norm(quad_config.pos - initial_pos)
    print(f"Position change with direct commands: {pos_change_direct:.6f} m")
    
    # Reset quad
    quad_config = ConfigurableQuadcopter(0, drone_type="quad", arm_length=0.11, prop_size=5)
    
    # Test 2: Controller-generated commands
    trajSelect = np.array([0, 3, 1])  # Hover
    traj = Trajectory(quad_config, "xyz_pos", trajSelect)
    ctrl = Control(quad_config, traj.yawType)
    
    sDes = traj.desiredState(0, 0.005, quad_config)
    ctrl.controller(traj, quad_config, sDes, 0.005)
    
    print(f"Controller commands: {ctrl.w_cmd}")
    
    # Store initial state
    initial_pos = quad_config.pos.copy()
    
    # Apply controller commands
    quad_config.update(0, 0.005, ctrl.w_cmd, None)
    
    pos_change_ctrl = np.linalg.norm(quad_config.pos - initial_pos)
    print(f"Position change with controller commands: {pos_change_ctrl:.6f} m")
    
    # Compare
    if pos_change_ctrl > 10 * pos_change_direct:
        print("❌ ISSUE: Controller commands cause excessive movement")
        print("   The problem is in the control system integration")
        return False
    else:
        print("✓ Both command types produce reasonable movement")
        return True

def main():
    print("CONTROLLER INTEGRATION DEBUGGING")
    print("=" * 45)
    
    # Debug 1: Controller integration
    integration_ok = debug_controller_integration()
    
    # Debug 2: Direct vs controller commands
    commands_ok = test_direct_vs_controller_commands()
    
    print(f"\n\nDEBUG SUMMARY")
    print("=" * 20)
    
    if integration_ok and commands_ok:
        print("✅ Controller integration appears to be working")
        print("   Issue may be elsewhere - need further investigation")
    else:
        print("❌ CONTROLLER INTEGRATION PROBLEM IDENTIFIED")
        print("   The quadratic fix is interfering with existing control system")
        print("   Need to revise the fix to maintain controller compatibility")

if __name__ == "__main__":
    main()