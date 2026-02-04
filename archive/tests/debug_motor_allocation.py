#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Motor Allocation Debugging Script

This script identifies why we're getting negative motor speeds squared and
significant trajectory divergence despite matching all parameters.
"""

import numpy as np
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from trajectory import Trajectory
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from utils.windModel import Wind
import utils

def debug_motor_allocation():
    """Debug motor allocation differences between systems."""
    
    print("="*70)
    print("MOTOR ALLOCATION DEBUGGING")
    print("="*70)
    
    # Create both systems
    print("\n1. Creating systems...")
    orig_quad = OriginalQuadcopter(0)
    
    # Create configurable with exact matching
    orig_temp = OriginalQuadcopter(0)
    arm_length = orig_temp.params["dxm"]
    
    propeller_config = create_standard_propeller_config(
        config_type="quad",
        arm_length=arm_length,
        prop_size="matched"
    )
    config_quad = ConfigurableQuadcopter(0, propellers=propeller_config)
    config_quad.force_original_parameters()
    
    # Create controllers
    orig_ctrl = OriginalControl(orig_quad, yawType=1)
    config_ctrl = GeneralizedControl(config_quad, yawType=1)
    
    print("\n2. Comparing mixer matrices...")
    print("Original mixer:")
    print(orig_quad.params["mixerFM"])
    print("\nConfigurable mixer:")
    print(config_quad.params["mixerFM"])
    print(f"\nMixer difference: {np.linalg.norm(orig_quad.params['mixerFM'] - config_quad.params['mixerFM'])}")
    
    print("\n3. Testing control allocation with identical inputs...")
    
    # Test with simple hover command
    test_thrust = 11.772  # Hover thrust
    test_moments = np.array([0.0, 0.0, 0.0])  # No moments
    
    print(f"\nTest input: Thrust={test_thrust:.3f}N, Moments={test_moments}")
    
    # Original system allocation
    orig_input = np.array([test_thrust, test_moments[0], test_moments[1], test_moments[2]])
    orig_motors_squared = orig_quad.params["mixerFMinv"] @ orig_input
    orig_motors = np.sqrt(np.abs(orig_motors_squared)) * np.sign(orig_motors_squared)
    
    print(f"\nOriginal system:")
    print(f"  Motor speeds squared: {orig_motors_squared}")
    print(f"  Motor speeds: {orig_motors}")
    print(f"  All positive: {np.all(orig_motors_squared >= 0)}")
    
    # Configurable system allocation  
    config_input = np.array([test_thrust, test_moments[0], test_moments[1], test_moments[2]])
    config_motors_squared = config_quad.params["mixerFMinv"] @ config_input
    config_motors = np.sqrt(np.abs(config_motors_squared)) * np.sign(config_motors_squared)
    
    print(f"\nConfigurable system:")
    print(f"  Motor speeds squared: {config_motors_squared}")
    print(f"  Motor speeds: {config_motors}")
    print(f"  All positive: {np.all(config_motors_squared >= 0)}")
    
    print("\n4. Testing with realistic control scenario...")
    
    # Create a trajectory and test at a specific time
    trajSelect = np.zeros(3)
    trajSelect[0] = 5  # minimum jerk trajectory
    trajSelect[1] = 3  # follow yaw
    trajSelect[2] = 1  # use average speed
    
    ctrlType = "xyz_pos"
    traj = Trajectory(orig_quad, ctrlType, trajSelect)
    
    # Test at t=1.0 seconds
    t_test = 1.0
    Ts = 0.01
    
    # Get desired state
    sDes_orig = traj.desiredState(t_test, Ts, orig_quad)
    sDes_config = traj.desiredState(t_test, Ts, config_quad)
    
    print(f"\nDesired state at t={t_test}s:")
    print(f"  Position: {sDes_orig[0:3]}")
    print(f"  Velocity: {sDes_orig[6:9]}")
    print(f"  Yaw: {sDes_orig[9]}")
    
    # Run control for both systems
    orig_ctrl.controller(traj, orig_quad, sDes_orig, Ts)
    config_ctrl.controller(traj, config_quad, sDes_config, Ts)
    
    print(f"\nControl outputs:")
    print(f"Original:")
    print(f"  Thrust setpoint: {orig_ctrl.thrust_sp}")
    print(f"  Rate setpoint: {orig_ctrl.rate_sp}")
    print(f"  Motor commands: {orig_ctrl.w_cmd[:4]}")
    
    print(f"Configurable:")
    print(f"  Thrust setpoint: {config_ctrl.thrust_sp}")
    print(f"  Rate setpoint: {config_ctrl.rate_sp}")
    print(f"  Motor commands: {config_ctrl.w_cmd[:4]}")
    
    # Check for differences in control outputs
    thrust_diff = np.linalg.norm(orig_ctrl.thrust_sp - config_ctrl.thrust_sp)
    rate_diff = np.linalg.norm(orig_ctrl.rate_sp - config_ctrl.rate_sp)
    motor_diff = np.linalg.norm(orig_ctrl.w_cmd[:4] - config_ctrl.w_cmd[:4])
    
    print(f"\nControl differences:")
    print(f"  Thrust difference: {thrust_diff:.6f}")
    print(f"  Rate difference: {rate_diff:.6f}")
    print(f"  Motor command difference: {motor_diff:.6f}")
    
    print("\n5. Analyzing allocation matrix vs mixer consistency...")
    
    # Test if the allocation matrices in configurable system produce same results as mixer
    test_cmd_normalized = np.array([0.5, 0.5, 0.5, 0.5])  # Normalized commands [0,1]
    
    # Using allocation matrices (configurable system approach)
    forces_alloc = config_quad.drone_sim.Bf @ (test_cmd_normalized**2)
    moments_alloc = config_quad.drone_sim.Bm @ (test_cmd_normalized**2)
    
    print(f"\nTest with normalized commands {test_cmd_normalized}:")
    print(f"Allocation matrices result:")
    print(f"  Forces: {forces_alloc}")
    print(f"  Moments: {moments_alloc}")
    
    # Convert to motor speeds and use mixer (original system approach)
    max_w = config_quad.params["maxWmotor"]
    motor_speeds = test_cmd_normalized * max_w
    motor_speeds_squared = motor_speeds**2
    
    kTh = config_quad.params["kTh"]
    kTo = config_quad.params["kTo"]
    dxm = config_quad.params["dxm"]
    dym = config_quad.params["dym"]
    
    # Manual calculation using original method
    thrust_manual = kTh * np.sum(motor_speeds_squared)
    roll_moment_manual = kTh * dym * (motor_speeds_squared[0] - motor_speeds_squared[1] - motor_speeds_squared[2] + motor_speeds_squared[3])
    pitch_moment_manual = kTh * dxm * (motor_speeds_squared[0] + motor_speeds_squared[1] - motor_speeds_squared[2] - motor_speeds_squared[3])
    yaw_moment_manual = kTo * (-motor_speeds_squared[0] + motor_speeds_squared[1] - motor_speeds_squared[2] + motor_speeds_squared[3])
    
    print(f"Manual calculation result:")
    print(f"  Thrust: {thrust_manual}")
    print(f"  Roll moment: {roll_moment_manual}")
    print(f"  Pitch moment: {pitch_moment_manual}")
    print(f"  Yaw moment: {yaw_moment_manual}")
    
    # Check if they match
    force_z_diff = abs(-forces_alloc[2] - thrust_manual)  # Negative Z vs positive thrust
    moment_x_diff = abs(moments_alloc[0] - roll_moment_manual)
    moment_y_diff = abs(moments_alloc[1] - pitch_moment_manual)
    moment_z_diff = abs(moments_alloc[2] - yaw_moment_manual)
    
    print(f"\nConsistency check:")
    print(f"  Force Z difference: {force_z_diff:.2e}")
    print(f"  Moment X difference: {moment_x_diff:.2e}")
    print(f"  Moment Y difference: {moment_y_diff:.2e}")
    print(f"  Moment Z difference: {moment_z_diff:.2e}")
    
    if max(force_z_diff, moment_x_diff, moment_y_diff, moment_z_diff) > 1e-10:
        print("❌ CRITICAL: Allocation matrices do not match mixer behavior!")
    else:
        print("✅ Allocation matrices match mixer behavior")
        
    print("\n6. Integration method analysis...")
    
    # Test single integration step
    state_orig = orig_quad.state.copy()
    state_config = config_quad.drone_sim.state.copy()
    
    print(f"Original state shape: {state_orig.shape}")
    print(f"Configurable state shape: {state_config.shape}")
    
    # Original uses dopri5, configurable uses RK4
    print(f"Original integration: scipy dopri5 with adaptive timestep")
    print(f"Configurable integration: RK4 with fixed timestep {config_quad.drone_sim.dt}")
    
    wind = Wind("None", 0, 0, 0)
    
    # Single step with same motor commands
    test_motors = np.array([400, 400, 400, 400])
    
    orig_quad.update(0, 0.01, test_motors, wind)
    config_quad.update(0, 0.01, test_motors, wind)
    
    pos_diff = np.linalg.norm(orig_quad.pos - config_quad.pos)
    vel_diff = np.linalg.norm(orig_quad.vel - config_quad.vel)
    
    print(f"\nSingle step differences with identical motor commands:")
    print(f"  Position difference: {pos_diff:.2e} m")
    print(f"  Velocity difference: {vel_diff:.2e} m/s")
    
    print("\n" + "="*70)
    print("SUMMARY OF REMAINING ISSUES")
    print("="*70)
    
    issues_found = []
    
    if np.any(config_motors_squared < 0):
        issues_found.append("❌ Negative motor speeds squared in allocation")
        
    if max(force_z_diff, moment_x_diff, moment_y_diff, moment_z_diff) > 1e-10:
        issues_found.append("❌ Allocation matrices don't match mixer")
        
    if pos_diff > 1e-6:
        issues_found.append("❌ Integration method differences")
        
    if motor_diff > 1e-6:
        issues_found.append("❌ Control output differences")
        
    if not issues_found:
        print("✅ No critical issues found - trajectory divergence may be due to:")
        print("   • Accumulation of small numerical differences")
        print("   • Motor dynamics differences (original has motor dynamics, configurable doesn't)")
        print("   • Quaternion vs Euler angle representation differences")
    else:
        print("Critical issues found:")
        for issue in issues_found:
            print(f"   {issue}")

if __name__ == "__main__":
    debug_motor_allocation()