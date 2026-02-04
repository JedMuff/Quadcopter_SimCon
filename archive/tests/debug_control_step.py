#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step-by-Step Control Analysis

This script analyzes the control pipeline step by step to identify
where the generalized controller differs from the original.

Author: Debug Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from trajectory import Trajectory
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from utils.windModel import Wind
import utils

def create_matched_drone(Ti=0):
    """Create drone that matches original quadcopter parameters."""
    propeller_config = create_standard_propeller_config(
        config_type="quad",
        arm_length=0.175,
        prop_size="matched"
    )
    return ConfigurableQuadcopter(Ti, propellers=propeller_config)

def analyze_single_control_step():
    """Analyze a single control computation step in detail."""
    
    print("Step-by-Step Control Analysis")
    print("="*50)
    
    # Initialize both systems
    Ti = 0
    quad_orig = OriginalQuadcopter(Ti)
    quad_gen = create_matched_drone(Ti)
    
    ctrl_orig = OriginalControl(quad_orig, yawType=1)
    ctrl_gen = GeneralizedControl(quad_gen, yawType=1)
    
    # Set up identical initial conditions
    # Move both quads to same position with small error from desired
    quad_orig.pos = np.array([0.1, 0.1, -0.1])  # Small offset from origin
    quad_orig.vel = np.array([0.05, 0.0, 0.02])  # Small velocity
    
    quad_gen.pos = quad_orig.pos.copy()
    quad_gen.vel = quad_orig.vel.copy()
    quad_gen.quat = quad_orig.quat.copy()
    quad_gen.omega = quad_orig.omega.copy()
    
    # Create simple trajectory
    trajSelect = np.array([5, 3, 1])  # minimum jerk, follow yaw, average speed
    ctrlType = "xyz_pos"
    traj = Trajectory(quad_orig, ctrlType, trajSelect)
    
    # Get desired state at t=1.0
    t = 1.0
    Ts = 0.01
    sDes_orig = traj.desiredState(t, Ts, quad_orig)
    sDes_gen = traj.desiredState(t, Ts, quad_gen)
    
    print(f"\nInitial Conditions:")
    print(f"Position: [{quad_orig.pos[0]:.3f}, {quad_orig.pos[1]:.3f}, {quad_orig.pos[2]:.3f}]")
    print(f"Velocity: [{quad_orig.vel[0]:.3f}, {quad_orig.vel[1]:.3f}, {quad_orig.vel[2]:.3f}]")
    print(f"Desired:  [{traj.sDes[0]:.3f}, {traj.sDes[1]:.3f}, {traj.sDes[2]:.3f}]")
    
    # Run control computation
    print(f"\nRunning Control Computation...")
    
    # Original controller
    ctrl_orig.controller(traj, quad_orig, sDes_orig, Ts)
    # Generalized controller 
    ctrl_gen.controller(traj, quad_gen, sDes_gen, Ts)

    print(f"Control Inputs, Thrust: {ctrl_orig.thrust_sp}, Rate: {ctrl_orig.rateCtrl}")
    
    orig_w_cmd = utils.mixerFM(quad_orig, np.linalg.norm(ctrl_orig.thrust_sp), ctrl_orig.rateCtrl)
    print(f"Original Motor Commands: {orig_w_cmd}")

    gen_w_cmd = utils.mixerFM(quad_gen, np.linalg.norm(ctrl_gen.thrust_sp), ctrl_gen.rateCtrl)
    print(f"Generalized Motor Commands: {gen_w_cmd}")

    # Debug: Show what the original mixer actually gets
    # thrust_mag_orig = np.linalg.norm(ctrl_orig.thrust_sp)
    # orig_mixer_input = np.array([thrust_mag_orig, ctrl_orig.rateCtrl[0], ctrl_orig.rateCtrl[1], ctrl_orig.rateCtrl[2]])
    # orig_mixer_output = quad_orig.params["mixerFMinv"] @ orig_mixer_input
    # print(f"  Original mixer output: {orig_mixer_output}")
    # print(f"  Original mixer output 2:  {np.maximum(orig_mixer_output, 0)}")
    # orig_motors_calc = np.sqrt(np.maximum(orig_mixer_output, 0))
    
    # print(f"  Original mixer calc: input={orig_mixer_input}, motors={orig_motors_calc}")
    # print(f"  Original ctrl.w_cmd: {ctrl_orig.w_cmd}")


    
    # Compare intermediate values
    print(f"\n" + "="*60)
    print("CONTROL PIPELINE COMPARISON")
    print("="*60)
    
    # Position control
    print(f"\n1. Position Control:")
    print(f"   Position setpoint - Original:    [{ctrl_orig.pos_sp[0]:.3f}, {ctrl_orig.pos_sp[1]:.3f}, {ctrl_orig.pos_sp[2]:.3f}]")
    print(f"   Position setpoint - Generalized: [{ctrl_gen.pos_sp[0]:.3f}, {ctrl_gen.pos_sp[1]:.3f}, {ctrl_gen.pos_sp[2]:.3f}]")
    
    # Velocity control
    print(f"\n2. Velocity Control:")
    print(f"   Velocity setpoint - Original:    [{ctrl_orig.vel_sp[0]:.3f}, {ctrl_orig.vel_sp[1]:.3f}, {ctrl_orig.vel_sp[2]:.3f}]")
    print(f"   Velocity setpoint - Generalized: [{ctrl_gen.vel_sp[0]:.3f}, {ctrl_gen.vel_sp[1]:.3f}, {ctrl_gen.vel_sp[2]:.3f}]")
    
    # Thrust control
    print(f"\n3. Thrust Control:")
    print(f"   Thrust setpoint - Original:    [{ctrl_orig.thrust_sp[0]:.3f}, {ctrl_orig.thrust_sp[1]:.3f}, {ctrl_orig.thrust_sp[2]:.3f}]")
    print(f"   Thrust setpoint - Generalized: [{ctrl_gen.thrust_sp[0]:.3f}, {ctrl_gen.thrust_sp[1]:.3f}, {ctrl_gen.thrust_sp[2]:.3f}]")
    print(f"   Thrust magnitude - Original:    {np.linalg.norm(ctrl_orig.thrust_sp):.3f}N")
    print(f"   Thrust magnitude - Generalized: {np.linalg.norm(ctrl_gen.thrust_sp):.3f}N")
    
    # Attitude control
    print(f"\n4. Attitude Control:")
    print(f"   Rate setpoint - Original:    [{ctrl_orig.rate_sp[0]:.3f}, {ctrl_orig.rate_sp[1]:.3f}, {ctrl_orig.rate_sp[2]:.3f}] rad/s")
    print(f"   Rate setpoint - Generalized: [{ctrl_gen.rate_sp[0]:.3f}, {ctrl_gen.rate_sp[1]:.3f}, {ctrl_gen.rate_sp[2]:.3f}] rad/s")
    
    # Rate control
    print(f"\n5. Rate Control:")
    print(f"   Rate control output - Original:    [{ctrl_orig.rateCtrl[0]:.3f}, {ctrl_orig.rateCtrl[1]:.3f}, {ctrl_orig.rateCtrl[2]:.3f}] rad/s²")
    print(f"   Rate control output - Generalized: [{ctrl_gen.rateCtrl[0]:.3f}, {ctrl_gen.rateCtrl[1]:.3f}, {ctrl_gen.rateCtrl[2]:.3f}] rad/s²")
    
    # Show the moment conversion that we added
    # if hasattr(ctrl_gen, 'desired_moments'):
    #     print(f"   Desired moments - Generalized: [{ctrl_gen.desired_moments[0]:.6f}, {ctrl_gen.desired_moments[1]:.6f}, {ctrl_gen.desired_moments[2]:.6f}] N⋅m")
    
    # Motor commands
    print(f"\n6. Motor Commands:")
    print(f"   Motor speeds - Original:    [{ctrl_orig.w_cmd[0]:.1f}, {ctrl_orig.w_cmd[1]:.1f}, {ctrl_orig.w_cmd[2]:.1f}, {ctrl_orig.w_cmd[3]:.1f}] rad/s")
    print(f"   Motor speeds - Generalized: [{ctrl_gen.w_cmd[0]:.1f}, {ctrl_gen.w_cmd[1]:.1f}, {ctrl_gen.w_cmd[2]:.1f}, {ctrl_gen.w_cmd[3]:.1f}] rad/s")
    
    # Compute differences
    print(f"\n" + "="*60)
    print("DIFFERENCE ANALYSIS")
    print("="*60)
    
    thrust_diff = np.linalg.norm(ctrl_orig.thrust_sp - ctrl_gen.thrust_sp)
    rate_diff = np.linalg.norm(ctrl_orig.rate_sp - ctrl_gen.rate_sp)
    rateCtrl_diff = np.linalg.norm(ctrl_orig.rateCtrl - ctrl_gen.rateCtrl)
    motor_diff = np.linalg.norm(ctrl_orig.w_cmd[:4] - ctrl_gen.w_cmd[:4])
    
    print(f"Thrust setpoint difference:     {thrust_diff:.6f}N")
    print(f"Rate setpoint difference:       {rate_diff:.6f} rad/s")
    print(f"Rate control output difference: {rateCtrl_diff:.6f} rad/s²")
    print(f"Motor command difference:       {motor_diff:.1f} rad/s")
    
def compare_allocation_matrices():
    """Compare the allocation matrices between systems."""
    
    print("\n" + "="*60)  
    print("ALLOCATION MATRIX COMPARISON")
    print("="*60)
    
    # Original system parameters
    quad_orig = OriginalQuadcopter(0)
    
    # Generalized system
    quad_gen = create_matched_drone(0)
    quad_gen.force_original_parameters()
    
    print("Original system mixer matrix (mixerFMinv):")
    mixer_orig = quad_orig.params["mixerFMinv"]
    print(f"Shape: {mixer_orig.shape}")
    print(mixer_orig)

    print("\nGeneralized system mixer matrix (mixerFMinv):")
    mixer_gen = quad_gen.params["mixerFMinv"]
    print(f"Shape: {mixer_gen.shape}")
    print(mixer_gen)

    print(f"\nGeneralized system allocation matrices:")
    Bf, Bm = quad_gen.drone_sim.config.get_allocation_matrices()
    B_combined, B_pinv = quad_gen.drone_sim.config.get_control_allocation()
    
    print(f"Force allocation (Bf) shape: {Bf.shape}")
    print(Bf)
    print(f"\nMoment allocation (Bm) shape: {Bm.shape}")  
    print(Bm)
    print(f"\nCombined allocation (B_combined) shape: {B_combined.shape}")
    print(B_combined)
    print(f"\nPseudo-inverse (B_pinv) shape: {B_pinv.shape}")
    print(B_pinv)

if __name__ == "__main__":
    analyze_single_control_step()
    compare_allocation_matrices()