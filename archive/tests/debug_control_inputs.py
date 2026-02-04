#!/usr/bin/env python3
"""
Direct Control Input Comparison

Compares the inputs fed to the mixer by the original controller vs GeneralizedControl
to identify exactly where they diverge.
"""

import numpy as np
import matplotlib.pyplot as plt
from trajectory import Trajectory
from ctrl import Control
from generalized_ctrl import GeneralizedControl
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config

def run_control_step(quad, ctrl, traj, t, Ts):
    """Run a single control step and capture mixer inputs."""
    
    # Update quad state first
    sDes = traj.desiredState(t, Ts, quad)
    
    # Store state before control
    state_before = {
        'pos': quad.pos.copy(),
        'vel': quad.vel.copy(), 
        'quat': quad.quat.copy(),
        'omega': quad.omega.copy()
    }
    
    # Run controller
    ctrl.controller(traj, quad, sDes, Ts)
    
    # Capture control outputs
    if hasattr(ctrl, 'rateCtrl'):
        rate_ctrl = ctrl.rateCtrl.copy()
    else:
        rate_ctrl = np.zeros(3)
        
    if hasattr(ctrl, 'thrust_sp'):
        thrust_sp = ctrl.thrust_sp.copy()
        thrust_magnitude = np.linalg.norm(thrust_sp)
    else:
        thrust_sp = np.zeros(3)
        thrust_magnitude = 0
    
    # For GeneralizedControl, get the traditional mixer inputs
    if hasattr(ctrl, 'allocator'):
        # This is GeneralizedControl - use converted moments, not raw rate control
        if hasattr(ctrl, 'desired_moments'):
            moments = ctrl.desired_moments.copy()
        else:
            # Fallback: convert rate control to moments like original controller
            moments = quad.params["IB"] @ rate_ctrl
        wrench_input = np.array([thrust_magnitude, moments[0], moments[1], moments[2]])
        motor_speeds_sq = ctrl.allocator.allocate_traditional(wrench_input)
        control_type = "generalized"
    else:
        # This is original Control - we need to compute what would go to mixer
        # The original uses utils.mixer.mixerFM(quad, thrust_magnitude, moments)
        # Where moments come from rate control converted to moments
        moments = quad.params["IB"] @ rate_ctrl
        wrench_input = np.array([thrust_magnitude, moments[0], moments[1], moments[2]])
        
        # Apply original mixer
        import utils.mixer as mixer
        w_cmd = mixer.mixerFM(quad, thrust_magnitude, moments)
        motor_speeds_sq = w_cmd**2
        control_type = "original"
    
    return {
        'control_type': control_type,
        'state_before': state_before,
        'sDes': sDes.copy(),
        'thrust_sp': thrust_sp,
        'thrust_magnitude': thrust_magnitude,
        'rate_ctrl': rate_ctrl,
        'wrench_input': wrench_input,
        'motor_speeds_sq': motor_speeds_sq,
        'w_cmd': ctrl.w_cmd.copy()
    }

def compare_controllers():
    """Compare original vs generalized controller step by step."""
    
    print("=" * 80)
    print("CONTROL INPUT COMPARISON: Original vs Generalized")
    print("=" * 80)
    
    # Create matched quadcopters
    Ti = 0
    
    # Original system
    quad_orig = Quadcopter(Ti)
    
    # Matched configurable system  
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    quad_gen = ConfigurableQuadcopter(Ti, propellers=propellers)
    
    # Create trajectory and controllers
    ctrlType = "xyz_pos"
    trajSelect = np.array([5, 3, 1])  # minimum jerk, follow yaw, average speed
    
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    traj_gen = Trajectory(quad_gen, ctrlType, trajSelect)
    
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    ctrl_gen = GeneralizedControl(quad_gen, traj_gen.yawType)
    
    # Run comparison for several time steps
    Ts = 0.005
    times = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    results = []
    
    for t in times:
        print(f"\n--- Time: {t:.1f}s ---")
        
        # Run original controller
        orig_result = run_control_step(quad_orig, ctrl_orig, traj_orig, t, Ts)
        
        # Set generalized quad to same state as original
        quad_gen.pos = quad_orig.pos.copy()
        quad_gen.vel = quad_orig.vel.copy()
        quad_gen.quat = quad_orig.quat.copy() 
        quad_gen.omega = quad_orig.omega.copy()
        quad_gen.euler = quad_orig.euler.copy()
        
        # Run generalized controller 
        gen_result = run_control_step(quad_gen, ctrl_gen, traj_gen, t, Ts)
        
        # Compare inputs to mixer
        print(f"Original wrench input:    [{orig_result['wrench_input'][0]:.3f}, {orig_result['wrench_input'][1]:.3f}, {orig_result['wrench_input'][2]:.3f}, {orig_result['wrench_input'][3]:.3f}]")
        print(f"Generalized wrench input: [{gen_result['wrench_input'][0]:.3f}, {gen_result['wrench_input'][1]:.3f}, {gen_result['wrench_input'][2]:.3f}, {gen_result['wrench_input'][3]:.3f}]")
        
        # Check differences
        wrench_diff = gen_result['wrench_input'] - orig_result['wrench_input']
        print(f"Wrench difference:        [{wrench_diff[0]:.6f}, {wrench_diff[1]:.6f}, {wrench_diff[2]:.6f}, {wrench_diff[3]:.6f}]")
        
        # Compare motor outputs
        motor_diff = gen_result['motor_speeds_sq'] - orig_result['motor_speeds_sq']
        print(f"Motor speeds² difference: [{motor_diff[0]:.1f}, {motor_diff[1]:.1f}, {motor_diff[2]:.1f}, {motor_diff[3]:.1f}]")
        
        # Store for detailed analysis
        results.append({
            'time': t,
            'original': orig_result,
            'generalized': gen_result,
            'wrench_diff': wrench_diff,
            'motor_diff': motor_diff
        })
        
        # Check for significant differences
        if np.max(np.abs(wrench_diff)) > 1e-6:
            print(f"⚠️  SIGNIFICANT WRENCH DIFFERENCE at t={t:.1f}s: max={np.max(np.abs(wrench_diff)):.2e}")
            
            # Detailed breakdown
            print(f"  Thrust magnitude diff: {wrench_diff[0]:.6f}")
            print(f"  Rate control diff: [{wrench_diff[1]:.6f}, {wrench_diff[2]:.6f}, {wrench_diff[3]:.6f}]")
            
            # Check rate control components
            rate_diff = gen_result['rate_ctrl'] - orig_result['rate_ctrl']
            print(f"  Raw rate control diff: [{rate_diff[0]:.6f}, {rate_diff[1]:.6f}, {rate_diff[2]:.6f}]")
            
            # Check thrust components
            thrust_diff = gen_result['thrust_sp'] - orig_result['thrust_sp']
            print(f"  Thrust vector diff: [{thrust_diff[0]:.6f}, {thrust_diff[1]:.6f}, {thrust_diff[2]:.6f}]")
    
    return results

def analyze_parameter_differences():
    """Analyze parameter differences between original and generalized systems."""
    
    print("\n" + "=" * 80)
    print("PARAMETER COMPARISON")
    print("=" * 80)
    
    # Create both systems
    quad_orig = Quadcopter(0)
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    quad_gen = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Compare key parameters
    params_to_check = ["mB", "g", "IB", "kTh", "kTo", "dxm", "dym", "minWmotor", "maxWmotor", "minThr", "maxThr"]
    
    for param in params_to_check:
        if param in quad_orig.params and param in quad_gen.params:
            orig_val = quad_orig.params[param]
            gen_val = quad_gen.params[param]
            
            if isinstance(orig_val, np.ndarray):
                diff = np.max(np.abs(orig_val - gen_val))
                print(f"{param:12}: orig={orig_val.flatten()}, gen={gen_val.flatten()}, max_diff={diff:.2e}")
            else:
                diff = abs(orig_val - gen_val)
                print(f"{param:12}: orig={orig_val:.6e}, gen={gen_val:.6e}, diff={diff:.2e}")
                
            if diff > 1e-10:
                print(f"  ⚠️  PARAMETER MISMATCH: {param}")
    
    print(f"\nMixer matrix comparison:")
    print(f"Original mixer matrix shape: {quad_orig.params['mixerFM'].shape}")
    print(f"Generalized mixer matrix shape: {quad_gen.params['mixerFM'].shape}")
    
    mixer_diff = np.max(np.abs(quad_orig.params['mixerFM'] - quad_gen.params['mixerFM']))
    print(f"Max mixer matrix difference: {mixer_diff:.2e}")
    
    if mixer_diff > 1e-10:
        print("⚠️  MIXER MATRIX MISMATCH")
        print("Original:")
        print(quad_orig.params['mixerFM'])
        print("Generalized:")
        print(quad_gen.params['mixerFM'])

if __name__ == "__main__":
    # Run parameter comparison first
    analyze_parameter_differences()
    
    # Run control input comparison
    results = compare_controllers()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    max_wrench_diff = max([np.max(np.abs(r['wrench_diff'])) for r in results])
    max_motor_diff = max([np.max(np.abs(r['motor_diff'])) for r in results])
    
    print(f"Maximum wrench input difference: {max_wrench_diff:.2e}")
    print(f"Maximum motor output difference: {max_motor_diff:.1f}")
    
    if max_wrench_diff < 1e-6:
        print("✅ Controllers produce IDENTICAL mixer inputs")
    else:
        print("❌ Controllers produce DIFFERENT mixer inputs")
        print("   This explains the trajectory divergence!")