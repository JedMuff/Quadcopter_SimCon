#!/usr/bin/env python3
"""
Debug parameter consistency during simulation to find root cause of divergence.
"""

import numpy as np
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from trajectory import Trajectory
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from drone_simulator import ConfigurableQuadcopter

def compare_parameters(quad_orig, quad_gen, label=""):
    """Compare parameters between original and generalized systems."""
    print(f"\n--- Parameter Comparison {label} ---")
    
    # Key parameters to check
    params_to_check = ['mB', 'IB', 'kTh', 'kTo', 'maxWmotor', 'minWmotor']
    
    mismatches = []
    for param in params_to_check:
        if param in quad_orig.params and param in quad_gen.params:
            orig_val = quad_orig.params[param]
            gen_val = quad_gen.params[param]
            
            if isinstance(orig_val, np.ndarray):
                diff = np.max(np.abs(orig_val - gen_val))
                if diff > 1e-10:
                    mismatches.append(f"{param}: max_diff={diff:.2e}")
                    print(f"  ⚠️  {param}: orig={orig_val.flatten()[:3]}, gen={gen_val.flatten()[:3]}, max_diff={diff:.2e}")
            else:
                diff = abs(orig_val - gen_val)
                if diff > 1e-10:
                    mismatches.append(f"{param}: diff={diff:.2e}")
                    print(f"  ⚠️  {param}: orig={orig_val:.6e}, gen={gen_val:.6e}, diff={diff:.2e}")
        else:
            print(f"  ❌ {param}: Missing in one system")
    
    if not mismatches:
        print("  ✅ All parameters match")
    
    return len(mismatches) == 0

def debug_simulation_step():
    """Debug a single simulation step to check parameter usage."""
    
    print("=" * 80)
    print("PARAMETER CONSISTENCY DURING SIMULATION DEBUG")
    print("=" * 80)
    
    # Create systems
    Ti = 0
    Ts = 0.005
    
    # Original system
    quad_orig = OriginalQuadcopter(Ti)
    ctrlType = "xyz_pos"
    trajSelect = np.array([5, 3, 1])
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    ctrl_orig = OriginalControl(quad_orig, traj_orig.yawType)
    
    # Generalized system with matched configuration
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    quad_gen = ConfigurableQuadcopter(Ti, propellers=propellers)
    traj_gen = Trajectory(quad_gen, ctrlType, trajSelect)
    ctrl_gen = GeneralizedControl(quad_gen, traj_gen.yawType)
    
    # Check initial parameters
    print("\n🔍 Initial parameter comparison:")
    match_initial = compare_parameters(quad_orig, quad_gen, "Initial")
    
    # Force original parameters
    print("\n🔧 Forcing original parameters...")
    quad_gen.force_original_parameters()
    
    # Check after forcing
    print("\n🔍 Post-forcing parameter comparison:")
    match_forced = compare_parameters(quad_orig, quad_gen, "After Forcing")
    
    # Run a few simulation steps and check parameters at each step
    times = [0.1, 0.5, 1.0, 2.0]
    
    for t in times:
        print(f"\n📍 Time: {t:.1f}s")
        
        # Get desired state for both systems
        sDes_orig = traj_orig.desiredState(t, Ts, quad_orig)
        sDes_gen = traj_gen.desiredState(t, Ts, quad_gen)
        
        # Run control step
        ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
        ctrl_gen.controller(traj_gen, quad_gen, sDes_gen, Ts)
        
        # Check if parameters are still matching
        match_during = compare_parameters(quad_orig, quad_gen, f"During t={t:.1f}s")
        
        # Check control outputs
        print(f"  Control outputs:")
        print(f"    Original thrust mag: {np.linalg.norm(ctrl_orig.thrust_sp):.3f}N")
        print(f"    Generalized thrust mag: {np.linalg.norm(ctrl_gen.thrust_sp):.3f}N")
        print(f"    Original rate ctrl: {ctrl_orig.rateCtrl}")
        print(f"    Generalized rate ctrl: {ctrl_gen.rateCtrl}")
        
        if hasattr(ctrl_gen, 'desired_moments'):
            moments_orig = quad_orig.params["IB"] @ ctrl_orig.rateCtrl
            moments_gen = ctrl_gen.desired_moments
            print(f"    Original moments: {moments_orig}")
            print(f"    Generalized moments: {moments_gen}")
            print(f"    Moment difference: {np.abs(moments_orig - moments_gen).max():.2e}")
        
        # Update quad states (simulate dynamics integration step)
        # Note: Not running full dynamics, just checking parameter consistency
        
        if not match_during:
            print(f"  ❌ Parameter mismatch detected at t={t:.1f}s!")
            break
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Initial parameters match: {match_initial}")
    print(f"Parameters match after forcing: {match_forced}")
    
if __name__ == "__main__":
    debug_simulation_step()