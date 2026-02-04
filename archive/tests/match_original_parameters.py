#!/usr/bin/env python3
"""
Create Configurable Framework that Matches Original Parameters

Design a propeller configuration that produces identical parameters to the original framework.
"""

import numpy as np
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter
# import utils.LoadConfiguration

def analyze_original_parameters():
    """Analyze the original framework parameters."""
    print("ORIGINAL FRAMEWORK PARAMETERS")
    print("=" * 35)
    
    quad_orig = OriginalQuadcopter(0)
    params = quad_orig.params
    
    print(f"Physical properties:")
    print(f"  Mass (mB): {params['mB']:.6f} kg")
    print(f"  kTh: {params['kTh']:.6e}")
    print(f"  kTo: {params['kTo']:.6e}")
    print(f"  w_hover: {params['w_hover']:.2f} rad/s")
    print(f"  thr_hover: {params['thr_hover']:.6f} N")
    
    print(f"\nGeometric properties:")
    print(f"  dxm: {params['dxm']:.6f} m")
    print(f"  dym: {params['dym']:.6f} m")
    
    print(f"\nMixer matrix (first column):")
    print(f"  mixerFM[:, 0]: {params['mixerFM'][:, 0]}")
    
    print(f"\nInertia:")
    print(f"  IB:\n{params['IB']}")
    
    return params

def create_matching_configurable():
    """Create configurable framework that matches original parameters."""
    print(f"\n\nCREATING MATCHING CONFIGURABLE FRAMEWORK")
    print("=" * 50)
    
    # Get original parameters
    quad_orig = OriginalQuadcopter(0)
    orig_params = quad_orig.params
    
    # Design propellers to match original parameters
    target_mass = orig_params['mB']
    target_kTh = orig_params['kTh']
    target_arm_length = orig_params['dxm']  # Use arm length from original
    
    print(f"Target parameters:")
    print(f"  Mass: {target_mass:.6f} kg")
    print(f"  kTh: {target_kTh:.6e}")
    print(f"  Arm length: {target_arm_length:.6f} m")
    
    # Create a configurable drone that tries to match these parameters
    # We'll need to reverse-engineer the propeller specifications
    
    # Try different propeller sizes to find one that produces similar kTh
    from propeller_data import get_propeller_specs
    
    best_match = None
    best_error = float('inf')
    
    for prop_size in [4, 5, 6, 7, 8]:
        specs = get_propeller_specs(prop_size)
        k_f = specs["constants"][0]
        
        error = abs(k_f - target_kTh)
        print(f"  Prop size {prop_size}: k_f = {k_f:.6e}, error = {error:.6e}")
        
        if error < best_error:
            best_error = error
            best_match = prop_size
    
    print(f"Best matching propeller size: {best_match}")
    
    # Create configurable drone with matching propeller
    propellers = [
        {"loc": [target_arm_length, target_arm_length, 0], "dir": [0, 0, -1, "ccw"], "propsize": best_match},
        {"loc": [-target_arm_length, target_arm_length, 0], "dir": [0, 0, -1, "cw"], "propsize": best_match},
        {"loc": [-target_arm_length, -target_arm_length, 0], "dir": [0, 0, -1, "ccw"], "propsize": best_match},
        {"loc": [target_arm_length, -target_arm_length, 0], "dir": [0, 0, -1, "cw"], "propsize": best_match}
    ]
    
    quad_config = ConfigurableQuadcopter(0, propellers=propellers)
    config_params = quad_config.params
    
    print(f"\nConfigurable framework results:")
    print(f"  Mass: {config_params['mB']:.6f} kg (target: {target_mass:.6f})")
    print(f"  kTh: {config_params['kTh']:.6e} (target: {target_kTh:.6e})")
    print(f"  w_hover: {config_params['w_hover']:.2f} rad/s (orig: {orig_params['w_hover']:.2f})")
    
    # Compare key parameters
    mass_match = abs(config_params['mB'] - target_mass) < 0.1
    kth_match = abs(config_params['kTh'] - target_kTh) / target_kTh < 0.1
    
    print(f"\nParameter matching:")
    print(f"  Mass match: {'✅' if mass_match else '❌'}")
    print(f"  kTh match: {'✅' if kth_match else '❌'}")
    
    return quad_config, mass_match and kth_match

def test_matched_frameworks():
    """Test if matched frameworks behave similarly."""
    print(f"\n\nTESTING MATCHED FRAMEWORKS")
    print("=" * 35)
    
    quad_orig = OriginalQuadcopter(0)
    
    # Create the best matching configurable framework
    quad_config, params_match = create_matching_configurable()
    
    if not params_match:
        print("❌ Could not create matching parameters - trajectory differences expected")
        return False
    
    # Test hover behavior
    from trajectory import Trajectory
    from ctrl import Control
    
    trajSelect = np.array([0, 3, 1])  # Hover
    
    traj_orig = Trajectory(quad_orig, "xyz_pos", trajSelect)
    traj_config = Trajectory(quad_config, "xyz_pos", trajSelect)
    
    ctrl_orig = Control(quad_orig, traj_orig.yawType)
    ctrl_config = Control(quad_config, traj_config.yawType)
    
    # Generate hover commands
    sDes = traj_orig.desiredState(0, 0.005, quad_orig)
    ctrl_orig.controller(traj_orig, quad_orig, sDes, 0.005)
    
    sDes = traj_config.desiredState(0, 0.005, quad_config)
    ctrl_config.controller(traj_config, quad_config, sDes, 0.005)
    
    print(f"Hover commands:")
    print(f"  Original: {ctrl_orig.w_cmd[0]:.2f} rad/s")
    print(f"  Configurable: {ctrl_config.w_cmd[0]:.2f} rad/s")
    
    # Test one simulation step
    from utils.windModel import Wind
    wind = Wind('None', 2.0, 90, -15)
    
    quad_orig.update(0, 0.005, ctrl_orig.w_cmd, wind)
    quad_config.update(0, 0.005, ctrl_config.w_cmd, wind)
    
    print(f"Position after hover step:")
    print(f"  Original: {quad_orig.pos}")
    print(f"  Configurable: {quad_config.pos}")
    
    pos_diff = np.linalg.norm(quad_orig.pos - quad_config.pos)
    print(f"Position difference: {pos_diff:.8f} m")
    
    similar_behavior = pos_diff < 1e-6
    print(f"Similar behavior: {'✅' if similar_behavior else '❌'}")
    
    return similar_behavior

def main():
    print("PARAMETER MATCHING ANALYSIS")
    print("=" * 40)
    
    # Analyze original parameters
    orig_params = analyze_original_parameters()
    
    # Test matched frameworks
    success = test_matched_frameworks()
    
    print(f"\n\nCONCLUSION")
    print("=" * 15)
    
    if success:
        print("✅ PARAMETER MATCHING SUCCESSFUL")
        print("   Configurable framework can match original behavior")
        print("   Use matched parameters for trajectory tracking")
    else:
        print("⚠️  PARAMETER MATCHING CHALLENGES")
        print("   Difficult to exactly match original framework parameters")
        print("   May need to adjust propeller specifications or configuration")

if __name__ == "__main__":
    main()