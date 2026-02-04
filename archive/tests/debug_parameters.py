#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Comparison Script

Compare parameters between original and generalized systems.
"""

import numpy as np
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from propeller_data import create_standard_propeller_config

def compare_parameters():
    """Compare parameters between systems."""
    
    print("Parameter Comparison")
    print("="*50)
    
    # Create both systems
    quad_orig = OriginalQuadcopter(0)
    
    propeller_config = create_standard_propeller_config(
        config_type="quad",
        arm_length=0.175,
        prop_size="matched"
    )
    quad_gen = ConfigurableQuadcopter(0, propellers=propeller_config)
    
    print("Original System Parameters:")
    print(f"  Mass (mB): {quad_orig.params['mB']:.6f} kg")
    print(f"  Gravity (g): {quad_orig.params['g']:.6f} m/s²")
    print(f"  Max Thrust: {quad_orig.params['maxThr']:.6f} N")
    print(f"  Min Thrust: {quad_orig.params['minThr']:.6f} N")
    
    if "IB" in quad_orig.params:
        IB_orig = quad_orig.params["IB"]
        print(f"  Inertia matrix IB:")
        print(f"    [{IB_orig[0,0]:.6f}, {IB_orig[0,1]:.6f}, {IB_orig[0,2]:.6f}]")
        print(f"    [{IB_orig[1,0]:.6f}, {IB_orig[1,1]:.6f}, {IB_orig[1,2]:.6f}]")
        print(f"    [{IB_orig[2,0]:.6f}, {IB_orig[2,1]:.6f}, {IB_orig[2,2]:.6f}]")
    
    print(f"\nGeneralized System Parameters:")
    print(f"  Mass (mB): {quad_gen.params['mB']:.6f} kg")
    print(f"  Gravity (g): {quad_gen.params['g']:.6f} m/s²")
    print(f"  Max Thrust: {quad_gen.params['maxThr']:.6f} N")
    print(f"  Min Thrust: {quad_gen.params['minThr']:.6f} N")
    
    if "IB" in quad_gen.params:
        IB_gen = quad_gen.params["IB"]
        print(f"  Inertia matrix IB:")
        print(f"    [{IB_gen[0,0]:.6f}, {IB_gen[0,1]:.6f}, {IB_gen[0,2]:.6f}]")
        print(f"    [{IB_gen[1,0]:.6f}, {IB_gen[1,1]:.6f}, {IB_gen[1,2]:.6f}]")
        print(f"    [{IB_gen[2,0]:.6f}, {IB_gen[2,1]:.6f}, {IB_gen[2,2]:.6f}]")
    
    print(f"\nParameter Differences:")
    mass_diff = abs(quad_orig.params['mB'] - quad_gen.params['mB'])
    thrust_diff = abs(quad_orig.params['maxThr'] - quad_gen.params['maxThr'])
    
    print(f"  Mass difference: {mass_diff:.6f} kg ({mass_diff/quad_orig.params['mB']*100:.2f}%)")
    print(f"  Max thrust difference: {thrust_diff:.6f} N ({thrust_diff/quad_orig.params['maxThr']*100:.2f}%)")
    
    if "IB" in quad_orig.params and "IB" in quad_gen.params:
        IB_diff = np.linalg.norm(quad_orig.params["IB"] - quad_gen.params["IB"])
        print(f"  Inertia matrix difference (Frobenius norm): {IB_diff:.6f}")
    
    # Compare specific propeller parameters
    print(f"\nPropeller Configuration:")
    print(f"  Original system: kTh={quad_orig.params.get('kTh', 'N/A'):.2e}")
    print(f"  Generalized system propeller details:")
    
    for i, prop in enumerate(propeller_config):
        if "constants" in prop:
            k_f, k_m = prop["constants"]
            print(f"    Motor {i+1}: k_f={k_f:.2e}, k_m={k_m:.2e}, wmax={prop.get('wmax', 'N/A')}")
            
    # Check what the drone object actually uses
    print(f"\nActual loaded values in generalized drone:")
    if hasattr(quad_gen, 'drone_sim') and hasattr(quad_gen.drone_sim, 'config'):
        for i, prop in enumerate(quad_gen.drone_sim.config.propellers):
            k_f, k_m = prop["constants"] 
            print(f"    Motor {i+1}: k_f={k_f:.2e}, k_m={k_m:.2e}, mass={prop.get('mass', 'N/A')}")

if __name__ == "__main__":
    compare_parameters()