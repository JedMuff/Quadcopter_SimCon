#!/usr/bin/env python3
"""
Test 3: Allocation Matrix Issue

Test if the allocation matrices are producing the correct force scaling.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_allocation_matrix_scaling():
    """Test the allocation matrix force scaling."""
    print("Testing Allocation Matrix Scaling")
    print("-" * 35)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Get allocation matrices
    Bf, Bm = quad.drone_sim.config.get_allocation_matrices()
    
    print("Allocation Matrix Bf (forces):")
    print(Bf)
    print("\\nAllocation Matrix Bm (moments):")
    print(Bm)
    
    # Check what the allocation matrices expect for input
    print("\\n" + "="*50)
    print("ALLOCATION MATRIX ANALYSIS")
    print("="*50)
    
    # The allocation matrices should map motor commands [0,1] to forces in Newtons
    # Let's see what motor commands = [1, 1, 1, 1] produces
    full_motor_commands = np.ones(4)
    forces_at_full = Bf @ full_motor_commands
    moments_at_full = Bm @ full_motor_commands
    
    print(f"Motor commands [1, 1, 1, 1] produce:")
    print(f"  Forces: {forces_at_full}")
    print(f"  Moments: {moments_at_full}")
    print(f"  Total upward force: {-forces_at_full[2]:.3f} N")
    
    # What should [1, 1, 1, 1] produce? It should be max thrust!
    kTh = quad.params["kTh"]
    w_max = quad.drone_sim.config.propellers[0]["wmax"]
    max_thrust_per_motor = kTh * w_max**2
    expected_max_thrust = max_thrust_per_motor * 4
    
    print(f"\\nExpected max thrust at motor_commands=[1,1,1,1]:")
    print(f"  kTh: {kTh:.2e}")
    print(f"  w_max: {w_max:.1f} rad/s")
    print(f"  Max thrust per motor: {max_thrust_per_motor:.3f} N")
    print(f"  Expected total max thrust: {expected_max_thrust:.3f} N")
    print(f"  Actual from allocation: {-forces_at_full[2]:.3f} N")
    print(f"  Ratio (actual/expected): {-forces_at_full[2]/expected_max_thrust:.3f}")
    
    # Now test at hover level
    print("\\n" + "-"*50)
    hover_normalized = quad.drone_sim.motor_commands  # Current hover commands
    forces_at_hover = Bf @ hover_normalized
    
    expected_hover_thrust = quad.params["thr_hover"] * 4
    actual_hover_thrust = -forces_at_hover[2]
    
    print(f"At hover motor commands {hover_normalized[0]:.3f}:")
    print(f"  Expected hover thrust: {expected_hover_thrust:.3f} N")
    print(f"  Actual from allocation: {actual_hover_thrust:.3f} N")
    print(f"  Ratio (actual/expected): {actual_hover_thrust/expected_hover_thrust:.3f}")
    
    return actual_hover_thrust / expected_hover_thrust

def test_propeller_force_calculation():
    """Test individual propeller force calculations."""
    print("\\n\\nTesting Individual Propeller Forces")
    print("-" * 38)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Check how individual propeller forces are calculated in the allocation matrix
    for i, prop in enumerate(quad.drone_sim.config.propellers):
        k_f, k_m = prop["constants"]
        w_max = prop["wmax"]
        prop_dir = np.array(prop["dir"][:3])
        prop_dir = prop_dir / np.linalg.norm(prop_dir)
        
        print(f"\\nPropeller {i+1}:")
        print(f"  k_f: {k_f:.2e}")
        print(f"  w_max: {w_max:.1f} rad/s")
        print(f"  Direction: {prop_dir}")
        
        # Check allocation matrix entry
        Bf, Bm = quad.drone_sim.config.get_allocation_matrices()
        print(f"  Bf column {i}: {Bf[:, i]}")
        
        # What should this be?
        # In allocation matrix creation, we use: k_f * w_max^2 * prop_dir
        expected_force_vector = k_f * w_max**2 * prop_dir
        print(f"  Expected: k_f * w_max^2 * dir = {expected_force_vector}")
        print(f"  Ratio: {Bf[:, i] / expected_force_vector}")

def main():
    print("="*60)
    print("ALLOCATION MATRIX DIAGNOSTIC")
    print("="*60)
    
    # Test 1: Overall allocation matrix scaling
    scaling_ratio = test_allocation_matrix_scaling()
    
    # Test 2: Individual propeller calculations
    test_propeller_force_calculation()
    
    print("\\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    
    if abs(scaling_ratio - 1.0) > 0.1:
        print(f"🔍 ISSUE FOUND: Allocation matrix scaling is off by {scaling_ratio:.2f}x")
        print("   - Allocation matrices produce wrong force magnitudes")
        print("   - Need to fix force scaling in DroneConfiguration")
        
        if scaling_ratio > 2:
            print("   - Forces are too large - allocation matrices over-scaled")
        else:
            print("   - Forces are too small - allocation matrices under-scaled")
    else:
        print("✓ Allocation matrix scaling looks correct")
        print("  - Issue must be elsewhere in the pipeline")

if __name__ == "__main__":
    main()