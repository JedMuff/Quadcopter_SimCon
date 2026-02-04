#!/usr/bin/env python3
"""
Fix Allocation Matrix Scaling Issue

The issue is that allocation matrices assume LINEAR relationship but motor physics is QUADRATIC.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def analyze_scaling_issue():
    """Analyze the linear vs quadratic scaling issue."""
    print("ALLOCATION SCALING ROOT CAUSE ANALYSIS")
    print("=" * 50)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    k_f = quad.drone_sim.config.propellers[0]["constants"][0]
    w_max = quad.drone_sim.config.propellers[0]["wmax"]
    w_hover = quad.params["w_hover"]
    mass = quad.params["mB"]
    g = quad.params["g"]
    
    print(f"Key parameters:")
    print(f"  k_f: {k_f:.2e} N⋅s²⋅m⁻²")
    print(f"  w_max: {w_max:.1f} rad/s")
    print(f"  w_hover: {w_hover:.1f} rad/s")
    print(f"  mass: {mass:.3f} kg")
    print(f"  Required hover force: {mass * g:.3f} N")
    
    # Get allocation matrix
    Bf, _ = quad.drone_sim.config.get_allocation_matrices()
    
    print(f"\nAllocation matrix Bf[2,0] (force per motor): {Bf[2,0]:.3f}")
    
    # Test normalized hover command
    normalized_hover = w_hover / w_max
    print(f"\nNormalized hover command: {normalized_hover:.3f}")
    
    # Current allocation matrix approach (LINEAR)
    linear_force_per_motor = -Bf[2, 0] * normalized_hover
    linear_total_force = linear_force_per_motor * 4
    
    print(f"\nCURRENT (Linear) approach:")
    print(f"  Force = Bf * motor_cmd")
    print(f"  Force per motor: {Bf[2,0]:.3f} * {normalized_hover:.3f} = {linear_force_per_motor:.3f} N")
    print(f"  Total force: {linear_total_force:.3f} N")
    
    # Correct physics approach (QUADRATIC)
    quadratic_force_per_motor = k_f * (normalized_hover * w_max)**2
    quadratic_total_force = quadratic_force_per_motor * 4
    
    print(f"\nCORRECT (Quadratic) approach:")
    print(f"  Force = k_f * (motor_cmd * w_max)²")
    print(f"  Force per motor: {k_f:.2e} * ({normalized_hover:.3f} * {w_max:.0f})² = {quadratic_force_per_motor:.3f} N")
    print(f"  Total force: {quadratic_total_force:.3f} N")
    
    # Compare
    scaling_error = linear_total_force / quadratic_total_force
    
    print(f"\nCOMPARISON:")
    print(f"  Linear/Quadratic ratio: {scaling_error:.3f}")
    print(f"  Error factor: {scaling_error:.2f}x")
    
    print(f"\nVERIFICATION:")
    print(f"  Expected hover force: {mass * g:.3f} N")
    print(f"  Linear produces: {linear_total_force:.3f} N")
    print(f"  Quadratic produces: {quadratic_total_force:.3f} N")
    print(f"  Quadratic matches expected: {abs(quadratic_total_force - mass * g) < 0.1}")
    
    return scaling_error

def propose_fixes():
    """Propose fixes for the scaling issue."""
    print("\n\nPROPOSED FIXES")
    print("=" * 20)
    
    print("The allocation matrices currently use:")
    print("  F_body = Bf @ motor_commands")
    print("  where motor_commands are in [0,1]")
    print()
    print("But motor physics is quadratic:")
    print("  thrust = k_f * w²")
    print("  w = motor_cmd * w_max")
    print("  thrust = k_f * (motor_cmd * w_max)²")
    print()
    print("SOLUTION OPTIONS:")
    print()
    print("Option 1: Square motor commands in dynamics")
    print("  Change: F_body = Bf @ motor_commands")
    print("  To:     F_body = Bf @ (motor_commands²)")
    print()
    print("Option 2: Take square root in motor command conversion")
    print("  Change: motor_cmd = w_input / w_max")
    print("  To:     motor_cmd = sqrt(w_input / w_max)")
    print()
    print("Option 3: Rescale allocation matrices")
    print("  Divide Bf and Bm by sqrt(scaling_factor)")
    print()
    print("RECOMMENDATION: Option 1 (square in dynamics)")
    print("  - Most physically correct")
    print("  - Preserves [0,1] normalized command range")
    print("  - Minimal changes to existing code")

def main():
    scaling_error = analyze_scaling_issue()
    propose_fixes()
    
    print(f"\n\nCONCLUSION:")
    print("=" * 15)
    print(f"🔍 ROOT CAUSE: Linear vs Quadratic motor mapping")
    print(f"   Current system has {scaling_error:.2f}x scaling error")
    print(f"   Need to implement quadratic motor command relationship")

if __name__ == "__main__":
    main()