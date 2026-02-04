#!/usr/bin/env python3
"""
Test 1: Coordinate Frame Issue

Test if the coordinate frame mismatch is causing the dynamics issue.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_gravity_direction():
    """Test if gravity is applied in the correct direction."""
    print("Testing Gravity Direction")
    print("-" * 30)
    
    # Create simple quadcopter
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    print(f"Initial position: {quad.pos}")
    print(f"Initial velocity: {quad.vel}")
    
    # Test: Let drone fall under gravity with no motor commands (all zero)
    print("\\nTesting free fall (no thrust)...")
    
    dt = 0.01
    zero_commands = np.zeros(4)  # No motor thrust
    
    for i in range(100):  # 1 second of free fall
        quad.update(i*dt, dt, zero_commands, None)
        
        if i % 20 == 0:  # Print every 0.2 seconds
            print(f"t={i*dt:.1f}s: pos={quad.pos}, vel={quad.vel}")
    
    print(f"\\nAfter 1s free fall:")
    print(f"Final position: {quad.pos}")
    print(f"Final velocity: {quad.vel}")
    
    # Expected behavior in NED: positive Z velocity (falling down)
    # Expected behavior in ENU: negative Z velocity (falling down)
    
    if quad.vel[2] > 0:
        print("✓ Gravity acts in +Z direction (NED-like)")
        return "NED"
    elif quad.vel[2] < 0:
        print("✓ Gravity acts in -Z direction (ENU-like)")
        return "ENU"
    else:
        print("? No gravity effect detected")
        return "NONE"

def test_thrust_direction():
    """Test thrust direction with minimal motor commands."""
    print("\\n\\nTesting Thrust Direction")
    print("-" * 30)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    print(f"Initial position: {quad.pos}")
    
    # Test with hover-level motor commands
    dt = 0.01
    hover_commands = np.ones(4) * quad.params["w_hover"]  # Hover motor speed
    
    print(f"Using hover motor commands: {hover_commands[0]:.1f} rad/s")
    
    for i in range(100):  # 1 second
        quad.update(i*dt, dt, hover_commands, None)
        
        if i % 20 == 0:  # Print every 0.2 seconds
            print(f"t={i*dt:.1f}s: pos={quad.pos}, vel={quad.vel}")
    
    print(f"\\nAfter 1s with hover thrust:")
    print(f"Final position: {quad.pos}")
    print(f"Final velocity: {quad.vel}")
    
    # In NED: hover should maintain altitude (near zero Z velocity)
    # If Z velocity is significantly positive: thrust is too weak or wrong direction
    # If Z velocity is significantly negative: thrust is too strong or wrong direction
    
    if abs(quad.vel[2]) < 0.1:
        print("✓ Thrust approximately balances gravity")
        return "BALANCED"
    elif quad.vel[2] > 0.1:
        print("⚠ Thrust too weak or wrong direction (falling)")
        return "WEAK"
    else:
        print("⚠ Thrust too strong or wrong direction (climbing)")
        return "STRONG"

def main():
    print("="*50)
    print("COORDINATE FRAME DIAGNOSTIC")
    print("="*50)
    
    # Test 1: Gravity direction
    gravity_result = test_gravity_direction()
    
    # Test 2: Thrust direction
    thrust_result = test_thrust_direction()
    
    print("\\n" + "="*50)
    print("DIAGNOSIS")
    print("="*50)
    print(f"Gravity direction: {gravity_result}")
    print(f"Thrust balance: {thrust_result}")
    
    if gravity_result == "ENU" and thrust_result in ["WEAK", "STRONG"]:
        print("\\n🔍 LIKELY ISSUE: Coordinate frame mismatch!")
        print("   - DroneSimulator uses ENU coordinates (+Z up)")
        print("   - Original framework expects NED coordinates (+Z down)")
        print("   - Solution: Fix gravity direction in DroneSimulator")
    elif gravity_result == "NED" and thrust_result == "BALANCED":
        print("\\n✓ Coordinate frames look correct")
        print("   - Issue likely elsewhere (motor scaling, state conversion, etc.)")
    else:
        print(f"\\n❓ Unexpected combination: {gravity_result} + {thrust_result}")

if __name__ == "__main__":
    main()