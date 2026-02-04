#!/usr/bin/env python3
"""
Test Matched Drone Hover

Simple test of the matched drone's hover capability.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_matched_drone_hover():
    """Test hover capability of matched drone."""
    print("TESTING MATCHED DRONE HOVER")
    print("=" * 35)
    
    # Create matched drone
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    print(f"Matched drone parameters:")
    print(f"  Mass: {quad.params['mB']:.3f} kg")
    print(f"  kTh: {quad.params['kTh']:.2e}")
    print(f"  w_hover: {quad.params['w_hover']:.1f} rad/s")
    
    # Test direct hover commands
    w_hover = quad.params["w_hover"]
    hover_cmds = np.ones(4) * w_hover
    
    print(f"\nApplying direct hover commands: {w_hover:.1f} rad/s")
    
    # Test multiple steps
    for i in range(10):
        quad.update(i * 0.01, 0.01, hover_cmds, None)
        
        if i % 3 == 0:  # Print every 3rd step
            print(f"  Step {i}: pos_z = {quad.pos[2]:.6f} m, vel_z = {quad.vel[2]:.6f} m/s")
    
    final_pos = quad.pos[2]
    final_vel = quad.vel[2]
    
    print(f"\nFinal state:")
    print(f"  Position Z: {final_pos:.6f} m")
    print(f"  Velocity Z: {final_vel:.6f} m/s")
    
    # Success criteria
    pos_stable = abs(final_pos) < 0.001  # Within 1mm
    vel_stable = abs(final_vel) < 0.01   # Within 1cm/s
    
    if pos_stable and vel_stable:
        print("✅ SUCCESS: Matched drone hovers stably!")
        return True
    else:
        print("❌ FAILED: Matched drone does not hover properly")
        return False

def main():
    success = test_matched_drone_hover()
    
    if success:
        print("\n🎉 MATCHED DRONE IS READY FOR TRAJECTORY TESTING!")
    else:
        print("\n⚠️ Need to fix matched drone before trajectory testing")

if __name__ == "__main__":
    main()