#!/usr/bin/env python3
"""
Test Quadratic Fix

Verify that the quadratic motor command fix resolves the scaling issue.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_quadratic_fix():
    """Test if the quadratic fix produces correct hover behavior."""
    print("TESTING QUADRATIC FIX")
    print("=" * 30)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    mass = quad.params["mB"]
    g = quad.params["g"]
    w_hover = quad.params["w_hover"]
    
    print(f"Test parameters:")
    print(f"  Mass: {mass:.3f} kg")
    print(f"  Weight: {mass * g:.3f} N")
    print(f"  Hover speed: {w_hover:.1f} rad/s")
    
    # Apply hover commands
    hover_cmds = np.ones(4) * w_hover
    
    print(f"\nApplying hover commands: {hover_cmds[0]:.1f} rad/s")
    
    # Test multiple time steps to see if it hovers
    print("\nTime step progression:")
    print("Time    Position Z    Velocity Z    Net Force")
    print("-" * 45)
    
    for i in range(10):
        t = i * 0.01
        quad.update(t, 0.01, hover_cmds, None)
        
        pos_z = quad.pos[2]
        vel_z = quad.vel[2]
        
        # Calculate net force from acceleration
        # F_net = ma, but we need to account for gravity
        # If hovering perfectly: a = 0
        # If falling: a > 0 (downward in NED)
        # If rising: a < 0 (upward in NED)
        
        # From velocity change, estimate acceleration
        if i > 0:
            acc_z = (vel_z - prev_vel_z) / 0.01
            net_force = mass * acc_z
        else:
            acc_z = 0
            net_force = 0
            
        print(f"{t:4.2f}s  {pos_z:10.6f}   {vel_z:10.6f}   {net_force:8.3f} N")
        
        prev_vel_z = vel_z
    
    # Final assessment
    final_pos = quad.pos[2]
    final_vel = quad.vel[2]
    
    print("\nFINAL ASSESSMENT:")
    print("-" * 20)
    print(f"Final position Z: {final_pos:.6f} m")
    print(f"Final velocity Z: {final_vel:.6f} m/s")
    print(f"Position drift: {abs(final_pos):.6f} m")
    print(f"Velocity magnitude: {abs(final_vel):.6f} m/s")
    
    # Success criteria
    pos_stable = abs(final_pos) < 0.001  # Position drift < 1mm
    vel_stable = abs(final_vel) < 0.01   # Velocity < 1cm/s
    
    if pos_stable and vel_stable:
        print("\n✅ SUCCESS: Quadcopter is hovering stably!")
        print("   Quadratic fix resolved the scaling issue")
        return True
    else:
        print("\n❌ FAILED: Quadcopter is not hovering properly")
        if not pos_stable:
            print(f"   Position drift too large: {abs(final_pos):.6f} m")
        if not vel_stable:
            print(f"   Velocity too large: {abs(final_vel):.6f} m/s")
        return False

def test_force_balance():
    """Test that forces are properly balanced at hover."""
    print("\n\nTESTING FORCE BALANCE")
    print("=" * 25)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Set hover commands
    w_hover = quad.params["w_hover"]
    w_max = quad.drone_sim.config.propellers[0]["wmax"]
    normalized_hover = w_hover / w_max
    
    hover_cmds = np.ones(4) * w_hover
    quad.update(0, 0.01, hover_cmds, None)
    
    # Check actual motor commands stored in simulator
    motor_cmds = quad.drone_sim.motor_commands
    
    print(f"Hover calculations:")
    print(f"  w_hover: {w_hover:.1f} rad/s")
    print(f"  w_max: {w_max:.1f} rad/s") 
    print(f"  Normalized hover: {normalized_hover:.3f}")
    print(f"  Stored motor_commands: {motor_cmds}")
    
    # Calculate forces manually using quadratic relationship
    k_f = quad.drone_sim.config.propellers[0]["constants"][0]
    
    print(f"\nForce calculation (quadratic):")
    print(f"  k_f: {k_f:.2e}")
    
    forces_per_motor = []
    for i, cmd in enumerate(motor_cmds):
        # Quadratic: F = k_f * (cmd * w_max)^2
        actual_w = cmd * w_max
        force = k_f * actual_w**2
        forces_per_motor.append(force)
        print(f"  Motor {i+1}: {cmd:.3f} → {actual_w:.1f} rad/s → {force:.3f} N")
    
    total_thrust = sum(forces_per_motor)
    weight = quad.params["mB"] * quad.params["g"]
    
    print(f"\nForce balance:")
    print(f"  Total thrust: {total_thrust:.3f} N")
    print(f"  Aircraft weight: {weight:.3f} N")
    print(f"  Net force: {total_thrust - weight:.3f} N")
    print(f"  Balance error: {abs(total_thrust - weight)/weight*100:.2f}%")
    
    return abs(total_thrust - weight) < 0.1

def main():
    print("QUADRATIC MOTOR COMMAND FIX VALIDATION")
    print("=" * 50)
    
    # Test 1: Hover stability
    hover_success = test_quadratic_fix()
    
    # Test 2: Force balance
    force_balance_success = test_force_balance()
    
    print("\n\nOVERALL RESULTS")
    print("=" * 20)
    
    if hover_success and force_balance_success:
        print("🎉 QUADRATIC FIX SUCCESSFUL!")
        print("   - Hover stability: ✅")
        print("   - Force balance: ✅") 
        print("   - Allocation matrix scaling issue resolved")
    else:
        print("❌ FIX INCOMPLETE:")
        print(f"   - Hover stability: {'✅' if hover_success else '❌'}")
        print(f"   - Force balance: {'✅' if force_balance_success else '❌'}")
        print("   - Need further debugging")

if __name__ == "__main__":
    main()