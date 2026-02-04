#!/usr/bin/env python3
"""
Test 2: Motor Command Scaling Issue

Test if the motor command scaling is causing excessive thrust.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_hover_calculation():
    """Test the hover motor speed calculation."""
    print("Testing Hover Calculation")
    print("-" * 30)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Check hover calculations
    mass = quad.params["mB"]
    g = quad.params["g"]
    kTh = quad.params["kTh"]
    w_hover = quad.params["w_hover"]
    thr_hover = quad.params["thr_hover"]
    
    print(f"Mass: {mass:.3f} kg")
    print(f"Gravity: {g:.2f} m/s²")
    print(f"Weight: {mass * g:.3f} N")
    print(f"kTh (thrust coeff): {kTh:.2e} N⋅s²⋅m⁻²")
    print(f"Calculated hover motor speed: {w_hover:.1f} rad/s")
    print(f"Calculated hover thrust per motor: {thr_hover:.3f} N")
    print(f"Total hover thrust (4 motors): {thr_hover * 4:.3f} N")
    print(f"Hover thrust / Weight ratio: {(thr_hover * 4) / (mass * g):.3f}")
    
    # Check if hover thrust balances weight
    if abs(thr_hover * 4 - mass * g) < 0.1:
        print("✓ Hover thrust calculation looks correct")
    else:
        print("❌ Hover thrust does not balance weight!")
        print(f"  Expected total thrust: {mass * g:.3f} N")
        print(f"  Calculated total thrust: {thr_hover * 4:.3f} N")
        print(f"  Difference: {thr_hover * 4 - mass * g:.3f} N")

def test_motor_command_conversion():
    """Test the motor command conversion from rad/s to simulation inputs."""
    print("\\n\\nTesting Motor Command Conversion")
    print("-" * 40)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Test different motor command levels
    test_speeds = [0, 100, quad.params["w_hover"], 800, quad.params["maxWmotor"]]
    
    print("Motor Speed (rad/s) → Thrust per Motor (N)")
    print("-" * 40)
    
    for w_cmd in test_speeds:
        # Calculate expected thrust
        kTh = quad.params["kTh"]
        expected_thrust = kTh * w_cmd**2
        
        print(f"{w_cmd:6.1f} rad/s → {expected_thrust:8.3f} N")
    
    print("\\nTesting conversion in update_from_controller()...")
    
    # Test the conversion pipeline
    w_hover = quad.params["w_hover"]
    w_max_list = [prop["wmax"] for prop in quad.drone_sim.config.propellers]
    w_max = w_max_list[0]  # All should be the same
    
    print(f"w_hover: {w_hover:.1f} rad/s")
    print(f"w_max: {w_max:.1f} rad/s")
    
    # Simulate the conversion in update_from_controller
    w_cmd_input = np.ones(4) * w_hover  # Input motor commands
    print(f"Input motor commands: {w_cmd_input[0]:.1f} rad/s")
    
    # This is what happens in update_from_controller:
    motor_commands_normalized = np.clip(w_cmd_input / w_max, 0, 1)
    print(f"Normalized commands [0,1]: {motor_commands_normalized[0]:.3f}")
    
    # This gets passed to DroneSimulator.step()
    print(f"Commands passed to DroneSimulator: {motor_commands_normalized[0]:.3f}")
    
    # Check what thrust this should produce
    # In DroneSimulator dynamics, this gets converted back:
    actual_w = motor_commands_normalized[0] * w_max
    actual_thrust = kTh * actual_w**2
    print(f"Reconstructed motor speed: {actual_w:.1f} rad/s")
    print(f"Reconstructed thrust per motor: {actual_thrust:.3f} N")
    print(f"Total reconstructed thrust: {actual_thrust * 4:.3f} N")

def test_dynamics_motor_conversion():
    """Test what happens inside the DroneSimulator dynamics."""
    print("\\n\\nTesting DroneSimulator Internal Conversion")
    print("-" * 45)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Check the motor command values stored in the simulator
    print(f"Initial motor_commands in DroneSimulator: {quad.drone_sim.motor_commands}")
    
    # Apply hover commands
    w_hover = quad.params["w_hover"]
    hover_cmds = np.ones(4) * w_hover
    
    # Simulate one update step
    quad.update(0, 0.01, hover_cmds, None)
    
    print(f"After update with {w_hover:.1f} rad/s commands:")
    print(f"DroneSimulator.motor_commands: {quad.drone_sim.motor_commands}")
    print(f"DroneSimulator.state (relevant): {quad.drone_sim.state}")
    
    # Check the forces calculated
    Bf, Bm = quad.drone_sim.config.get_allocation_matrices()
    forces = Bf @ quad.drone_sim.motor_commands
    moments = Bm @ quad.drone_sim.motor_commands
    
    print(f"Forces from allocation [Fx, Fy, Fz]: {forces}")
    print(f"Moments from allocation [Mx, My, Mz]: {moments}")
    print(f"Total upward force: {-forces[2]:.3f} N")  # Negative because Fz is down
    print(f"Weight to overcome: {quad.params['mB'] * quad.params['g']:.3f} N")
    print(f"Net force: {-forces[2] - quad.params['mB'] * quad.params['g']:.3f} N")

def main():
    print("="*50)
    print("MOTOR SCALING DIAGNOSTIC")
    print("="*50)
    
    # Test 1: Hover calculation
    test_hover_calculation()
    
    # Test 2: Motor command conversion
    test_motor_command_conversion()
    
    # Test 3: Dynamics internal conversion
    test_dynamics_motor_conversion()

if __name__ == "__main__":
    main()