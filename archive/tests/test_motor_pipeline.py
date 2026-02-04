#!/usr/bin/env python3
"""
Test 4: Motor Command Pipeline

Trace the exact path of motor commands through the system.
"""

import numpy as np
from drone_simulator import ConfigurableQuadcopter

def test_motor_command_pipeline():
    """Trace motor commands step by step."""
    print("Testing Motor Command Pipeline")
    print("=" * 35)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    print("Step 1: Initial State")
    print("-" * 20)
    print(f"DroneSimulator.motor_commands: {quad.drone_sim.motor_commands}")
    print(f"ConfigurableQuadcopter motor speeds: {quad.wMotor}")
    
    print("\\nStep 2: Calculate Hover Commands")
    print("-" * 35)
    w_hover = quad.params["w_hover"]
    hover_cmds = np.ones(4) * w_hover
    print(f"Calculated hover commands: {hover_cmds} rad/s")
    print(f"Expected hover speed: {w_hover:.1f} rad/s")
    
    print("\\nStep 3: Apply Commands (Tracing Each Step)")
    print("-" * 48)
    
    # Manually trace what should happen:
    print(f"Input to ConfigurableQuadcopter.update(): {hover_cmds}")
    
    # This gets passed to drone_sim.update_from_controller()
    print("\\nInside update_from_controller():")
    
    full_w_cmd = hover_cmds  # No w_cmd_full initially
    print(f"  full_w_cmd: {full_w_cmd}")
    
    propeller_w_max = [prop["wmax"] for prop in quad.drone_sim.config.propellers]
    print(f"  propeller_w_max: {propeller_w_max}")
    
    motor_commands = np.zeros(quad.drone_sim.num_motors)
    for i in range(min(len(full_w_cmd), quad.drone_sim.num_motors)):
        w_max = propeller_w_max[i] if i < len(propeller_w_max) else propeller_w_max[0]
        motor_commands[i] = np.clip(full_w_cmd[i] / w_max, 0, 1)
        print(f"  motor_commands[{i}]: {full_w_cmd[i]:.1f} / {w_max:.1f} = {motor_commands[i]:.3f}")
    
    print(f"  Final motor_commands array: {motor_commands}")
    
    print("\\nStep 4: Apply Update")
    print("-" * 20)
    
    # Now actually apply the update
    result = quad.update(0, 0.01, hover_cmds, None)
    
    print(f"After update:")
    print(f"  DroneSimulator.motor_commands: {quad.drone_sim.motor_commands}")
    print(f"  ConfigurableQuadcopter.wMotor: {quad.wMotor}")
    print(f"  Position change: {quad.pos}")
    print(f"  Velocity: {quad.vel}")
    
    print("\\nStep 5: Force Calculation Check")
    print("-" * 30)
    
    Bf, Bm = quad.drone_sim.config.get_allocation_matrices()
    forces = Bf @ quad.drone_sim.motor_commands
    print(f"Forces from allocation (Bf @ motor_commands):")
    print(f"  Bf shape: {Bf.shape}")
    print(f"  motor_commands: {quad.drone_sim.motor_commands}")
    print(f"  Bf @ motor_commands = {forces}")
    print(f"  Total upward force: {-forces[2]:.3f} N")
    
    # What should this be?
    expected_force_per_motor = quad.params["kTh"] * w_hover**2
    expected_total = expected_force_per_motor * 4
    print(f"  Expected total force: {expected_total:.3f} N")
    print(f"  Ratio (actual/expected): {-forces[2]/expected_total:.3f}")

def test_step_simulation():
    """Test the DroneSimulator.step() method directly."""
    print("\\n\\nTesting DroneSimulator.step() Directly")
    print("=" * 43)
    
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
    ]
    
    quad = ConfigurableQuadcopter(0, propellers=propellers)
    
    # Calculate normalized hover commands
    w_hover = quad.params["w_hover"]
    w_max = quad.drone_sim.config.propellers[0]["wmax"]
    normalized_hover = w_hover / w_max
    
    print(f"w_hover: {w_hover:.1f} rad/s")
    print(f"w_max: {w_max:.1f} rad/s") 
    print(f"normalized_hover: {normalized_hover:.3f}")
    
    # Apply normalized commands directly to DroneSimulator
    print("\\nCalling DroneSimulator.step() directly...")
    normalized_cmds = np.ones(4) * normalized_hover
    print(f"Input normalized commands: {normalized_cmds}")
    
    result = quad.drone_sim.step(normalized_cmds)
    
    print(f"After DroneSimulator.step():")
    print(f"  motor_commands: {quad.drone_sim.motor_commands}")
    print(f"  state: {quad.drone_sim.state}")
    print(f"  position: {quad.drone_sim.state[0:3]}")
    print(f"  velocity: {quad.drone_sim.state[3:6]}")
    
    # Check forces
    Bf, Bm = quad.drone_sim.config.get_allocation_matrices()
    forces = Bf @ quad.drone_sim.motor_commands
    print(f"  Forces: {forces}")
    print(f"  Upward force: {-forces[2]:.3f} N")

def main():
    test_motor_command_pipeline()
    test_step_simulation()

if __name__ == "__main__":
    main()