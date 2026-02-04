#!/usr/bin/env python3
"""
Simple Direct Hover Test

This bypasses the trajectory system and directly commands the drone to hover at a target altitude.
Tests the controller and mixer integration more directly.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

# Import frameworks
from ctrl import Control
from quadFiles.quad import Quadcopter
from drone_simulator import ConfigurableQuadcopter
from utils.windModel import Wind
import utils
import config

def test_direct_hover(use_configurable=False, target_z=1.0, sim_time=10):
    """Test direct hover without trajectory system."""
    
    framework_name = "Configurable" if use_configurable else "Original"
    print(f"Testing {framework_name} Framework - Direct Hover")
    print("-" * 50)
    
    # Initialize drone
    if use_configurable:
        # Use 7-inch props to better match original kTh
        propellers = [
            {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
            {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
            {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
            {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
        ]
        quad = ConfigurableQuadcopter(0, propellers=propellers)
    else:
        quad = Quadcopter(0)
    
    # Initialize controller (without trajectory system)
    ctrl = Control(quad, 0)  # yawType = 0 (no yaw control)
    wind = Wind('None', 0, 0, 0)
    
    # Print initial state
    print(f"Initial position: {quad.pos}")
    print(f"Target Z: {target_z}")
    print(f"Mass: {quad.params['mB']:.3f} kg")
    print(f"kTh: {quad.params['kTh']:.2e}")
    print(f"Hover thrust per motor: {quad.params['thr_hover']:.3f} N")
    print(f"Hover motor speed: {quad.params['w_hover']:.1f} rad/s")
    
    # Print mixer matrix for comparison
    print(f"Mixer matrix shape: {quad.params['mixerFM'].shape}")
    print(f"Mixer matrix diagonal (thrust): {np.diag(quad.params['mixerFM'][:4, :4])}")
    
    # Simulation setup
    dt = 0.005
    num_steps = int(sim_time / dt)
    
    # Data arrays
    times = []
    positions = []
    velocities = []
    motor_commands = []
    thrust_commands = []
    errors = []
    
    # Control parameters - manual setpoints
    target_pos = np.array([0.0, 0.0, target_z])
    
    # Run simulation with direct control
    t = 0
    for i in range(num_steps):
        
        # Direct position and velocity control (bypass trajectory)
        ctrl.pos_sp = target_pos
        ctrl.vel_sp = np.zeros(3)  # Hover velocity
        ctrl.acc_sp = np.zeros(3)  # No acceleration 
        ctrl.thrust_sp = np.zeros(3)
        ctrl.eul_sp = np.zeros(3)  # Level attitude
        ctrl.pqr_sp = np.zeros(3)  # No rotation
        ctrl.yawFF = np.zeros(3)
        
        # Run position control
        ctrl.z_pos_control(quad, dt)
        ctrl.xy_pos_control(quad, dt)
        
        # Saturate velocity
        ctrl.saturateVel()
        
        # Run velocity control
        ctrl.z_vel_control(quad, dt)
        ctrl.xy_vel_control(quad, dt)
        
        # Convert thrust to attitude
        ctrl.thrustToAttitude(quad, dt)
        
        # Run attitude control
        ctrl.attitude_control(quad, dt)
        
        # Run rate control
        ctrl.rate_control(quad, dt)
        
        # Mix controls to motor commands
        ctrl.w_cmd = utils.mixerFM(quad, np.linalg.norm(ctrl.thrust_sp), ctrl.rateCtrl)
        
        # Update drone dynamics
        quad.update(t, dt, ctrl.w_cmd, wind)
        t += dt
        
        # Log data
        times.append(t)
        positions.append(quad.pos.copy())
        velocities.append(quad.vel.copy())
        motor_commands.append(ctrl.w_cmd.copy())
        thrust_commands.append(ctrl.thrust_sp.copy())
        
        error = np.linalg.norm(quad.pos - target_pos)
        errors.append(error)
        
        # Progress update every 1 second
        if i % int(1.0/dt) == 0:
            print(f"t={t:.1f}s: pos={quad.pos}, vel_mag={np.linalg.norm(quad.vel):.3f}, error={error:.3f}m")
            print(f"       thrust_sp={ctrl.thrust_sp}, w_cmd_avg={np.mean(ctrl.w_cmd):.1f}")
    
    # Final results
    final_pos = quad.pos
    final_error = np.linalg.norm(final_pos - target_pos)
    success = final_error < 0.25
    
    print(f"\\nFINAL RESULTS:")
    print(f"Target position: {target_pos}")
    print(f"Final position: {final_pos}")
    print(f"Final error: {final_error:.4f} m")
    print(f"Success (< 0.25m): {'✓' if success else '✗'}")
    
    return {
        'success': success,
        'final_error': final_error,
        'final_pos': final_pos,
        'times': np.array(times),
        'positions': np.array(positions),
        'velocities': np.array(velocities),
        'motor_commands': np.array(motor_commands),
        'thrust_commands': np.array(thrust_commands),
        'errors': np.array(errors)
    }

def plot_results(results, title):
    """Plot simulation results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Direct Hover Test Results - {title}')
    
    times = results['times']
    positions = results['positions']
    velocities = results['velocities']
    motor_commands = results['motor_commands']
    thrust_commands = results['thrust_commands']
    errors = results['errors']
    
    # Position vs time
    axes[0,0].plot(times, positions[:, 0], 'r-', label='X', linewidth=2)
    axes[0,0].plot(times, positions[:, 1], 'g-', label='Y', linewidth=2)
    axes[0,0].plot(times, positions[:, 2], 'b-', label='Z', linewidth=2)
    axes[0,0].axhline(y=1.0, color='b', linestyle='--', alpha=0.5, label='Target Z')
    axes[0,0].set_title('Position vs Time')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Position (m)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Velocity vs time
    axes[0,1].plot(times, velocities[:, 0], 'r-', label='vX', linewidth=2)
    axes[0,1].plot(times, velocities[:, 1], 'g-', label='vY', linewidth=2)
    axes[0,1].plot(times, velocities[:, 2], 'b-', label='vZ', linewidth=2)
    axes[0,1].set_title('Velocity vs Time')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Velocity (m/s)')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Error vs time
    axes[0,2].plot(times, errors, 'b-', linewidth=2)
    axes[0,2].axhline(y=0.25, color='r', linestyle='--', label='Success threshold')
    axes[0,2].set_title('Position Error vs Time')
    axes[0,2].set_xlabel('Time (s)')
    axes[0,2].set_ylabel('Error (m)')
    axes[0,2].set_yscale('log')
    axes[0,2].legend()
    axes[0,2].grid(True)
    
    # Motor commands
    for i in range(min(4, motor_commands.shape[1])):
        axes[1,0].plot(times, motor_commands[:, i], label=f'Motor {i+1}', linewidth=2)
    axes[1,0].set_title('Motor Commands')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Motor Speed (rad/s)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Thrust commands
    axes[1,1].plot(times, thrust_commands[:, 0], 'r-', label='Thrust X', linewidth=2)
    axes[1,1].plot(times, thrust_commands[:, 1], 'g-', label='Thrust Y', linewidth=2)
    axes[1,1].plot(times, thrust_commands[:, 2], 'b-', label='Thrust Z', linewidth=2)
    axes[1,1].set_title('Thrust Commands')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Thrust (N)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 3D trajectory
    axes[1,2] = fig.add_subplot(2, 3, 6, projection='3d')
    axes[1,2].plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
    axes[1,2].scatter([0], [0], [0], color='g', s=100, label='Start')
    axes[1,2].scatter([0], [0], [1], color='r', s=100, label='Target')
    axes[1,2].set_title('3D Trajectory')
    axes[1,2].set_xlabel('X (m)')
    axes[1,2].set_ylabel('Y (m)')
    axes[1,2].set_zlabel('Z (m)')
    axes[1,2].legend()
    
    plt.tight_layout()
    plt.show()

def compare_frameworks(target_z=1.0, sim_time=10):
    """Compare both frameworks."""
    print("="*60)
    print("DIRECT HOVER COMPARISON")
    print("="*60)
    
    # Test original
    orig_results = test_direct_hover(use_configurable=False, target_z=target_z, sim_time=sim_time)
    
    # Test configurable
    config_results = test_direct_hover(use_configurable=True, target_z=target_z, sim_time=sim_time)
    
    # Summary
    print("\\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"Target Z: {target_z}")
    print(f"Original Framework     - Success: {'✓' if orig_results['success'] else '✗'}, Error: {orig_results['final_error']:.4f}m")
    print(f"Configurable Framework - Success: {'✓' if config_results['success'] else '✗'}, Error: {config_results['final_error']:.4f}m")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Framework Comparison - Direct Hover')
    
    # Position Z comparison
    axes[0,0].plot(orig_results['times'], orig_results['positions'][:, 2], 'b-', label='Original', linewidth=2)
    axes[0,0].plot(config_results['times'], config_results['positions'][:, 2], 'r--', label='Configurable', linewidth=2)
    axes[0,0].axhline(y=target_z, color='g', linestyle=':', label='Target')
    axes[0,0].set_title('Z Position Comparison')
    axes[0,0].set_xlabel('Time (s)')
    axes[0,0].set_ylabel('Z Position (m)')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Error comparison
    axes[0,1].plot(orig_results['times'], orig_results['errors'], 'b-', label='Original', linewidth=2)
    axes[0,1].plot(config_results['times'], config_results['errors'], 'r--', label='Configurable', linewidth=2)
    axes[0,1].axhline(y=0.25, color='g', linestyle=':', label='Success threshold')
    axes[0,1].set_title('Error Comparison')
    axes[0,1].set_xlabel('Time (s)')
    axes[0,1].set_ylabel('Error (m)')
    axes[0,1].set_yscale('log')
    axes[0,1].legend()
    axes[0,1].grid(True)
    
    # Motor command comparison (Motor 1)
    axes[1,0].plot(orig_results['times'], orig_results['motor_commands'][:, 0], 'b-', label='Original', linewidth=2)
    axes[1,0].plot(config_results['times'], config_results['motor_commands'][:, 0], 'r--', label='Configurable', linewidth=2)
    axes[1,0].set_title('Motor 1 Command Comparison')
    axes[1,0].set_xlabel('Time (s)')
    axes[1,0].set_ylabel('Motor Speed (rad/s)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Thrust Z comparison
    axes[1,1].plot(orig_results['times'], orig_results['thrust_commands'][:, 2], 'b-', label='Original', linewidth=2)
    axes[1,1].plot(config_results['times'], config_results['thrust_commands'][:, 2], 'r--', label='Configurable', linewidth=2)
    axes[1,1].set_title('Z Thrust Command Comparison')
    axes[1,1].set_xlabel('Time (s)')
    axes[1,1].set_ylabel('Z Thrust (N)')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Simple direct hover test')
    parser.add_argument('--framework', choices=['original', 'config', 'both'], 
                       default='both', help='Which framework to test')
    parser.add_argument('--target-z', type=float, default=1.0, 
                       help='Target altitude')
    parser.add_argument('--time', type=float, default=10, 
                       help='Simulation time (s)')
    parser.add_argument('--plot', action='store_true', 
                       help='Show individual plots')
    
    args = parser.parse_args()
    
    # Run tests
    if args.framework == 'original':
        results = test_direct_hover(use_configurable=False, target_z=args.target_z, sim_time=args.time)
        if args.plot:
            plot_results(results, "Original Framework")
    elif args.framework == 'config':
        results = test_direct_hover(use_configurable=True, target_z=args.target_z, sim_time=args.time)
        if args.plot:
            plot_results(results, "Configurable Framework")
    elif args.framework == 'both':
        compare_frameworks(args.target_z, args.time)

if __name__ == "__main__":
    main()