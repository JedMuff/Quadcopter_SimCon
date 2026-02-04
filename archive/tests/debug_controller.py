#!/usr/bin/env python3
"""
Controller Debugging Tool

This script runs both the original and configurable drone frameworks side-by-side
to compare all intermediate values and identify controller integration issues.

Usage:
    python debug_controller.py --time 5 --target 0,0,1
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

# Import original framework
from trajectory import Trajectory
from ctrl import Control
from quadFiles.quad import Quadcopter
from utils.windModel import Wind
import utils
import config

# Import configurable framework  
from drone_simulator import ConfigurableQuadcopter

class ControllerDebugger:
    """Side-by-side comparison of original vs configurable drone frameworks."""
    
    def __init__(self, target_pos=[0, 0, 1], simulation_time=10, dt=0.005):
        self.target_pos = np.array(target_pos)
        self.sim_time = simulation_time
        self.dt = dt
        
        # Debug logs
        self.debug_log = {
            'time': [],
            'original': {'state': [], 'params': {}, 'control': [], 'error': []},
            'configurable': {'state': [], 'params': {}, 'control': [], 'error': []}
        }
        
        # Initialize both frameworks
        self._init_original_framework()
        self._init_configurable_framework()
        
    def _init_original_framework(self):
        """Initialize original quadcopter framework."""
        print("Initializing original framework...")
        
        self.quad_orig = Quadcopter(0)
        
        # Create simple hover trajectory
        trajSelect = np.zeros(3)
        trajSelect[0] = 1  # pos_waypoint_timed
        trajSelect[1] = 0  # no yaw
        trajSelect[2] = 0  # waypoint time
        
        self.traj_orig = Trajectory(self.quad_orig, "xyz_pos", trajSelect) 
        self.ctrl_orig = Control(self.quad_orig, self.traj_orig.yawType)
        self.wind_orig = Wind('None', 0, 0, 0)  # No wind
        
        # Override trajectory to hover at target position
        self.traj_orig.wps = np.array([[0, 0, 0], self.target_pos])
        self.traj_orig.t_wps = np.array([0, self.sim_time])
        
        # Store original parameters
        self.debug_log['original']['params'] = self.quad_orig.params.copy()
        
        print(f"Original - Mass: {self.quad_orig.params['mB']:.3f} kg")
        print(f"Original - kTh: {self.quad_orig.params['kTh']:.2e}")
        print(f"Original - Arms: {self.quad_orig.params['dxm']:.3f} m")
        
    def _init_configurable_framework(self):
        """Initialize configurable quadcopter framework."""
        print("\\nInitializing configurable framework...")
        
        # Create similar quadcopter configuration to original
        # Using prop size 7 to better match original kTh values
        propellers = [
            {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
            {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7},
            {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": 7},
            {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": 7}
        ]
        
        self.quad_config = ConfigurableQuadcopter(0, propellers=propellers)
        
        # Create trajectory
        trajSelect = np.zeros(3) 
        trajSelect[0] = 1  # pos_waypoint_timed
        trajSelect[1] = 0  # no yaw
        trajSelect[2] = 0  # waypoint time
        
        self.traj_config = Trajectory(self.quad_config, "xyz_pos", trajSelect)
        self.ctrl_config = Control(self.quad_config, self.traj_config.yawType)
        self.wind_config = Wind('None', 0, 0, 0)  # No wind
        
        # Override trajectory to hover at target position
        self.traj_config.wps = np.array([[0, 0, 0], self.target_pos])
        self.traj_config.t_wps = np.array([0, self.sim_time])
        
        # Store configurable parameters
        self.debug_log['configurable']['params'] = self.quad_config.params.copy()
        
        print(f"Configurable - Mass: {self.quad_config.params['mB']:.3f} kg")
        print(f"Configurable - kTh: {self.quad_config.params['kTh']:.2e}")
        print(f"Configurable - Arms: {self.quad_config.params['dxm']:.3f} m")
        
    def compare_parameters(self):
        """Compare key parameters between frameworks."""
        print("\\n" + "="*60)
        print("PARAMETER COMPARISON")
        print("="*60)
        
        orig = self.debug_log['original']['params']
        conf = self.debug_log['configurable']['params']
        
        key_params = ['mB', 'kTh', 'kTo', 'dxm', 'dym', 'maxThr', 'w_hover']
        
        for param in key_params:
            if param in orig and param in conf:
                ratio = conf[param] / orig[param] if orig[param] != 0 else float('inf')
                print(f"{param:10s}: Original={orig[param]:.4e}, Config={conf[param]:.4e}, Ratio={ratio:.3f}")
        
        # Compare mixer matrices
        print("\\nMixer Matrix Comparison:")
        print("Original mixerFM shape:", orig['mixerFM'].shape)
        print("Original mixerFM:\\n", orig['mixerFM'])
        print("\\nConfigurable mixerFM shape:", conf['mixerFM'].shape) 
        print("Configurable mixerFM:\\n", conf['mixerFM'])
        
        # Compare inertia matrices
        print("\\nInertia Matrix Comparison:")
        print("Original IB:\\n", orig['IB'])
        print("\\nConfigurable IB:\\n", conf['IB'])
        
    def run_comparison(self):
        """Run both simulations and collect comparison data."""
        print(f"\\nRunning comparison simulation for {self.sim_time}s...")
        
        num_steps = int(self.sim_time / self.dt)
        t = 0
        
        for i in range(num_steps):
            self.debug_log['time'].append(t)
            
            # Step original framework
            orig_state, orig_control = self._step_original(t)
            self.debug_log['original']['state'].append(orig_state)
            self.debug_log['original']['control'].append(orig_control)
            
            # Step configurable framework
            config_state, config_control = self._step_configurable(t)
            self.debug_log['configurable']['state'].append(config_state)
            self.debug_log['configurable']['control'].append(config_control)
            
            # Calculate errors
            orig_error = np.linalg.norm(orig_state['pos'] - self.target_pos)
            config_error = np.linalg.norm(config_state['pos'] - self.target_pos)
            
            self.debug_log['original']['error'].append(orig_error)
            self.debug_log['configurable']['error'].append(config_error)
            
            t += self.dt
            
            # Print progress every second
            if i % int(1.0/self.dt) == 0:
                print(f"t={t:.1f}s: Orig_error={orig_error:.3f}m, Config_error={config_error:.3f}m")
        
        print("Simulation complete!")
        
    def _step_original(self, t):
        """Step the original framework."""
        # Get desired states
        sDes = self.traj_orig.desiredState(t, self.dt, self.quad_orig)
        
        # Run controller
        self.ctrl_orig.controller(self.traj_orig, self.quad_orig, sDes, self.dt)
        
        # Update dynamics
        self.quad_orig.update(t, self.dt, self.ctrl_orig.w_cmd, self.wind_orig)
        
        # Collect state info
        state_info = {
            'pos': self.quad_orig.pos.copy(),
            'vel': self.quad_orig.vel.copy(),
            'euler': self.quad_orig.euler.copy(),
            'omega': self.quad_orig.omega.copy()
        }
        
        # Collect control info
        control_info = {
            'thrust_sp': self.ctrl_orig.thrust_sp.copy(),
            'rate_sp': self.ctrl_orig.rate_sp.copy(),
            'w_cmd': self.ctrl_orig.w_cmd.copy(),
            'pos_error': self.ctrl_orig.pos_sp - self.quad_orig.pos,
            'vel_error': self.ctrl_orig.vel_sp - self.quad_orig.vel
        }
        
        return state_info, control_info
        
    def _step_configurable(self, t):
        """Step the configurable framework."""
        # Get desired states  
        sDes = self.traj_config.desiredState(t, self.dt, self.quad_config)
        
        # Run controller
        self.ctrl_config.controller(self.traj_config, self.quad_config, sDes, self.dt)
        
        # Update dynamics
        self.quad_config.update(t, self.dt, self.ctrl_config.w_cmd, self.wind_config)
        
        # Collect state info
        state_info = {
            'pos': self.quad_config.pos.copy(),
            'vel': self.quad_config.vel.copy(), 
            'euler': self.quad_config.euler.copy(),
            'omega': self.quad_config.omega.copy()
        }
        
        # Collect control info
        control_info = {
            'thrust_sp': self.ctrl_config.thrust_sp.copy(),
            'rate_sp': self.ctrl_config.rate_sp.copy(),
            'w_cmd': self.ctrl_config.w_cmd.copy(),
            'pos_error': self.ctrl_config.pos_sp - self.quad_config.pos,
            'vel_error': self.ctrl_config.vel_sp - self.quad_config.vel
        }
        
        return state_info, control_info
        
    def analyze_results(self):
        """Analyze and display comparison results."""
        print("\\n" + "="*60)
        print("RESULTS ANALYSIS")
        print("="*60)
        
        # Final positions
        orig_final_pos = self.debug_log['original']['state'][-1]['pos']
        config_final_pos = self.debug_log['configurable']['state'][-1]['pos']
        
        orig_final_error = np.linalg.norm(orig_final_pos - self.target_pos)
        config_final_error = np.linalg.norm(config_final_pos - self.target_pos)
        
        print(f"Target position: {self.target_pos}")
        print(f"Original final position: {orig_final_pos}")
        print(f"Original final error: {orig_final_error:.4f} m")
        print(f"Configurable final position: {config_final_pos}")
        print(f"Configurable final error: {config_final_error:.4f} m")
        
        # Success criteria
        success_threshold = 0.25
        orig_success = orig_final_error < success_threshold
        config_success = config_final_error < success_threshold
        
        print(f"\\nSuccess (< {success_threshold}m):")
        print(f"Original: {'✓' if orig_success else '✗'}")
        print(f"Configurable: {'✓' if config_success else '✗'}")
        
        # Analyze control signals
        self._analyze_control_signals()
        
    def _analyze_control_signals(self):
        """Analyze control signal differences."""
        print("\\n" + "-"*40)
        print("CONTROL SIGNAL ANALYSIS")
        print("-"*40)
        
        # Get final control values
        orig_ctrl = self.debug_log['original']['control'][-1]
        config_ctrl = self.debug_log['configurable']['control'][-1]
        
        print("Final thrust setpoints:")
        print(f"Original: {orig_ctrl['thrust_sp']}")
        print(f"Configurable: {config_ctrl['thrust_sp']}")
        
        print("\\nFinal motor commands:")
        print(f"Original: {orig_ctrl['w_cmd']}")
        print(f"Configurable: {config_ctrl['w_cmd']}")
        
        print("\\nFinal position errors:")
        print(f"Original: {orig_ctrl['pos_error']}")
        print(f"Configurable: {config_ctrl['pos_error']}")
        
    def plot_comparison(self):
        """Plot comparison results."""
        time_array = np.array(self.debug_log['time'])
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Original vs Configurable Framework Comparison')
        
        # Position comparison
        for i, axis in enumerate(['X', 'Y', 'Z']):
            orig_pos = [state['pos'][i] for state in self.debug_log['original']['state']]
            config_pos = [state['pos'][i] for state in self.debug_log['configurable']['state']]
            
            axes[0,i].plot(time_array, orig_pos, 'b-', label='Original', linewidth=2)
            axes[0,i].plot(time_array, config_pos, 'r--', label='Configurable', linewidth=2)
            axes[0,i].axhline(y=self.target_pos[i], color='g', linestyle=':', label='Target')
            axes[0,i].set_title(f'Position {axis}')
            axes[0,i].set_xlabel('Time (s)')
            axes[0,i].set_ylabel('Position (m)')
            axes[0,i].legend()
            axes[0,i].grid(True)
        
        # Error comparison
        axes[1,0].plot(time_array, self.debug_log['original']['error'], 'b-', label='Original', linewidth=2)
        axes[1,0].plot(time_array, self.debug_log['configurable']['error'], 'r--', label='Configurable', linewidth=2)
        axes[1,0].axhline(y=0.25, color='g', linestyle=':', label='Success threshold')
        axes[1,0].set_title('Position Error')
        axes[1,0].set_xlabel('Time (s)')
        axes[1,0].set_ylabel('Error (m)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        axes[1,0].set_yscale('log')
        
        # Motor commands comparison
        orig_w_cmd = [ctrl['w_cmd'][0] for ctrl in self.debug_log['original']['control']]
        config_w_cmd = [ctrl['w_cmd'][0] for ctrl in self.debug_log['configurable']['control']]
        
        axes[1,1].plot(time_array, orig_w_cmd, 'b-', label='Original', linewidth=2)
        axes[1,1].plot(time_array, config_w_cmd, 'r--', label='Configurable', linewidth=2)
        axes[1,1].set_title('Motor Command (Motor 1)')
        axes[1,1].set_xlabel('Time (s)')
        axes[1,1].set_ylabel('Motor Speed (rad/s)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Thrust comparison
        orig_thrust = [np.linalg.norm(ctrl['thrust_sp']) for ctrl in self.debug_log['original']['control']]
        config_thrust = [np.linalg.norm(ctrl['thrust_sp']) for ctrl in self.debug_log['configurable']['control']]
        
        axes[1,2].plot(time_array, orig_thrust, 'b-', label='Original', linewidth=2)
        axes[1,2].plot(time_array, config_thrust, 'r--', label='Configurable', linewidth=2)
        axes[1,2].set_title('Total Thrust Command')
        axes[1,2].set_xlabel('Time (s)')
        axes[1,2].set_ylabel('Thrust (N)')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Debug controller integration')
    parser.add_argument('--time', type=float, default=10, help='Simulation time (s)')
    parser.add_argument('--target', type=str, default='0,0,1', help='Target position x,y,z')
    parser.add_argument('--plot', action='store_true', help='Show plots')
    
    args = parser.parse_args()
    
    # Parse target position
    target_pos = [float(x) for x in args.target.split(',')]
    
    # Create debugger
    debugger = ControllerDebugger(target_pos, args.time)
    
    # Compare parameters
    debugger.compare_parameters()
    
    # Run comparison
    debugger.run_comparison()
    
    # Analyze results
    debugger.analyze_results()
    
    # Show plots if requested
    if args.plot:
        debugger.plot_comparison()

if __name__ == "__main__":
    main()