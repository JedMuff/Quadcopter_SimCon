#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Controller Debugging and Comparison Script

This script runs both the original and generalized controllers with identical
conditions and compares their outputs to identify discrepancies.

Author: Debug Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from scipy.spatial.distance import euclidean

# Import both controller types
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from trajectory import Trajectory
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from utils.windModel import Wind
import utils
import config

class ControllerDebugger:
    """Debug and compare controller performance."""
    
    def __init__(self):
        self.data = {
            'original': {'time': [], 'pos': [], 'vel': [], 'euler': [], 'omega': [], 
                        'thrust_sp': [], 'rate_sp': [], 'motor_cmds': [], 'pos_error': [],
                        'vel_error': [], 'att_error': []},
            'generalized': {'time': [], 'pos': [], 'vel': [], 'euler': [], 'omega': [], 
                           'thrust_sp': [], 'rate_sp': [], 'motor_cmds': [], 'pos_error': [],
                           'vel_error': [], 'att_error': []}
        }
        
    def run_comparison_simulation(self, duration=10.0, Ts=0.01):
        """Run both controllers with identical conditions."""
        
        print("Setting up comparison simulation...")
        
        # Common parameters
        Ti = 0
        Tf = duration
        t_all = np.arange(Ti, Tf + Ts, Ts)
        
        # Create identical trajectory for both systems
        print("Creating trajectory...")
        trajSelect = np.zeros(3)
        trajSelect[0] = 5  # minimum jerk trajectory
        trajSelect[1] = 3  # follow yaw
        trajSelect[2] = 1  # use average speed
        
        ctrlType = "xyz_pos"
        waypoints = np.array([
            [0, 0, 0, 0],
            [2, 0, -2, 0], 
            [2, 2, -2, np.pi/2],
            [0, 2, -2, np.pi],
            [0, 0, -2, 0]
        ])
        
        # Create a dummy quad for trajectory generation (using original)
        dummy_quad = OriginalQuadcopter(Ti)
        traj = Trajectory(dummy_quad, ctrlType, trajSelect)
        
        # Wind model (identical for both)
        wind = Wind("None", 0, 0, 0)
        
        # Setup original system
        print("Setting up original system...")
        quad_orig = OriginalQuadcopter(Ti)
        ctrl_orig = OriginalControl(quad_orig, yawType=1)
        
        # Setup generalized system with matched configuration
        print("Setting up generalized system...")
        config_dict = {
            "type": "matched",
            "arm_length": "matched", 
            "prop_size": "matched"
        }
        quad_gen = self._create_matched_drone(Ti)
        ctrl_gen = GeneralizedControl(quad_gen, yawType=1)
        
        print("Running simulations...")
        
        # Run original system
        print("  Running original controller...")
        self._run_single_simulation("original", quad_orig, ctrl_orig, wind, traj, t_all, Ts)
        
        # Reset and run generalized system  
        print("  Running generalized controller...")
        quad_gen = self._create_matched_drone(Ti)  # Reset state
        ctrl_gen = GeneralizedControl(quad_gen, yawType=1)
        self._run_single_simulation("generalized", quad_gen, ctrl_gen, wind, traj, t_all, Ts)
        
        print("Simulations complete. Analyzing results...")
        self._analyze_results()
        
    def _create_matched_drone(self, Ti):
        """Create drone that matches original quadcopter parameters."""
        propeller_config = create_standard_propeller_config(
            config_type="quad",
            arm_length=0.175,  # From original params  
            prop_size="matched"
        )
        return ConfigurableQuadcopter(Ti, propellers=propeller_config)
        
    def _run_single_simulation(self, system_name, quad, ctrl, wind, traj, t_all, Ts):
        """Run simulation for one controller system."""
        
        data = self.data[system_name]
        
        for t in t_all:
            # Get desired state
            sDes = traj.desiredState(t, Ts, quad)
            
            # Store pre-control state
            data['time'].append(t)
            data['pos'].append(quad.pos.copy())
            data['vel'].append(quad.vel.copy())
            data['euler'].append(utils.quatToYPR_ZYX(quad.quat).copy())
            data['omega'].append(quad.omega.copy())
            
            # Calculate control
            ctrl.controller(traj, quad, sDes, Ts)
            
            # Store control outputs
            data['thrust_sp'].append(ctrl.thrust_sp.copy())
            data['rate_sp'].append(ctrl.rate_sp.copy())
            data['motor_cmds'].append(ctrl.w_cmd[:4].copy())  # First 4 for comparison
            
            # Calculate errors
            pos_error = np.linalg.norm(ctrl.pos_sp - quad.pos)
            vel_error = np.linalg.norm(ctrl.vel_sp - quad.vel)
            att_error = np.linalg.norm(ctrl.rate_sp - quad.omega)
            
            data['pos_error'].append(pos_error)
            data['vel_error'].append(vel_error) 
            data['att_error'].append(att_error)
            
            # Update quad state
            quad.update(t, Ts, ctrl.w_cmd, wind)
            
        # Convert to numpy arrays
        for key in data:
            if key != 'time':
                data[key] = np.array(data[key])
            else:
                data[key] = np.array(data[key])
                
    def _analyze_results(self):
        """Analyze and compare the results."""
        
        orig = self.data['original']
        gen = self.data['generalized']
        
        print("\n" + "="*60)
        print("CONTROLLER COMPARISON ANALYSIS")
        print("="*60)
        
        # Final position comparison
        final_pos_orig = orig['pos'][-1]
        final_pos_gen = gen['pos'][-1]
        final_pos_diff = np.linalg.norm(final_pos_orig - final_pos_gen)
        
        print(f"\nFinal Position Comparison:")
        print(f"  Original:    [{final_pos_orig[0]:.3f}, {final_pos_orig[1]:.3f}, {final_pos_orig[2]:.3f}]")
        print(f"  Generalized: [{final_pos_gen[0]:.3f}, {final_pos_gen[1]:.3f}, {final_pos_gen[2]:.3f}]")
        print(f"  Difference:  {final_pos_diff:.3f}m")
        
        # Trajectory following errors
        avg_pos_error_orig = np.mean(orig['pos_error'])
        avg_pos_error_gen = np.mean(gen['pos_error'])
        max_pos_error_orig = np.max(orig['pos_error'])
        max_pos_error_gen = np.max(gen['pos_error'])
        
        print(f"\nTrajectory Following Performance:")
        print(f"  Original Controller:")
        print(f"    Average position error: {avg_pos_error_orig:.3f}m")
        print(f"    Maximum position error: {max_pos_error_orig:.3f}m")
        print(f"  Generalized Controller:")
        print(f"    Average position error: {avg_pos_error_gen:.3f}m")
        print(f"    Maximum position error: {max_pos_error_gen:.3f}m")
        
        # Control signal analysis
        print(f"\nControl Signal Analysis:")
        
        # Thrust commands
        thrust_mag_orig = np.linalg.norm(orig['thrust_sp'], axis=1)
        thrust_mag_gen = np.linalg.norm(gen['thrust_sp'], axis=1)
        print(f"  Thrust magnitude - Original: {np.mean(thrust_mag_orig):.3f}±{np.std(thrust_mag_orig):.3f}N")
        print(f"  Thrust magnitude - Generalized: {np.mean(thrust_mag_gen):.3f}±{np.std(thrust_mag_gen):.3f}N")
        
        # Motor commands
        motor_orig = np.mean(orig['motor_cmds'], axis=0)
        motor_gen = np.mean(gen['motor_cmds'], axis=0)
        print(f"  Average motor speeds:")
        print(f"    Original:    [{motor_orig[0]:.1f}, {motor_orig[1]:.1f}, {motor_orig[2]:.1f}, {motor_orig[3]:.1f}] rad/s")
        print(f"    Generalized: [{motor_gen[0]:.1f}, {motor_gen[1]:.1f}, {motor_gen[2]:.1f}, {motor_gen[3]:.1f}] rad/s")
        
        # Detect potential issues
        print(f"\nPotential Issues Detected:")
        
        if avg_pos_error_gen > 2 * avg_pos_error_orig:
            print(f"  ⚠️  Generalized controller has {avg_pos_error_gen/avg_pos_error_orig:.1f}x worse position tracking")
            
        motor_diff = np.linalg.norm(motor_orig - motor_gen)
        if motor_diff > 50:  # rad/s threshold
            print(f"  ⚠️  Significant motor command differences: {motor_diff:.1f} rad/s RMS")
            
        thrust_diff = np.mean(np.abs(thrust_mag_orig - thrust_mag_gen))
        if thrust_diff > 1.0:  # Newton threshold
            print(f"  ⚠️  Significant thrust command differences: {thrust_diff:.1f}N average")
            
        if final_pos_diff > 0.5:  # meter threshold
            print(f"  ⚠️  Large final position difference: {final_pos_diff:.3f}m")
            
        # Generate diagnostic plots
        self._create_diagnostic_plots()
        
    def _create_diagnostic_plots(self):
        """Create diagnostic plots comparing the two controllers."""
        
        orig = self.data['original']
        gen = self.data['generalized']
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Controller Comparison Analysis', fontsize=16)
        
        # Position tracking
        axes[0,0].plot(orig['time'], orig['pos_error'], 'b-', label='Original', linewidth=2)
        axes[0,0].plot(gen['time'], gen['pos_error'], 'r--', label='Generalized', linewidth=2)
        axes[0,0].set_ylabel('Position Error (m)')
        axes[0,0].set_title('Position Tracking Error')
        axes[0,0].legend()
        axes[0,0].grid(True)
        
        # 3D trajectory
        axes[0,1].plot(orig['pos'][:,0], orig['pos'][:,1], 'b-', label='Original', linewidth=2)
        axes[0,1].plot(gen['pos'][:,0], gen['pos'][:,1], 'r--', label='Generalized', linewidth=2)
        axes[0,1].set_xlabel('X (m)')
        axes[0,1].set_ylabel('Y (m)')
        axes[0,1].set_title('XY Trajectory')
        axes[0,1].legend()
        axes[0,1].grid(True)
        axes[0,1].axis('equal')
        
        # Thrust commands
        thrust_orig = np.linalg.norm(orig['thrust_sp'], axis=1)
        thrust_gen = np.linalg.norm(gen['thrust_sp'], axis=1)
        axes[1,0].plot(orig['time'], thrust_orig, 'b-', label='Original', linewidth=2)
        axes[1,0].plot(gen['time'], thrust_gen, 'r--', label='Generalized', linewidth=2)
        axes[1,0].set_ylabel('Thrust Magnitude (N)')
        axes[1,0].set_title('Thrust Commands')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Motor commands
        axes[1,1].plot(orig['time'], orig['motor_cmds'][:,0], 'b-', label='M1 Original', alpha=0.7)
        axes[1,1].plot(orig['time'], orig['motor_cmds'][:,1], 'b--', label='M2 Original', alpha=0.7)
        axes[1,1].plot(gen['time'], gen['motor_cmds'][:,0], 'r-', label='M1 Generalized', alpha=0.7)
        axes[1,1].plot(gen['time'], gen['motor_cmds'][:,1], 'r--', label='M2 Generalized', alpha=0.7)
        axes[1,1].set_ylabel('Motor Speed (rad/s)')
        axes[1,1].set_title('Motor Commands (M1, M2)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Altitude tracking
        axes[2,0].plot(orig['time'], orig['pos'][:,2], 'b-', label='Original', linewidth=2)
        axes[2,0].plot(gen['time'], gen['pos'][:,2], 'r--', label='Generalized', linewidth=2)
        axes[2,0].set_ylabel('Altitude (m)')
        axes[2,0].set_xlabel('Time (s)')
        axes[2,0].set_title('Altitude Tracking')
        axes[2,0].legend()
        axes[2,0].grid(True)
        
        # Angular rates
        axes[2,1].plot(orig['time'], orig['omega'][:,0], 'b-', label='Roll Rate Original', alpha=0.7)
        axes[2,1].plot(orig['time'], orig['omega'][:,1], 'b--', label='Pitch Rate Original', alpha=0.7)
        axes[2,1].plot(gen['time'], gen['omega'][:,0], 'r-', label='Roll Rate Generalized', alpha=0.7)
        axes[2,1].plot(gen['time'], gen['omega'][:,1], 'r--', label='Pitch Rate Generalized', alpha=0.7)
        axes[2,1].set_ylabel('Angular Rate (rad/s)')
        axes[2,1].set_xlabel('Time (s)')
        axes[2,1].set_title('Angular Rates')
        axes[2,1].legend()
        axes[2,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('controller_comparison.png', dpi=150, bbox_inches='tight')
        print("\n📊 Diagnostic plots saved as 'controller_comparison.png'")
        
    def save_detailed_data(self, filename='controller_debug_data.json'):
        """Save detailed data for further analysis."""
        
        # Convert numpy arrays to lists for JSON serialization
        data_serializable = {}
        for system in self.data:
            data_serializable[system] = {}
            for key, value in self.data[system].items():
                if isinstance(value, np.ndarray):
                    data_serializable[system][key] = value.tolist()
                else:
                    data_serializable[system][key] = value
                    
        with open(filename, 'w') as f:
            json.dump(data_serializable, f, indent=2)
            
        print(f"📁 Detailed data saved to '{filename}'")

def main():
    """Run the controller comparison."""
    
    print("Controller Debugging and Comparison Tool")
    print("="*50)
    
    debugger = ControllerDebugger()
    
    try:
        debugger.run_comparison_simulation(duration=15.0, Ts=0.01)
        debugger.save_detailed_data()
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error during simulation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()