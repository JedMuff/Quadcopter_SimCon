#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Parameter Verification Script

This script performs deep parameter comparison between the original and configurable
quadcopter systems to identify ALL parameter mismatches that could cause trajectory
divergence despite successful single-step control matching.

Focus Areas:
1. Core dynamics parameters (mass, inertia, gravity)
2. Motor dynamics parameters (tau, kp, damp)
3. Thrust/torque coefficients and motor limits
4. Allocation matrices and mixer matrices
5. Wind model parameters
6. Integration method differences
7. State representation differences

Author: Parameter Analysis Debug
"""

import numpy as np
import json
from scipy.spatial.distance import euclidean

# Import both systems
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from utils.windModel import Wind
import config

class ParameterVerifier:
    """Comprehensive parameter verification between original and configurable systems."""
    
    def __init__(self):
        self.tolerance = 1e-10  # Strict tolerance for parameter matching
        self.differences = {}
        self.critical_issues = []
        self.warnings = []
        
    def run_comprehensive_verification(self):
        """Run complete parameter verification analysis."""
        
        print("="*80)
        print("COMPREHENSIVE PARAMETER VERIFICATION")
        print("="*80)
        
        # Create both systems
        print("\n1. Creating system instances...")
        orig_quad = self._create_original_system()
        config_quad = self._create_configurable_system()
        
        # Force parameter matching
        print("\n2. Forcing parameter matching...")
        config_quad.force_original_parameters()
        
        # Comprehensive parameter comparison
        print("\n3. Comparing all parameters...")
        self._compare_physical_parameters(orig_quad, config_quad)
        self._compare_motor_parameters(orig_quad, config_quad)
        self._compare_mixer_matrices(orig_quad, config_quad)
        self._compare_dynamics_parameters(orig_quad, config_quad)
        self._compare_integration_methods(orig_quad, config_quad)
        self._compare_state_representations(orig_quad, config_quad)
        
        # Test single-step dynamics
        print("\n4. Testing single-step dynamics...")
        self._test_single_step_dynamics(orig_quad, config_quad)
        
        # Test motor command processing
        print("\n5. Testing motor command processing...")
        self._test_motor_command_processing(orig_quad, config_quad)
        
        # Test allocation matrix application
        print("\n6. Testing allocation matrix application...")
        self._test_allocation_matrices(orig_quad, config_quad)
        
        # Wind model verification
        print("\n7. Verifying wind model...")
        self._test_wind_model()
        
        # Summary report
        print("\n8. Generating comprehensive report...")
        self._generate_verification_report()
        
    def _create_original_system(self):
        """Create original quadcopter system."""
        return OriginalQuadcopter(0)
        
    def _create_configurable_system(self):
        """Create configurable system with matched parameters."""
        # Get original system to match its arm length
        orig_temp = OriginalQuadcopter(0)
        arm_length = orig_temp.params["dxm"]  # Use original arm length
        
        propeller_config = create_standard_propeller_config(
            config_type="quad",
            arm_length=arm_length,
            prop_size="matched"
        )
        return ConfigurableQuadcopter(0, propellers=propeller_config)
        
    def _compare_physical_parameters(self, orig, config):
        """Compare physical parameters (mass, inertia, gravity)."""
        
        print("   Comparing physical parameters...")
        
        # Mass comparison
        mass_diff = abs(orig.params["mB"] - config.params["mB"])
        self._record_difference("mass", orig.params["mB"], config.params["mB"], mass_diff)
        
        # Gravity comparison
        gravity_diff = abs(orig.params["g"] - config.params["g"])
        self._record_difference("gravity", orig.params["g"], config.params["g"], gravity_diff)
        
        # Inertia matrix comparison
        inertia_diff = np.linalg.norm(orig.params["IB"] - config.params["IB"])
        self._record_difference("inertia_matrix", orig.params["IB"], config.params["IB"], inertia_diff)
        
        # Inverse inertia comparison
        inv_inertia_diff = np.linalg.norm(orig.params["invI"] - config.params["invI"])
        self._record_difference("inverse_inertia", orig.params["invI"], config.params["invI"], inv_inertia_diff)
        
        # Center of mass/geometry
        dxm_diff = abs(orig.params["dxm"] - config.params["dxm"])
        dym_diff = abs(orig.params["dym"] - config.params["dym"])
        self._record_difference("arm_length_x", orig.params["dxm"], config.params["dxm"], dxm_diff)
        self._record_difference("arm_length_y", orig.params["dym"], config.params["dym"], dym_diff)
        
        # Drag coefficient
        cd_diff = abs(orig.params["Cd"] - config.params["Cd"])
        self._record_difference("drag_coefficient", orig.params["Cd"], config.params["Cd"], cd_diff)
        
    def _compare_motor_parameters(self, orig, config):
        """Compare motor dynamics parameters."""
        
        print("   Comparing motor parameters...")
        
        # Thrust and torque coefficients
        kth_diff = abs(orig.params["kTh"] - config.params["kTh"])
        kto_diff = abs(orig.params["kTo"] - config.params["kTo"])
        self._record_difference("thrust_coefficient", orig.params["kTh"], config.params["kTh"], kth_diff)
        self._record_difference("torque_coefficient", orig.params["kTo"], config.params["kTo"], kto_diff)
        
        # Motor dynamics parameters
        tau_diff = abs(orig.params["tau"] - config.params["tau"])
        kp_diff = abs(orig.params["kp"] - config.params["kp"])
        damp_diff = abs(orig.params["damp"] - config.params["damp"])
        self._record_difference("motor_tau", orig.params["tau"], config.params["tau"], tau_diff)
        self._record_difference("motor_kp", orig.params["kp"], config.params["kp"], kp_diff)
        self._record_difference("motor_damp", orig.params["damp"], config.params["damp"], damp_diff)
        
        # Motor limits
        min_w_diff = abs(orig.params["minWmotor"] - config.params["minWmotor"])
        max_w_diff = abs(orig.params["maxWmotor"] - config.params["maxWmotor"])
        min_thr_diff = abs(orig.params["minThr"] - config.params["minThr"])
        max_thr_diff = abs(orig.params["maxThr"] - config.params["maxThr"])
        
        self._record_difference("min_motor_speed", orig.params["minWmotor"], config.params["minWmotor"], min_w_diff)
        self._record_difference("max_motor_speed", orig.params["maxWmotor"], config.params["maxWmotor"], max_w_diff)
        self._record_difference("min_thrust", orig.params["minThr"], config.params["minThr"], min_thr_diff)
        self._record_difference("max_thrust", orig.params["maxThr"], config.params["maxThr"], max_thr_diff)
        
        # Hover parameters
        if "w_hover" in orig.params and "w_hover" in config.params:
            w_hover_diff = abs(orig.params["w_hover"] - config.params["w_hover"])
            self._record_difference("hover_motor_speed", orig.params["w_hover"], config.params["w_hover"], w_hover_diff)
            
        if "thr_hover" in orig.params and "thr_hover" in config.params:
            thr_hover_diff = abs(orig.params["thr_hover"] - config.params["thr_hover"])
            self._record_difference("hover_thrust", orig.params["thr_hover"], config.params["thr_hover"], thr_hover_diff)
            
    def _compare_mixer_matrices(self, orig, config):
        """Compare mixer matrices and allocation matrices."""
        
        print("   Comparing mixer/allocation matrices...")
        
        # Mixer matrix comparison
        mixer_diff = np.linalg.norm(orig.params["mixerFM"] - config.params["mixerFM"])
        self._record_difference("mixer_matrix", orig.params["mixerFM"], config.params["mixerFM"], mixer_diff)
        
        # Inverse mixer matrix comparison
        mixer_inv_diff = np.linalg.norm(orig.params["mixerFMinv"] - config.params["mixerFMinv"])
        self._record_difference("mixer_inverse", orig.params["mixerFMinv"], config.params["mixerFMinv"], mixer_inv_diff)
        
        # Check if configurable system has allocation matrices
        if hasattr(config.drone_sim, 'Bf') and hasattr(config.drone_sim, 'Bm'):
            # Compare allocation matrices if available
            print("     Found allocation matrices in configurable system")
            
            # The allocation matrices should produce equivalent results to mixer when properly converted
            # This is a complex comparison that requires understanding the conversion process
            
    def _compare_dynamics_parameters(self, orig, config):
        """Compare dynamics calculation parameters."""
        
        print("   Comparing dynamics parameters...")
        
        # Rotor inertia for gyroscopic effects
        if "IRzz" in orig.params and "IRzz" in config.params:
            irz_diff = abs(orig.params["IRzz"] - config.params["IRzz"])
            self._record_difference("rotor_inertia", orig.params["IRzz"], config.params["IRzz"], irz_diff)
            
        # Check configuration settings that affect dynamics
        config_differences = []
        
        # Import the global config to check settings
        import config as global_config
        print(f"     Original uses orientation: {global_config.orient}")
        print(f"     Original uses precession: {global_config.usePrecession}")
        
    def _compare_integration_methods(self, orig, config):
        """Compare integration methods between systems."""
        
        print("   Comparing integration methods...")
        
        # Original uses scipy.integrate.ode with dopri5
        print("     Original system: scipy.integrate.ode with dopri5")
        print("     Original tolerances: atol=10e-6, rtol=10e-6, first_step=0.00005")
        
        # Configurable uses RK4 with fixed step
        print("     Configurable system: RK4 with fixed time step")
        print(f"     Configurable dt: {config.drone_sim.dt}")
        
        self.warnings.append("Integration method difference: Original uses adaptive dopri5, Configurable uses fixed RK4")
        
    def _compare_state_representations(self, orig, config):
        """Compare state vector representations."""
        
        print("   Comparing state representations...")
        
        print(f"     Original state vector length: {len(orig.state)}")
        print(f"     Configurable state vector length: {len(config.drone_sim.state)}")
        
        # Original: [x,y,z, q0,q1,q2,q3, vx,vy,vz, p,q,r, wM1,wM1dot,wM2,wM2dot,wM3,wM3dot,wM4,wM4dot]
        # Configurable: [x,y,z, vx,vy,vz, phi,theta,psi, p,q,r]
        
        print("     Original: includes motor states and uses quaternions")
        print("     Configurable: simplified state with Euler angles")
        
        self.warnings.append("State representation difference: Original includes motor dynamics, Configurable does not")
        
    def _test_single_step_dynamics(self, orig, config):
        """Test single time step dynamics with identical inputs."""
        
        print("   Testing single-step dynamics...")
        
        # Set identical initial conditions
        wind = Wind("None", 0, 0, 0)
        motor_cmd = np.array([400, 400, 400, 400])  # Fixed motor command
        
        # Get initial states
        orig_pos_initial = orig.pos.copy()
        config_pos_initial = config.pos.copy()
        
        # Single step update
        orig.update(0, 0.01, motor_cmd, wind)
        config.update(0, 0.01, motor_cmd, wind)
        
        # Compare position changes
        orig_pos_change = np.linalg.norm(orig.pos - orig_pos_initial)
        config_pos_change = np.linalg.norm(config.pos - config_pos_initial)
        
        pos_change_diff = abs(orig_pos_change - config_pos_change)
        self._record_difference("single_step_position_change", orig_pos_change, config_pos_change, pos_change_diff)
        
        # Compare final positions
        final_pos_diff = np.linalg.norm(orig.pos - config.pos)
        self._record_difference("single_step_final_position", orig.pos, config.pos, final_pos_diff)
        
    def _test_motor_command_processing(self, orig, config):
        """Test how motor commands are processed differently."""
        
        print("   Testing motor command processing...")
        
        test_cmd = np.array([400, 500, 350, 450])
        
        # Original system processes commands through motor dynamics
        # Configurable system may process differently
        
        # Test thrust calculation
        orig.wMotor = test_cmd
        orig.forces()
        orig_thrust = orig.thr.copy()
        
        # For configurable, we need to simulate the command processing
        # This is complex due to different state representations
        
        print(f"     Original thrust from test command: {orig_thrust}")
        
    def _test_allocation_matrices(self, orig, config):
        """Test allocation matrix application."""
        
        print("   Testing allocation matrices...")
        
        if hasattr(config.drone_sim, 'Bf') and hasattr(config.drone_sim, 'Bm'):
            # Test allocation matrix dimensions and values
            Bf = config.drone_sim.Bf
            Bm = config.drone_sim.Bm
            
            print(f"     Force allocation matrix shape: {Bf.shape}")
            print(f"     Moment allocation matrix shape: {Bm.shape}")
            
            # Test with unit commands
            unit_cmd = np.ones(4)
            forces = Bf @ (unit_cmd**2)
            moments = Bm @ (unit_cmd**2)
            
            print(f"     Forces from unit commands: {forces}")
            print(f"     Moments from unit commands: {moments}")
            
    def _test_wind_model(self):
        """Test wind model consistency."""
        
        print("   Testing wind model...")
        
        wind = Wind("None", 0, 0, 0)
        wind_data = wind.randomWind(0)
        
        print(f"     Wind model output: {wind_data}")
        
    def _record_difference(self, param_name, orig_value, config_value, difference):
        """Record parameter difference for analysis."""
        
        # Handle scalar vs array relative error calculation
        if np.isscalar(orig_value):
            relative_error = difference / abs(orig_value) if abs(orig_value) > 0 else float('inf')
        else:
            # For arrays, use norm for relative error
            orig_norm = np.linalg.norm(orig_value)
            relative_error = difference / orig_norm if orig_norm > 0 else float('inf')
        
        self.differences[param_name] = {
            'original': orig_value,
            'configurable': config_value,
            'difference': difference,
            'relative_error': relative_error
        }
        
        if difference > self.tolerance:
            if difference > 1e-6:  # Significant difference
                self.critical_issues.append(f"{param_name}: difference = {difference:.2e}")
            else:
                self.warnings.append(f"{param_name}: small difference = {difference:.2e}")
                
    def _generate_verification_report(self):
        """Generate comprehensive verification report."""
        
        print("\n" + "="*80)
        print("PARAMETER VERIFICATION REPORT")
        print("="*80)
        
        # Critical issues
        if self.critical_issues:
            print(f"\n🚨 CRITICAL ISSUES FOUND ({len(self.critical_issues)}):")
            for issue in self.critical_issues:
                print(f"   ❌ {issue}")
        else:
            print("\n✅ No critical parameter differences found")
            
        # Warnings
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
                
        # Detailed parameter analysis
        print(f"\n📊 DETAILED PARAMETER ANALYSIS:")
        print(f"   Total parameters compared: {len(self.differences)}")
        
        # Group by significance
        exact_matches = []
        small_differences = []
        significant_differences = []
        
        for param, data in self.differences.items():
            diff = data['difference']
            if diff == 0:
                exact_matches.append(param)
            elif diff < 1e-10:
                small_differences.append((param, diff))
            else:
                significant_differences.append((param, diff))
                
        print(f"   Exact matches: {len(exact_matches)}")
        print(f"   Small differences (<1e-10): {len(small_differences)}")
        print(f"   Significant differences (>=1e-10): {len(significant_differences)}")
        
        # Show significant differences in detail
        if significant_differences:
            print(f"\n🔍 SIGNIFICANT DIFFERENCES:")
            for param, diff in significant_differences:
                data = self.differences[param]
                print(f"   {param}:")
                print(f"     Original: {data['original']}")
                print(f"     Configurable: {data['configurable']}")
                print(f"     Absolute difference: {diff:.2e}")
                print(f"     Relative error: {data['relative_error']:.2e}")
                
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        if len(significant_differences) > 0:
            print("   1. Fix significant parameter differences in force_original_parameters()")
            print("   2. Ensure allocation matrices produce equivalent mixer results")
            print("   3. Verify motor command scaling and processing")
            
        if "State representation difference" in [w for w in self.warnings if "State representation" in w]:
            print("   4. Consider implementing motor dynamics in configurable system")
            
        if "Integration method difference" in [w for w in self.warnings if "Integration method" in w]:
            print("   5. Consider using same integration method for exact matching")
            
        # Save detailed report
        self._save_detailed_report()
        
    def _save_detailed_report(self):
        """Save detailed analysis to file."""
        
        report = {
            'summary': {
                'total_parameters': len(self.differences),
                'critical_issues': len(self.critical_issues),
                'warnings': len(self.warnings)
            },
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'parameter_differences': {}
        }
        
        # Convert numpy arrays to lists for JSON serialization
        for param, data in self.differences.items():
            report['parameter_differences'][param] = {
                'original': data['original'].tolist() if isinstance(data['original'], np.ndarray) else data['original'],
                'configurable': data['configurable'].tolist() if isinstance(data['configurable'], np.ndarray) else data['configurable'],
                'difference': float(data['difference']),
                'relative_error': float(data['relative_error'])
            }
            
        with open('parameter_verification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📄 Detailed report saved to 'parameter_verification_report.json'")


def main():
    """Run comprehensive parameter verification."""
    
    print("Comprehensive Parameter Verification Tool")
    print("This tool identifies ALL parameter mismatches that could cause trajectory divergence")
    print()
    
    verifier = ParameterVerifier()
    
    try:
        verifier.run_comprehensive_verification()
        
    except Exception as e:
        print(f"\n❌ Error during verification: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()