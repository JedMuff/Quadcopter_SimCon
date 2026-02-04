"""
Drone Configuration Class

This module provides automatic computation of drone physical properties and allocation
matrices from propeller configurations. Integrates algorithms from drone-hover package
for realistic mass, center of gravity, inertia, and control allocation calculation.
"""

import numpy as np
from numpy.linalg import norm, inv
from .propeller_data import (
    get_propeller_specs, validate_propeller_config, 
    GRAVITY, CONTROLLER_MASS, BEAM_DENSITY
)

class DroneConfiguration:
    """
    Automatically compute drone physical properties from propeller configuration.
    
    This class takes a list of propeller specifications and computes:
    - Total mass including controller, motors, and structure
    - Center of gravity location
    - Inertia matrix (Ix, Iy, Iz, Ixy, Ixz, Iyz)
    - Force allocation matrix (Bf) mapping motor commands to body forces
    - Moment allocation matrix (Bm) mapping motor commands to body moments
    """
    
    def __init__(self, propellers):
        """
        Initialize drone configuration from propeller specifications.
        
        Args:
            propellers (list): List of propeller dictionaries, each containing:
                - "loc": [x, y, z] position in body frame (meters)
                - "dir": [x, y, z, rotation] thrust direction and spin direction
                - "propsize": propeller size in inches (4-8)
                
        Example:
            propellers = [
                {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
                # ... more propellers
            ]
        """
        # Validate input format
        validate_propeller_config(propellers)
        
        self.propellers = propellers
        self.num_motors = len(propellers)
        
        # Add propeller specifications to each propeller
        self._add_propeller_specs()
        
        # Compute physical properties
        self._compute_mass_and_cg()
        self._compute_inertia()
        self._compute_allocation_matrices()
    
    def _add_propeller_specs(self):
        """Add force/moment constants and specifications to each propeller."""
        for prop in self.propellers:
            specs = get_propeller_specs(prop["propsize"])
            prop["constants"] = specs["constants"]  # [k_f, k_m]
            prop["wmax"] = specs["wmax"]
            prop["mass"] = specs["mass"]
    
    def _compute_mass_and_cg(self):
        """Compute total mass and center of gravity location."""
        # Start with controller mass
        self.mass = CONTROLLER_MASS
        
        # Add propeller and beam masses
        for prop in self.propellers:
            prop_mass = prop["mass"]
            beam_length = norm(np.array(prop["loc"]))
            beam_mass = BEAM_DENSITY * beam_length
            
            self.mass += prop_mass + beam_mass
        
        # Compute center of gravity
        self.cg = np.zeros(3)
        for prop in self.propellers:
            prop_mass = prop["mass"]
            beam_length = norm(np.array(prop["loc"]))
            beam_mass = BEAM_DENSITY * beam_length
            prop_loc = np.array(prop["loc"])
            
            # Propeller contributes at its location
            self.cg += (prop_mass / self.mass) * prop_loc
            
            # Beam contributes at its midpoint
            self.cg += (beam_mass / self.mass) * prop_loc * 0.5
    
    def _compute_inertia(self):
        """Compute inertia matrix components using parallel axis theorem."""
        # Controller inertia about its own center (approximated as rectangular block)
        # Typical flight controller dimensions: 105mm x 36mm x 35mm
        controller_Ix = (1/12) * CONTROLLER_MASS * (0.036**2 + 0.035**2)
        controller_Iy = (1/12) * CONTROLLER_MASS * (0.105**2 + 0.035**2)
        controller_Iz = (1/12) * CONTROLLER_MASS * (0.105**2 + 0.036**2)
        
        # Translate controller inertia to center of gravity using parallel axis theorem
        cg_offset_sq = np.dot(self.cg, self.cg)
        self.Ix = controller_Ix + CONTROLLER_MASS * (self.cg[1]**2 + self.cg[2]**2)
        self.Iy = controller_Iy + CONTROLLER_MASS * (self.cg[0]**2 + self.cg[2]**2)
        self.Iz = controller_Iz + CONTROLLER_MASS * (self.cg[0]**2 + self.cg[1]**2)
        
        # Initialize products of inertia
        self.Ixy = -CONTROLLER_MASS * self.cg[0] * self.cg[1]
        self.Ixz = -CONTROLLER_MASS * self.cg[0] * self.cg[2]
        self.Iyz = -CONTROLLER_MASS * self.cg[1] * self.cg[2]
        
        # Add contributions from propellers and beams
        for prop in self.propellers:
            prop_mass = prop["mass"]
            pos = np.array(prop["loc"])
            beam_length = norm(pos)
            beam_mass = BEAM_DENSITY * beam_length
            
            # Propeller position relative to CG
            r_prop = pos - self.cg
            
            # Propeller contributions (treated as point mass)
            self.Ix += prop_mass * (r_prop[1]**2 + r_prop[2]**2)
            self.Iy += prop_mass * (r_prop[0]**2 + r_prop[2]**2)
            self.Iz += prop_mass * (r_prop[0]**2 + r_prop[1]**2)
            self.Ixy -= prop_mass * r_prop[0] * r_prop[1]
            self.Ixz -= prop_mass * r_prop[0] * r_prop[2]
            self.Iyz -= prop_mass * r_prop[1] * r_prop[2]
            
            # Beam contributions (rod along beam direction)
            beam_center = pos * 0.5  # Beam center at midpoint
            r_beam = beam_center - self.cg
            
            # For a rod along the beam direction, add both translational and rotational inertia
            beam_direction = pos / beam_length  # Unit vector along beam
            
            # Perpendicular moment of inertia for rod: I_perp = (1/12) * m * L²
            I_beam_perp = (1/12) * beam_mass * beam_length**2
            
            # Add translational inertia (parallel axis theorem)
            self.Ix += beam_mass * (r_beam[1]**2 + r_beam[2]**2) + I_beam_perp * (beam_direction[1]**2 + beam_direction[2]**2)
            self.Iy += beam_mass * (r_beam[0]**2 + r_beam[2]**2) + I_beam_perp * (beam_direction[0]**2 + beam_direction[2]**2)
            self.Iz += beam_mass * (r_beam[0]**2 + r_beam[1]**2) + I_beam_perp * (beam_direction[0]**2 + beam_direction[1]**2)
            
            # Add products of inertia
            self.Ixy -= beam_mass * r_beam[0] * r_beam[1]
            self.Ixz -= beam_mass * r_beam[0] * r_beam[2]
            self.Iyz -= beam_mass * r_beam[1] * r_beam[2]
        
        # Create inertia matrix with numerical stability improvements
        self.inertia_matrix = np.array([
            [self.Ix, self.Ixy, self.Ixz],
            [self.Ixy, self.Iy, self.Iyz],
            [self.Ixz, self.Iyz, self.Iz]
        ])
        
        # Clean up extremely small values that cause numerical instability
        # Values smaller than 1e-12 are likely numerical noise
        tolerance = 1e-12
        self.inertia_matrix[np.abs(self.inertia_matrix) < tolerance] = 0.0
        
        # Ensure matrix is symmetric (fix any tiny asymmetries from numerical errors)
        self.inertia_matrix = 0.5 * (self.inertia_matrix + self.inertia_matrix.T)
        
        # Ensure positive definite by checking eigenvalues
        eigenvals = np.linalg.eigvals(self.inertia_matrix)
        if np.any(eigenvals <= 0):
            print(f"Warning: Non-positive eigenvalues detected: {eigenvals}")
            # Add small positive value to diagonal to ensure positive definiteness
            min_eigenval = max(1e-6, -np.min(eigenvals) + 1e-6)
            self.inertia_matrix += min_eigenval * np.eye(3)
    def _compute_allocation_matrices(self):
        """Compute force and moment allocation matrices."""
        self.Bf = np.zeros((3, self.num_motors))
        self.Bm = np.zeros((3, self.num_motors))
        
        for idx, prop in enumerate(self.propellers):
            k_f, k_m = prop["constants"]
            w_max = prop["wmax"]
            prop_loc = np.array(prop["loc"])
            prop_r = prop_loc - self.cg  # Position relative to CG
            
            # Thrust direction (normalized)
            prop_dir = np.array(prop["dir"][:3])
            prop_dir = prop_dir / norm(prop_dir)
            
            # Rotation direction (-1 for CCW, +1 for CW when viewed from above)
            prop_rot = -1 if prop["dir"][-1] == "ccw" else 1
            
            # Force allocation (thrust in prop_dir direction)
            # The allocation matrix should map normalized commands [0,1] to forces
            # When motor_command = 1.0, we want full thrust = k_f * w_max²
            max_thrust = k_f * w_max**2
            self.Bf[:, idx] = max_thrust * prop_dir
            
            # Moment allocation (cross product + propeller torque)
            # Similarly, when motor_command = 1.0, we want full moment
            moment_from_thrust = np.cross(prop_r, max_thrust * prop_dir)
            max_torque = k_m * w_max**2
            moment_from_torque = max_torque * prop_rot * prop_dir
            self.Bm[:, idx] = moment_from_thrust + moment_from_torque
        
        # Store unscaled matrices for reference
        self.Bf_unscaled = self.Bf.copy()
        self.Bm_unscaled = self.Bm.copy()
        
        # Store original matrices without pre-scaling
        # The dynamics equations will handle the mass/inertia scaling to avoid numerical issues
        
        # Combined allocation matrix for control allocation
        self.B_combined = np.vstack([self.Bf, self.Bm])  # (6 x num_motors)
        
        # Clean up tiny numerical values in allocation matrices to prevent NaN propagation
        tolerance = 1e-15
        self.Bf[np.abs(self.Bf) < tolerance] = 0.0
        self.Bm[np.abs(self.Bm) < tolerance] = 0.0
        self.B_combined[np.abs(self.B_combined) < tolerance] = 0.0
        
        # Compute pseudo-inverse with enhanced numerical stability
        try:
            self.B_pinv = np.linalg.pinv(self.B_combined, rcond=1e-10)
        except np.linalg.LinAlgError:
            print("Warning: Using fallback pseudo-inverse computation")
            # Fallback: use SVD decomposition with explicit tolerance
            U, s, Vt = np.linalg.svd(self.B_combined, full_matrices=False)
            s_inv = np.where(s > 1e-10, 1/s, 0)
            self.B_pinv = Vt.T @ np.diag(s_inv) @ U.T
        
        # Clean up tiny values in pseudo-inverse to prevent NaN propagation
        self.B_pinv[np.abs(self.B_pinv) < tolerance] = 0.0
    
    def get_physical_properties(self):
        """
        Get physical properties dictionary.
        
        Returns:
            dict: Physical properties including mass, CG, inertia components
        """
        return {
            'mass': self.mass,
            'center_of_gravity': self.cg.tolist(),
            'Ix': self.Ix,
            'Iy': self.Iy, 
            'Iz': self.Iz,
            'Ixy': self.Ixy,
            'Ixz': self.Ixz,
            'Iyz': self.Iyz,
            'inertia_matrix': self.inertia_matrix.tolist(),
            'num_motors': self.num_motors
        }
    
    def get_allocation_matrices(self):
        """
        Get force and moment allocation matrices.
        
        Returns:
            tuple: (Bf, Bm) force and moment allocation matrices
        """
        return self.Bf.copy(), self.Bm.copy()
    
    def get_control_allocation(self):
        """
        Get combined allocation matrix and its pseudo-inverse.
        
        Returns:
            tuple: (B_combined, B_pinv) for control allocation
        """
        return self.B_combined.copy(), self.B_pinv.copy()
    
    def is_over_actuated(self):
        """
        Check if drone is over-actuated (more motors than DOF).
        
        Returns:
            bool: True if over-actuated (num_motors > 6)
        """
        return self.num_motors > 6
    
    def get_motor_configuration_info(self):
        """
        Get comprehensive motor configuration information.
        
        Returns:
            dict: Configuration analysis including ranks, condition numbers, etc.
        """
        rank_f = np.linalg.matrix_rank(self.Bf)
        rank_m = np.linalg.matrix_rank(self.Bm)
        rank_combined = np.linalg.matrix_rank(self.B_combined)
        
        # Condition numbers for numerical analysis
        cond_f = np.linalg.cond(self.Bf @ self.Bf.T)
        cond_m = np.linalg.cond(self.Bm @ self.Bm.T)
        cond_combined = np.linalg.cond(self.B_combined @ self.B_combined.T)
        
        return {
            'num_motors': self.num_motors,
            'is_over_actuated': self.is_over_actuated(),
            'force_rank': rank_f,
            'moment_rank': rank_m,
            'combined_rank': rank_combined,
            'force_condition_number': cond_f,
            'moment_condition_number': cond_m,
            'combined_condition_number': cond_combined,
            'mass': self.mass,
            'center_of_gravity': self.cg.tolist()
        }
    
    @staticmethod
    def get_reference_properties():
        """
        Get reference drone properties for auto-scaling baseline.
        
        This creates the "matched" configuration that corresponds to the default
        setup in run_3D_simulation_configurable.py and extracts its actual
        physical properties for use as scaling reference.
        
        Returns:
            dict: Reference properties including mass, inertia, thrust capabilities, etc.
        """
        # Create the matched reference configuration
        # This matches the default in run_3D_simulation_configurable.py:
        # - 0.16m arm length (propellers at [±0.16, ±0.16, 0])
        # - "matched" propeller specs
        reference_propellers = [
            {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
            {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
            {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
            {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
        ]
        
        # Create reference configuration
        ref_config = DroneConfiguration(reference_propellers)
        
        # Calculate characteristic length (average distance from center)
        arm_lengths = []
        for prop in reference_propellers:
            arm_length = norm(np.array(prop["loc"]))
            if arm_length > 0:
                arm_lengths.append(arm_length)
        char_length = np.mean(arm_lengths)
        
        # Calculate thrust-to-weight ratio
        total_max_thrust = 0
        for prop in ref_config.propellers:
            k_f, k_m = prop["constants"]
            w_max = prop["wmax"]
            max_thrust = k_f * w_max**2
            total_max_thrust += max_thrust
        thrust_to_weight = total_max_thrust / (ref_config.mass * GRAVITY)
        
        # Calculate maximum control authority for each axis
        max_roll_torque = 0
        max_pitch_torque = 0
        max_yaw_torque = 0
        
        for i, prop in enumerate(ref_config.propellers):
            k_f, k_m = prop["constants"]
            w_max = prop["wmax"]
            max_thrust = k_f * w_max**2
            max_torque = k_m * w_max**2
            
            # Extract individual moment contributions at max thrust
            max_roll_torque = max(max_roll_torque, abs(ref_config.Bm[0, i]))
            max_pitch_torque = max(max_pitch_torque, abs(ref_config.Bm[1, i]))
            max_yaw_torque = max(max_yaw_torque, abs(ref_config.Bm[2, i]))
        
        return {
            'char_length': char_length,
            'mass': ref_config.mass,
            'inertia_trace': np.trace(ref_config.inertia_matrix),
            'Ix': ref_config.Ix,
            'Iy': ref_config.Iy,
            'Iz': ref_config.Iz,
            'num_motors': ref_config.num_motors,
            'thrust_to_weight': thrust_to_weight,
            'max_roll_torque': max_roll_torque,
            'max_pitch_torque': max_pitch_torque,
            'max_yaw_torque': max_yaw_torque,
            'total_max_thrust': total_max_thrust
        }