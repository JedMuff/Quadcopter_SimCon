"""
Drone Simulator with Propeller-Based Configuration

This module provides the core drone simulation framework using propeller configurations
for automatic computation of physical properties and allocation matrices.
"""

import numpy as np
from sympy import *
from .drone_configuration import DroneConfiguration
from .propeller_data import create_standard_propeller_config

class DroneSimulator:
    """
    Core drone simulation framework.
    Uses propeller configurations for automatic computation of mass, inertia, and allocation matrices.
    """
    
    def __init__(self, propellers=None, dt=0.005, gravity=9.81):
        """
        Initialize drone simulator from propeller configuration.
        
        Args:
            propellers (list): List of propeller dictionaries, each containing:
                - "loc": [x, y, z] position in body frame (meters)
                - "dir": [x, y, z, rotation] thrust direction and spin direction  
                - "propsize": propeller size in inches (4-8)
            dt (float): Integration time step
            gravity (float): Gravitational acceleration
            
        Example:
            # Standard quadrotor
            propellers = [
                {"loc": [0.11, 0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [-0.11, 0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
                {"loc": [-0.11, -0.11, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
                {"loc": [0.11, -0.11, 0], "dir": [0, 0, -1, "cw"], "propsize": 5}
            ]
            drone = DroneSimulator(propellers=propellers)
        """
        
        # Use default quadrotor if no propellers specified
        if propellers is None:
            propellers = create_standard_propeller_config("quad", arm_length=0.11, prop_size=5)
        
        # Create drone configuration and compute physical properties
        self.config = DroneConfiguration(propellers)
        
        # Extract computed properties
        self.Bf, self.Bm = self.config.get_allocation_matrices()
        self.num_motors = self.config.num_motors
        self.mass = self.config.mass
        self.inertia = self.config.inertia_matrix
        self.center_of_gravity = self.config.cg
        
        # Simulation parameters
        self.dt = dt
        self.g = gravity
        
        # Initialize symbolic dynamics
        self._setup_dynamics()
        
        # State: [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
        self.state = np.zeros(12, dtype=np.float64)
        self.motor_commands = np.zeros(self.num_motors, dtype=np.float64)
        
        # History for plotting/analysis
        self.time_history = []
        self.state_history = []
        self.control_history = []
    
    @classmethod
    def create_standard_drone(cls, drone_type="quad", arm_length=0.11, prop_size=5, **kwargs):
        """
        Create standard drone configuration.
        
        Args:
            drone_type (str): Type of drone ('quad', 'hex', 'tri', 'octo')
            arm_length (float): Length of drone arms in meters
            prop_size (int): Propeller size in inches (4-8)
            **kwargs: Additional arguments for DroneSimulator
            
        Returns:
            DroneSimulator: Configured drone simulator
        """
        propellers = create_standard_propeller_config(drone_type, arm_length, prop_size)
        return cls(propellers=propellers, **kwargs)
    
    def _setup_dynamics(self):
        """Setup symbolic equations of motion using SymPy."""
        
        # State variables
        state_vars = symbols('x y z v_x v_y v_z phi theta psi p q r')
        x, y, z, vx, vy, vz, phi, theta, psi, p, q, r = state_vars
        
        # Control inputs (motor commands in range [0,1])
        control_symbols = [symbols(f'U_{i}') for i in range(1, self.num_motors + 1)]

        # Motor commands (already in [0,1] range)
        U = Matrix(control_symbols)
        
        # Rotation matrices
        Rx = Matrix([[1, 0, 0], 
                    [0, cos(phi), -sin(phi)], 
                    [0, sin(phi), cos(phi)]])
        Ry = Matrix([[cos(theta), 0, sin(theta)], 
                    [0, 1, 0], 
                    [-sin(theta), 0, cos(theta)]])
        Rz = Matrix([[cos(psi), -sin(psi), 0], 
                    [sin(psi), cos(psi), 0], 
                    [0, 0, 1]])
        R = Rz * Ry * Rx
        
        # Convert allocation matrices to SymPy
        Bf_sym = Matrix(self.Bf)
        Bm_sym = Matrix(self.Bm)
        
        # Forces and moments in body frame
        # Apply quadratic relationship for proper motor physics
        U_squared = Matrix([u**2 for u in U])
        F_body = Bf_sym @ U_squared  # [Fx, Fy, Fz] in body frame
        M_body = Bm_sym @ U_squared  # [Mx, My, Mz] in body frame
        
        # Translational dynamics (Newton's laws)
        d_x = vx
        d_y = vy  
        d_z = vz
        
        # Transform body forces to world frame and add gravity
        F_world = R @ F_body + Matrix([0, 0, self.g * self.mass])
        d_vx = F_world[0] / self.mass
        d_vy = F_world[1] / self.mass
        d_vz = F_world[2] / self.mass
        
        # Rotational dynamics (Euler's equations)
        d_phi = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta)
        d_theta = q * cos(phi) - r * sin(phi)
        d_psi = q * sin(phi) / cos(theta) + r * cos(phi) / cos(theta)
        
        # Rotational dynamics using inertia matrix
        # M = I * omega_dot + omega x (I * omega)
        # Simplified version: omega_dot = I^(-1) * M_body
        I_inv = Matrix(np.linalg.inv(self.inertia))
        omega = Matrix([p, q, r])
        I_omega = Matrix(self.inertia) @ omega
        
        # Gyroscopic term: omega x (I * omega)
        gyroscopic = Matrix([
            q * I_omega[2] - r * I_omega[1],
            r * I_omega[0] - p * I_omega[2], 
            p * I_omega[1] - q * I_omega[0]
        ])
        
        omega_dot = I_inv @ (M_body - gyroscopic)
        d_p = omega_dot[0]
        d_q = omega_dot[1]
        d_r = omega_dot[2]
        
        # Complete state derivative
        state_dot = [d_x, d_y, d_z, d_vx, d_vy, d_vz, d_phi, d_theta, d_psi, d_p, d_q, d_r]
        
        # Create numerical function
        self.dynamics_func = lambdify((Array(state_vars), Array(control_symbols)), 
                                    Array(state_dot), 'numpy')

    def get_configuration_info(self):
        """
        Get comprehensive drone configuration information.
        
        Returns:
            dict: Complete configuration including physical properties and capabilities
        """
        info = self.config.get_physical_properties()
        info.update(self.config.get_motor_configuration_info())
        info['propeller_configuration'] = self.config.propellers
        return info
    
    def get_propeller_info(self):
        """
        Get propeller configuration details.
        
        Returns:
            list: List of propeller specifications
        """
        return [prop.copy() for prop in self.config.propellers]
    
    def set_state(self, position=None, velocity=None, attitude=None, angular_velocity=None):
        """
        Set drone state.
        
        Args:
            position: [x, y, z] in world frame
            velocity: [vx, vy, vz] in world frame  
            attitude: [phi, theta, psi] (roll, pitch, yaw) in radians
            angular_velocity: [p, q, r] in body frame
        """
        if position is not None:
            self.state[0:3] = position
        if velocity is not None:
            self.state[3:6] = velocity
        if attitude is not None:
            self.state[6:9] = attitude
        if angular_velocity is not None:
            self.state[9:12] = angular_velocity
    
    def get_state(self):
        """Get current state as dictionary."""
        return {
            'position': self.state[0:3].copy(),
            'velocity': self.state[3:6].copy(), 
            'attitude': self.state[6:9].copy(),
            'angular_velocity': self.state[9:12].copy(),
            'time': len(self.time_history) * self.dt
        }
    
    def _get_actual_motor_speeds(self):
        """Get actual motor speeds from normalized commands, properly scaled and padded."""
        motor_speeds = np.zeros(max(4, self.num_motors))
        
        for i in range(self.num_motors):
            prop = self.config.propellers[i]
            w_max = prop["wmax"]
            motor_speeds[i] = np.sqrt(self.motor_commands[i]) * w_max
        
        # Return first 4 values for compatibility (pad with zeros if needed)
        return motor_speeds[:4]
    
    def set_motor_commands(self, commands):
        """
        Set motor commands.
        
        Args:
            commands: Array of motor commands in range [0, 1]
        """
        self.motor_commands = np.clip(commands, 0, 1)
    
    def step(self, motor_commands=None):
        """
        Advance simulation by one time step using RK4 integration.
        
        Args:
            motor_commands: Optional motor commands for this step
        """
        if motor_commands is not None:
            self.set_motor_commands(motor_commands)
        
        # RK4 integration for better numerical stability
        k1 = self.dt * self.dynamics_func(self.state, self.motor_commands)
        k2 = self.dt * self.dynamics_func(self.state + 0.5 * k1, self.motor_commands)
        k3 = self.dt * self.dynamics_func(self.state + 0.5 * k2, self.motor_commands)
        k4 = self.dt * self.dynamics_func(self.state + k3, self.motor_commands)
        
        # Update state using RK4 formula
        self.state = self.state + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        
        # Store history
        self.time_history.append(len(self.time_history) * self.dt)
        self.state_history.append(self.state.copy())
        self.control_history.append(self.motor_commands.copy())
        
        return self.get_state()
    
    def get_params(self):
        """
        Get parameters in format compatible with existing controller framework.
        
        Returns:
            dict: Parameters compatible with existing Quadcopter class
        """
        # Create mixer matrix from allocation matrices
        # Convert allocation matrices to mixer format [F, Mx, My, Mz] x [motor1, motor2, ...]
        # The original mixer represents kTh*w^2 units, but allocation matrices are in Newtons
        # Need to convert: divide by max_force_per_motor to get normalized coefficients
        
        # Get the maximum force each motor can produce at w_max
        propeller_forces = []
        for prop in self.config.propellers:
            k_f, k_m = prop["constants"]
            w_max = prop["wmax"]
            max_force = k_f * w_max**2
            propeller_forces.append(max_force)
        
        # ==================================================================================
        # MIXER MATRIX CONSTRUCTION - CRITICAL FOR CONTROL ALLOCATION
        # ==================================================================================
        # 
        # OVERVIEW:
        # The mixer matrices convert between control commands and motor speeds.
        # This is the core of the control allocation system.
        #
        # PIPELINE:
        # Controller → [F_thrust, M_roll, M_pitch, M_yaw] → mixerFMinv → [w²_normalized] → [w_actual]
        #
        # COORDINATE SYSTEM:
        # - Thrust: Positive values represent upward thrust (against gravity)
        # - Moments: Standard body-frame convention (roll=X, pitch=Y, yaw=Z)
        # - Motor speeds: Always positive, in rad/s
        #
        # SCALING CONVENTION:
        # - The allocation matrices (Bf, Bm) are pre-scaled for normalized motor commands [0,1]
        # - When w²_normalized = 1.0 for all motors, they produce maximum thrust/moment capability
        # - When w²_normalized = 0.0, motors are off (minimum thrust)
        #
        # ALLOCATION MATRICES EXPLAINED:
        # - Bf[i,j] = force produced by motor j in direction i when w²_normalized = 1.0
        # - Bm[i,j] = moment produced by motor j about axis i when w²_normalized = 1.0
        # - These include the full physical scaling: k_f * w_max², lever arms, directions
        #
        # CONTROL ALLOCATION MATRIX:
        # We extract the control variables we care about: [thrust, roll_moment, pitch_moment, yaw_moment]
        # CRITICAL FIX: Roll and pitch directions were inverted - negate both to correct
        # Based on debug analysis showing continued instability after roll-only fix
        Bm_corrected = self.Bm.copy()
        # Bm_corrected[0, :] = -self.Bm[0, :]  # Fix roll direction inversion
        # Bm_corrected[1, :] = -self.Bm[1, :]  # Fix pitch direction inversion (additional fix)
        
        A_control = np.vstack([
            -self.Bf[2:3, :],  # Thrust row: negative because Bf[2,:] is downward force, we want upward thrust
            Bm_corrected       # Moment rows: [roll, pitch, yaw] moments about body axes (roll corrected)
        ])
        # A_control shape: (4, num_motors)
        # A_control @ [w²_norm_0, w²_norm_1, ...] = [F_thrust, M_roll, M_pitch, M_yaw]
        
        # MIXER MATRIX STORAGE:
        # The legacy parameter names are:
        # - "mixerFM": Forward mapping [w²_normalized] → [forces, moments] 
        # - "mixerFMinv": Inverse mapping [forces, moments] → [w²_normalized]
        #
        # IMPORTANT: Controller uses mixerFMinv to convert commands to motor speeds
        mixer_fm = A_control  # Forward mapping: [w²_normalized] → [F_thrust, M_roll, M_pitch, M_yaw]
        
        # Calculate motor parameters from propeller specs
        total_prop_mass = sum(prop["mass"] for prop in self.config.propellers)
        # Handle numeric and string propeller sizes
        prop_sizes = []
        for prop in self.config.propellers:
            if prop["propsize"] == "matched":
                prop_sizes.append(8)  # Use 8 as equivalent for averaging
            else:
                prop_sizes.append(prop["propsize"])
        avg_prop_size = np.mean(prop_sizes)
        
        # Estimate thrust and torque coefficients from first propeller
        first_prop = self.config.propellers[0]
        k_f, k_m = first_prop["constants"]
        w_max = first_prop["wmax"]
        
        # Hover calculations
        hover_thrust_per_motor = (self.mass * self.g) / self.num_motors
        w_hover = np.sqrt(hover_thrust_per_motor / k_f)
        
        params = {
            # Physical properties
            "mB": self.mass,
            "g": self.g,
            "IB": self.inertia,
            "invI": np.linalg.inv(self.inertia),
            
            # Geometric properties (use average arm length)
            "dxm": np.mean([abs(prop["loc"][0]) for prop in self.config.propellers if prop["loc"][0] != 0]),
            "dym": np.mean([abs(prop["loc"][1]) for prop in self.config.propellers if prop["loc"][1] != 0]),
            "dzm": 0.05,  # Default motor height
            
            # Motor properties
            "kTh": k_f,
            "kTo": k_m, 
            "w_hover": w_hover,
            "thr_hover": hover_thrust_per_motor,
            
            # Mixer matrices
            "mixerFM": mixer_fm,
            "mixerFMinv": np.linalg.pinv(mixer_fm),
            
            # Thrust limits
            "minThr": 0.1 * self.num_motors,
            "maxThr": k_f * w_max**2 * self.num_motors,
            "minWmotor": 75,
            "maxWmotor": w_max,
            
            # Motor dynamics (defaults from original framework)
            "tau": 0.015,
            "kp": 1.0,
            "damp": 1.0,
            "motorc1": 8.49,
            "motorc0": 74.7,
            "motordeadband": 1,
            
            # Other parameters
            "Cd": 0.1,
            "IRzz": 2.7e-5,
            "useIntergral": False,
            
            # Feed-forward command for hover
            "FF": (w_hover - 74.7) / 8.49  # Using motorc0 and motorc1 defaults
        }
        
        return params
    
    def get_quadcopter_state(self):
        """
        Get state in format compatible with existing Quadcopter class.
        
        Returns:
            dict: State variables in Quadcopter format
        """
        # Convert from Euler angles (if available) to quaternion, or use identity
        phi, theta, psi = self.state[6:9]  # attitude from our state
        
        # Convert Euler angles to quaternion [w, x, y, z]
        # Using ZYX rotation order (yaw-pitch-roll)
        cy = np.cos(psi * 0.5)
        sy = np.sin(psi * 0.5)
        cp = np.cos(theta * 0.5)
        sp = np.sin(theta * 0.5)
        cr = np.cos(phi * 0.5)
        sr = np.sin(phi * 0.5)

        quat = np.array([
            cr * cp * cy + sr * sp * sy,  # w
            sr * cp * cy - cr * sp * sy,  # x
            cr * sp * cy + sr * cp * sy,  # y
            cr * cp * sy - sr * sp * cy   # z
        ])
        
        # Convert from our 12-state format to extended state format
        extended_state = np.zeros(21)
        extended_state[0:3] = self.state[0:3]    # position
        extended_state[3:7] = quat               # quaternion  
        extended_state[7:10] = self.state[3:6]   # velocity
        extended_state[10:13] = self.state[9:12] # angular velocity
        
        # Motor states (proper motor speed conversion using actual w_max values)
        for i in range(min(4, self.num_motors)):
            prop = self.config.propellers[i]
            w_max = prop["wmax"]
            # Convert normalized motor command [0,1] to actual motor speed
            extended_state[13 + i*2] = np.sqrt(self.motor_commands[i]) * w_max
            extended_state[14 + i*2] = 0  # Motor acceleration (not tracked in current implementation)
            
        return {
            'state': extended_state,
            'pos': self.state[0:3],
            'vel': self.state[3:6], 
            'quat': quat,
            'omega': self.state[9:12],
            'euler': np.array([0, 0, 0]),  # Will be computed from quaternion
            'wMotor': self._get_actual_motor_speeds(),
            'vel_dot': np.zeros(3),  # Velocity derivative
            'omega_dot': np.zeros(3),  # Angular velocity derivative
            'acc': np.zeros(3),  # Acceleration
            'thr': self.motor_commands[:4] if self.num_motors >= 4 else np.pad(self.motor_commands, (0, 4-self.num_motors)),
            'tor': self.motor_commands[:4] if self.num_motors >= 4 else np.pad(self.motor_commands, (0, 4-self.num_motors)),
            'dcm': np.eye(3)  # Direction cosine matrix from quaternion
        }
    
    def update_from_controller(self, t, Ts, w_cmd, wind=None):
        """
        Update simulation using commands from existing controller framework.

        Args:
            t: Current time
            Ts: Time step
            w_cmd: Motor commands from controller (in rad/s)
            wind: Wind model (optional)
        """
        # Get the full motor command if available (for drones with >4 motors)
        if hasattr(self, 'w_cmd_full') and self.w_cmd_full is not None:
            full_w_cmd = self.w_cmd_full
        else:
            full_w_cmd = w_cmd

        # Convert motor speed commands to normalized [0,1] range
        # CRITICAL FIX: motor_commands should be w²_normalized, not w_normalized!
        # The allocation matrices expect w²_normalized where 1.0 means full throttle
        propeller_w_max = [prop["wmax"] for prop in self.config.propellers]
        motor_commands = np.zeros(self.num_motors)

        for i in range(min(len(full_w_cmd), self.num_motors)):
            w_max = propeller_w_max[i] if i < len(propeller_w_max) else propeller_w_max[0]
            # Convert: w_cmd (rad/s) → w_normalized
            # NOTE: Dynamics will square this internally, so don't square here!
            motor_commands[i] = np.clip(full_w_cmd[i] / w_max, 0, 1)

        # DEBUG: Print motor commands
        if not hasattr(self, '_motor_cmd_debug_counter'):
            self._motor_cmd_debug_counter = 0
        self._motor_cmd_debug_counter += 1
        if self._motor_cmd_debug_counter % 100 == 1:
            print(f"\nMOTOR COMMAND DEBUG (in simulator):")
            print(f"  w_cmd (input): {full_w_cmd[:4]}")
            print(f"  motor_commands (w²_norm): {motor_commands[:4]}")
            print(f"  Bf[2,:]: {self.config.Bf[2,:]}")
            expected_force_z = np.dot(self.config.Bf[2,:], motor_commands)
            print(f"  Expected F_body[2]: {expected_force_z:.2f} N")

        # Step the simulation
        self.step(motor_commands)
        
        return t + Ts

    def simulate(self, time_span, control_function=None):
        """
        Run simulation for specified time span.
        
        Args:
            time_span: Total simulation time
            control_function: Function that takes (time, state) and returns motor commands
        """
        num_steps = int(time_span / self.dt)
        
        for i in range(num_steps):
            current_time = i * self.dt
            current_state = self.get_state()
            
            if control_function is not None:
                commands = control_function(current_time, current_state)
                self.step(commands)
            else:
                self.step()
    
    def reset(self):
        """Reset simulation to initial conditions."""
        self.state = np.zeros(12)
        self.motor_commands = np.zeros(self.num_motors)
        self.time_history = []
        self.state_history = []
        self.control_history = []


# Factory functions for easy drone creation
def create_quadrotor(arm_length=0.11, prop_size=5, **kwargs):
    """Create standard quadrotor configuration."""
    return DroneSimulator.create_standard_drone("quad", arm_length, prop_size, **kwargs)

def create_hexarotor(arm_length=0.10, prop_size=4, **kwargs):
    """Create standard hexarotor configuration.""" 
    return DroneSimulator.create_standard_drone("hex", arm_length, prop_size, **kwargs)

def create_tricopter(arm_length=0.12, prop_size=6, **kwargs):
    """Create standard tricopter configuration."""
    return DroneSimulator.create_standard_drone("tri", arm_length, prop_size, **kwargs)

def create_octorotor(arm_length=0.09, prop_size=4, **kwargs):
    """Create standard octorotor configuration."""
    return DroneSimulator.create_standard_drone("octo", arm_length, prop_size, **kwargs)


class ConfigurableQuadcopter:
    """
    Wrapper class that makes DroneSimulator compatible with existing controller framework.
    
    This class provides the same interface as the original Quadcopter class but uses
    the configurable DroneSimulator internally for flexible drone configurations.
    """
    
    def __init__(self, Ti, propellers=None, drone_type="quad", arm_length=0.11, prop_size=5):
        """
        Initialize configurable quadcopter.
        
        Args:
            Ti: Initial time
            propellers: Custom propeller configuration (optional)
            drone_type: Standard drone type if propellers not specified
            arm_length: Arm length for standard configurations
            prop_size: Propeller size for standard configurations
        """
        # Create drone simulator
        if propellers is not None:
            self.drone_sim = DroneSimulator(propellers=propellers, dt=0.005)
        else:
            self.drone_sim = DroneSimulator.create_standard_drone(
                drone_type, arm_length, prop_size, dt=0.005
            )
        
        # Get parameters in compatible format
        self.params = self.drone_sim.get_params()
        
        # Initialize state variables in compatible format
        self._update_state_variables()
        
        # Initialize extended state variables
        self.extended_state()
        
        # Store initial time
        self.Ti = Ti
    
    def _update_state_variables(self):
        """Update all state variables from drone simulator."""
        quad_state = self.drone_sim.get_quadcopter_state()
        
        # Copy all state variables to be compatible with existing code
        self.state = quad_state['state']
        self.pos = quad_state['pos'] 
        self.vel = quad_state['vel']
        self.quat = quad_state['quat']
        self.omega = quad_state['omega']
        self.euler = quad_state['euler']
        self.wMotor = quad_state['wMotor']
        self.vel_dot = quad_state['vel_dot']
        self.omega_dot = quad_state['omega_dot']
        self.acc = quad_state['acc']
        self.thr = quad_state['thr']
        self.tor = quad_state['tor']
        self.dcm = quad_state['dcm']
        
    def extended_state(self):
        """Update extended state variables (quaternion to euler conversion, etc.)."""
        import drone_sim.utils as utils
        
        # Update DCM from quaternion
        self.dcm = utils.quat2Dcm(self.quat)
        
        # Update Euler angles from quaternion  
        YPR = utils.quatToYPR_ZYX(self.quat)
        self.euler = YPR[::-1]  # flip YPR so that euler state = phi, theta, psi
        self.psi = YPR[0]
        self.theta = YPR[1] 
        self.phi = YPR[2]
    
    def forces(self):
        """Calculate rotor thrusts and torques."""
        # Get motor commands from drone simulator
        motor_speeds = np.sqrt(self.drone_sim.motor_commands) * self.params["maxWmotor"]
        
        # Calculate thrusts and torques
        self.thr = self.params["kTh"] * motor_speeds * motor_speeds
        self.tor = self.params["kTo"] * motor_speeds * motor_speeds
        
        # Pad to 4 motors for compatibility
        if len(self.thr) < 4:
            self.thr = np.pad(self.thr, (0, 4 - len(self.thr)))
            self.tor = np.pad(self.tor, (0, 4 - len(self.tor)))
    
    def update(self, t, Ts, cmd, wind):
        """
        Update simulation state.
        
        Args:
            t: Current time
            Ts: Time step
            cmd: Motor commands
            wind: Wind model
        """
        # Store full motor command for drones with >4 motors
        if hasattr(self, 'w_cmd_full'):
            self.drone_sim.w_cmd_full = self.w_cmd_full
            
        # Update drone simulator
        new_t = self.drone_sim.update_from_controller(t, Ts, cmd, wind)
        
        # Update state variables
        self._update_state_variables()
        self.extended_state()
        self.forces()
        
        return new_t
    
    def get_configuration_info(self):
        """Get comprehensive configuration information."""
        return self.drone_sim.get_configuration_info()
    
    def get_propeller_info(self):
        """Get propeller configuration details.""" 
        return self.drone_sim.get_propeller_info()