# -*- coding: utf-8 -*-
"""
Configurable 3D Simulation for Multi-Rotor Drones

This simulation allows for flexible drone configurations using propeller-based
automatic computation of physical properties and allocation matrices.

DEFAULT BEHAVIOR: Uses "matched" configuration that produces identical trajectory
tracking performance to the original run_3D_simulation.py

Based on original simulation by:
author: John Bass
email: john.bobzwik@gmail.com
license: MIT

Enhanced with configurable drone framework supporting:
- Matched configuration (default) - equivalent to original framework
- Quadrotors, Hexarotors, Tricopters, Octorotors  
- Custom propeller configurations
- Automatic mass, inertia, and allocation matrix computation
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import argparse
import json

from trajectory import Trajectory
from ctrl import Control
from drone_simulator import ConfigurableQuadcopter, create_standard_propeller_config
from utils.windModel import Wind
import utils
import config

def quad_sim(t, Ts, quad, ctrl, wind, traj):
    """
    Single simulation step for configurable drone.
    
    Args:
        t: Current time
        Ts: Time step
        quad: ConfigurableQuadcopter instance
        ctrl: Control instance
        wind: Wind model
        traj: Trajectory instance
        
    Returns:
        Updated time
    """
    # Dynamics (using last timestep's commands)
    quad.update(t, Ts, ctrl.w_cmd, wind)
    t += Ts

    # Trajectory for Desired States 
    sDes = traj.desiredState(t, Ts, quad)        

    # Generate Commands (for next iteration)
    ctrl.controller(traj, quad, sDes, Ts)

    return t

def create_drone_from_config(config_dict, Ti=0):
    """
    Create drone from configuration dictionary.
    
    Args:
        config_dict: Dictionary containing drone configuration
        Ti: Initial time
        
    Returns:
        ConfigurableQuadcopter instance
    """
    if "propellers" in config_dict:
        # Custom propeller configuration
        return ConfigurableQuadcopter(Ti, propellers=config_dict["propellers"])
    else:
        # Standard configuration
        drone_type = config_dict.get("type", "quad")
        arm_length = config_dict.get("arm_length", 0.11)
        prop_size = config_dict.get("prop_size", 5)
        
        # Handle "matched" type - create configuration equivalent to original framework
        if drone_type == "matched" or prop_size == "matched":
            # Create matched drone configuration
            propellers = [
                {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
                {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
                {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
                {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
            ]
            return ConfigurableQuadcopter(Ti, propellers=propellers)
        else:
            # Convert string prop_size to int if needed
            if isinstance(prop_size, str) and prop_size.isdigit():
                prop_size = int(prop_size)
            
            return ConfigurableQuadcopter(Ti, 
                                        drone_type=drone_type,
                                        arm_length=arm_length, 
                                        prop_size=prop_size)

def print_drone_info(quad):
    """Print comprehensive drone configuration information."""
    config_info = quad.get_configuration_info()
    prop_info = quad.get_propeller_info()
    
    print("\n" + "="*60)
    print("DRONE CONFIGURATION")
    print("="*60)
    
    print(f"Drone Type: {config_info.get('num_motors')}-rotor")
    print(f"Mass: {config_info['mass']:.3f} kg")
    print(f"Center of Gravity: [{config_info['center_of_gravity'][0]:.3f}, {config_info['center_of_gravity'][1]:.3f}, {config_info['center_of_gravity'][2]:.3f}] m")
    
    print(f"\nInertia Properties:")
    print(f"  Ix: {config_info['Ix']:.6f} kg⋅m²")
    print(f"  Iy: {config_info['Iy']:.6f} kg⋅m²") 
    print(f"  Iz: {config_info['Iz']:.6f} kg⋅m²")
    
    print(f"\nControl Properties:")
    print(f"  Over-actuated: {config_info['is_over_actuated']}")
    print(f"  Force rank: {config_info['force_rank']}")
    print(f"  Moment rank: {config_info['moment_rank']}")
    print(f"  Combined rank: {config_info['combined_rank']}")
    
    print(f"\nPropeller Configuration:")
    for i, prop in enumerate(prop_info):
        print(f"  Motor {i+1}: {prop['propsize']}\" at [{prop['loc'][0]:.3f}, {prop['loc'][1]:.3f}, {prop['loc'][2]:.3f}] m, {prop['dir'][-1]} rotation")
    
    print("="*60)

def main():
    """Main simulation function with configurable drone support."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Configurable 3D Drone Simulation')
    parser.add_argument('--config', type=str, help='JSON configuration file')
    parser.add_argument('--type', type=str, choices=['quad', 'hex', 'tri', 'octo', 'matched'], 
                       default='matched', help='Standard drone type (matched = original framework equivalent)')
    parser.add_argument('--arm-length', type=float, default=0.16, 
                       help='Arm length in meters')
    parser.add_argument('--prop-size', type=str, choices=['4', '5', '6', '7', '8', 'matched'], 
                       default='matched', help='Propeller size in inches or "matched" for original framework equivalent')
    parser.add_argument('--time', type=float, default=20, 
                       help='Simulation time in seconds')
    parser.add_argument('--dt', type=float, default=0.005, 
                       help='Time step in seconds')
    parser.add_argument('--save', action='store_true', 
                       help='Save animation')
    parser.add_argument('--info', action='store_true',
                       help='Print detailed drone configuration')
    
    args = parser.parse_args()
    
    start_time = time.time()

    # Simulation Setup
    Ti = 0
    Ts = args.dt
    Tf = args.time
    ifsave = 1 if args.save else 0

    # Choose trajectory settings
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = np.zeros(3)

    # Select Control Type (0: xyz_pos, 1: xy_vel_z_pos, 2: xyz_vel)
    ctrlType = ctrlOptions[0]   
    
    # Trajectory configuration
    trajSelect[0] = 5    # minimum jerk trajectory
    trajSelect[1] = 3    # follow yaw
    trajSelect[2] = 1    # use average speed
    
    print("Control type: {}".format(ctrlType))


    # Create standard drone from command line arguments
    config_dict = {
        "type": args.type,
        "arm_length": args.arm_length,
        "prop_size": args.prop_size
    }
    quad = create_drone_from_config(config_dict, Ti)
    if args.type == "matched" or args.prop_size == "matched":
        print(f"Created matched drone (equivalent to original framework)")
    else:
        print(f"Created {args.type} with {args.arm_length}m arms and {args.prop_size}\" props")

    # Print drone information if requested
    if args.info:
        print_drone_info(quad)

    # Initialize Controller, Wind, Trajectory
    traj = Trajectory(quad, ctrlType, trajSelect)
    ctrl = Control(quad, traj.yawType)
    wind = Wind('None', 2.0, 90, -15)

    # Trajectory for First Desired States
    sDes = traj.desiredState(0, Ts, quad)        

    # Generate First Commands
    ctrl.controller(traj, quad, sDes, Ts)
    
    # Initialize Result Matrices
    numTimeStep = int(Tf/Ts+1)
    
    # Get the actual number of motors for this drone
    actual_num_motors = quad.drone_sim.num_motors
    
    t_all          = np.zeros(numTimeStep)
    s_all          = np.zeros([numTimeStep, len(quad.state)])
    pos_all        = np.zeros([numTimeStep, len(quad.pos)])
    vel_all        = np.zeros([numTimeStep, len(quad.vel)])
    quat_all       = np.zeros([numTimeStep, len(quad.quat)])
    omega_all      = np.zeros([numTimeStep, len(quad.omega)])
    euler_all      = np.zeros([numTimeStep, len(quad.euler)])
    sDes_traj_all  = np.zeros([numTimeStep, len(traj.sDes)])
    sDes_calc_all  = np.zeros([numTimeStep, len(ctrl.sDesCalc)])
    w_cmd_all      = np.zeros([numTimeStep, len(ctrl.w_cmd)])
    wMotor_all     = np.zeros([numTimeStep, max(4, actual_num_motors)])  # At least 4 for compatibility
    thr_all        = np.zeros([numTimeStep, max(4, actual_num_motors)])  # At least 4 for compatibility
    tor_all        = np.zeros([numTimeStep, max(4, actual_num_motors)])  # At least 4 for compatibility

    # Store initial values
    t_all[0]            = Ti
    s_all[0,:]          = quad.state
    pos_all[0,:]        = quad.pos
    vel_all[0,:]        = quad.vel
    quat_all[0,:]       = quad.quat
    omega_all[0,:]      = quad.omega
    euler_all[0,:]      = quad.euler
    sDes_traj_all[0,:]  = traj.sDes
    sDes_calc_all[0,:]  = ctrl.sDesCalc
    w_cmd_all[0,:]      = ctrl.w_cmd
    wMotor_all[0,:len(quad.wMotor)] = quad.wMotor
    thr_all[0,:len(quad.thr)]       = quad.thr
    tor_all[0,:len(quad.tor)]       = quad.tor

    # Run Simulation
    print(f"\nRunning simulation for {Tf}s with {quad.drone_sim.num_motors} motors...")
    t = Ti
    i = 1
    while round(t,3) < Tf:
        
        t = quad_sim(t, Ts, quad, ctrl, wind, traj)
        
        # Store results
        t_all[i]             = t
        s_all[i,:]           = quad.state
        pos_all[i,:]         = quad.pos
        vel_all[i,:]         = quad.vel
        quat_all[i,:]        = quad.quat
        omega_all[i,:]       = quad.omega
        euler_all[i,:]       = quad.euler
        sDes_traj_all[i,:]   = traj.sDes
        sDes_calc_all[i,:]   = ctrl.sDesCalc
        w_cmd_all[i,:]       = ctrl.w_cmd
        wMotor_all[i,:len(quad.wMotor)] = quad.wMotor
        thr_all[i,:len(quad.thr)]       = quad.thr
        tor_all[i,:len(quad.tor)]       = quad.tor
        
        pos_error = np.linalg.norm(quad.pos - traj.sDes[:3])
        print(f"t={t:.3f}s, pos_error={pos_error:.3f}m, vel={quad.vel}, w_cmd={ctrl.w_cmd}, thr={quad.thr}, tor={quad.tor}")
        i += 1
    
    end_time = time.time()
    print("Simulated {:.2f}s in {:.6f}s.".format(t, end_time - start_time))

    # View Results
    utils.makeFigures(quad.params, t_all, pos_all, vel_all, quat_all, omega_all, 
                     euler_all, w_cmd_all, wMotor_all, thr_all, tor_all, 
                     sDes_traj_all, sDes_calc_all)
    
    ani = utils.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, 
                                 Ts, quad.params, traj.xyzType, traj.yawType, ifsave)
    plt.show()

if __name__ == "__main__":
    # Run main simulation
    if (config.orient == "NED" or config.orient == "ENU"):
        main()
        # cProfile.run('main()')
    else:
        raise Exception("{} is not a valid orientation. Verify config.py file.".format(config.orient))