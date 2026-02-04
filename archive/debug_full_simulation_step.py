#!/usr/bin/env python3
"""
Debug full simulation loop to find where divergence occurs between systems.
"""

import numpy as np
from quadFiles.quad import Quadcopter as OriginalQuadcopter
from trajectory import Trajectory
from ctrl import Control as OriginalControl
from generalized_ctrl import GeneralizedControl
from drone_simulator import ConfigurableQuadcopter
from utils.windModel import Wind

def run_parallel_simulation():
    """Run three systems in parallel and compare at each step."""
    
    print("=" * 80)
    print("FULL SIMULATION STEP-BY-STEP COMPARISON")
    print("Original vs New Sim + Generalized Ctrl vs New Sim + Old Ctrl")
    print("=" * 80)
    
    # Setup
    Ti = 0
    Tf = 2.0  # Longer simulation
    Ts = 0.005
    
    # Original system - using same trajectory as run_3D_simulation.py
    quad_orig = OriginalQuadcopter(Ti)
    ctrlType = "xyz_pos"
    trajSelect = np.array([5, 3, 1])  # minimum jerk, follow yaw, average speed
    traj_orig = Trajectory(quad_orig, ctrlType, trajSelect)
    ctrl_orig = OriginalControl(quad_orig, traj_orig.yawType)
    
    # Generalized system with forced parameters  
    propellers = [
        {"loc": [0.16, 0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [-0.16, 0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"},
        {"loc": [-0.16, -0.16, 0], "dir": [0, 0, -1, "ccw"], "propsize": "matched"},
        {"loc": [0.16, -0.16, 0], "dir": [0, 0, -1, "cw"], "propsize": "matched"}
    ]
    quad_gen = ConfigurableQuadcopter(Ti, propellers=propellers)
    traj_gen = Trajectory(quad_gen, ctrlType, trajSelect)
    ctrl_gen = GeneralizedControl(quad_gen, traj_gen.yawType)
    
    # New simulator with old controller (third variant)
    quad_old_ctrl = ConfigurableQuadcopter(Ti, propellers=propellers)
    traj_old_ctrl = Trajectory(quad_old_ctrl, ctrlType, trajSelect)
    ctrl_old_ctrl = OriginalControl(quad_old_ctrl, traj_old_ctrl.yawType)
    
    # Wind
    wind = Wind('None', 2.0, 90, -15)
    
    # Initialize first desired states
    sDes_orig = traj_orig.desiredState(Ti, Ts, quad_orig)
    sDes_gen = traj_gen.desiredState(Ti, Ts, quad_gen)
    sDes_old_ctrl = traj_old_ctrl.desiredState(Ti, Ts, quad_old_ctrl)
    
    # Run initial control
    ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
    ctrl_gen.controller(traj_gen, quad_gen, sDes_gen, Ts)
    ctrl_old_ctrl.controller(traj_old_ctrl, quad_old_ctrl, sDes_old_ctrl, Ts)
    
    print(f"\nInitial Positions:")
    print(f"{'System':<15} {'X':<10} {'Y':<10} {'Z':<10}")
    print("-" * 50)
    print(f"{'Original':<15} {quad_orig.pos[0]:<10.6f} {quad_orig.pos[1]:<10.6f} {quad_orig.pos[2]:<10.6f}")
    print(f"{'Generalized':<15} {quad_gen.pos[0]:<10.6f} {quad_gen.pos[1]:<10.6f} {quad_gen.pos[2]:<10.6f}")
    print(f"{'Old Ctrl':<15} {quad_old_ctrl.pos[0]:<10.6f} {quad_old_ctrl.pos[1]:<10.6f} {quad_old_ctrl.pos[2]:<10.6f}")
    
    print(f"\nInitial Motor Commands:")
    print(f"{'System':<15} {'M1':<10} {'M2':<10} {'M3':<10} {'M4':<10}")
    print("-" * 65)
    print(f"{'Original':<15} {ctrl_orig.w_cmd[0]:<10.2f} {ctrl_orig.w_cmd[1]:<10.2f} {ctrl_orig.w_cmd[2]:<10.2f} {ctrl_orig.w_cmd[3]:<10.2f}")
    print(f"{'Generalized':<15} {ctrl_gen.w_cmd[0]:<10.2f} {ctrl_gen.w_cmd[1]:<10.2f} {ctrl_gen.w_cmd[2]:<10.2f} {ctrl_gen.w_cmd[3]:<10.2f}")
    print(f"{'Old Ctrl':<15} {ctrl_old_ctrl.w_cmd[0]:<10.2f} {ctrl_old_ctrl.w_cmd[1]:<10.2f} {ctrl_old_ctrl.w_cmd[2]:<10.2f} {ctrl_old_ctrl.w_cmd[3]:<10.2f}")
    
    # Initialize error tracking arrays
    max_steps = int((Tf - Ti) / Ts)
    pos_errors_gen = np.zeros(max_steps)
    pos_errors_old_ctrl = np.zeros(max_steps)
    ctrl_diff_gen = np.zeros(max_steps)
    ctrl_diff_old_ctrl = np.zeros(max_steps)
    dynamic_diff_gen = np.zeros(max_steps)
    dynamic_diff_old_ctrl = np.zeros(max_steps)
    
    # Print headers for step-by-step output
    print(f"\n\n{'='*120}")
    print(f"STEP-BY-STEP COMPARISON")
    print(f"{'='*140}")
    print(f"{'Step':<6} {'Time':<8} {'Pos Err Orig':<15} {'Pos Err Gen':<15} {'Pos Err OldCtrl':<15} {'Ctrl Diff Gen':<15} {'Ctrl Diff OldCtrl':<16} {'Dyn Diff Gen':<15} {'Dyn Diff OldCtrl':<15}")
    print("-" * 140)
    
    # Simulation loop
    t = Ti + Ts
    step = 1
    
    while step < max_steps and step < 4000:
        
        # Remove individual step headers - using tabular format instead
        
        # Store states before dynamics
        pos_orig_before = quad_orig.pos.copy()
        pos_gen_before = quad_gen.pos.copy()
        pos_old_ctrl_before = quad_old_ctrl.pos.copy()
        
        # Run dynamics update with motor commands (using SAME order as working version)
        
        # Original dynamics (this updates quad_orig state)
        quad_orig.update(t, Ts, ctrl_orig.w_cmd, wind)
        
        # Generalized dynamics (this updates quad_gen state) 
        quad_gen.update(t, Ts, ctrl_gen.w_cmd, wind)
        
        # New simulator with old controller dynamics
        quad_old_ctrl.update(t, Ts, ctrl_old_ctrl.w_cmd, wind)
        
        # Update time
        t += Ts
        
        # Get new desired states (BEFORE comparison - this is key!)
        sDes_orig = traj_orig.desiredState(t, Ts, quad_orig)
        sDes_gen = traj_gen.desiredState(t, Ts, quad_gen)
        sDes_old_ctrl = traj_old_ctrl.desiredState(t, Ts, quad_old_ctrl)
        
        # Run control for next step (BEFORE comparison - this is key!)
        ctrl_orig.controller(traj_orig, quad_orig, sDes_orig, Ts)
        ctrl_gen.controller(traj_gen, quad_gen, sDes_gen, Ts)
        ctrl_old_ctrl.controller(traj_old_ctrl, quad_old_ctrl, sDes_old_ctrl, Ts)
        
        # NOW compare states after dynamics (but with control updated for next step)
        pos_orig_after = quad_orig.pos.copy()
        pos_gen_after = quad_gen.pos.copy()
        pos_old_ctrl_after = quad_old_ctrl.pos.copy()
        
        # Calculate position errors from desired trajectory
        pos_err_orig = np.linalg.norm(pos_orig_after - sDes_orig[:3])
        pos_err_gen = np.linalg.norm(pos_gen_after - sDes_gen[:3])
        pos_err_old_ctrl = np.linalg.norm(pos_old_ctrl_after - sDes_old_ctrl[:3])
        
        # Calculate summed control differences (sum of absolute differences in motor commands)
        ctrl_sum_diff_gen = np.sum(np.abs(ctrl_orig.w_cmd - ctrl_gen.w_cmd))
        ctrl_sum_diff_old_ctrl = np.sum(np.abs(ctrl_orig.w_cmd - ctrl_old_ctrl.w_cmd))
        
        # Calculate dynamic differences (position change difference from original)
        pos_change_orig = np.linalg.norm(pos_orig_after - pos_orig_before)
        pos_change_gen = np.linalg.norm(pos_gen_after - pos_gen_before)
        pos_change_old_ctrl = np.linalg.norm(pos_old_ctrl_after - pos_old_ctrl_before)
        dyn_diff_gen = abs(pos_change_orig - pos_change_gen)
        dyn_diff_old_ctrl = abs(pos_change_orig - pos_change_old_ctrl)
        
        # Store errors (keeping gen and old_ctrl for statistics)
        pos_errors_gen[step-1] = pos_err_gen
        pos_errors_old_ctrl[step-1] = pos_err_old_ctrl
        ctrl_diff_gen[step-1] = ctrl_sum_diff_gen
        ctrl_diff_old_ctrl[step-1] = ctrl_sum_diff_old_ctrl
        dynamic_diff_gen[step-1] = dyn_diff_gen
        dynamic_diff_old_ctrl[step-1] = dyn_diff_old_ctrl
        
        # Print errors in tabular format for every step
        print(f"{step:<6} {t:<8.3f} {pos_err_orig:<15.6f} {pos_err_gen:<15.6f} {pos_err_old_ctrl:<15.6f} {ctrl_sum_diff_gen:<15.6f} {ctrl_sum_diff_old_ctrl:<16.6f} {dyn_diff_gen:<15.6f} {dyn_diff_old_ctrl:<15.6f}")
        
        # Check control outputs
        thrust_diff_gen = abs(np.linalg.norm(ctrl_orig.thrust_sp) - np.linalg.norm(ctrl_gen.thrust_sp))
        thrust_diff_old_ctrl = abs(np.linalg.norm(ctrl_orig.thrust_sp) - np.linalg.norm(ctrl_old_ctrl.thrust_sp))
        rate_diff_gen = np.linalg.norm(ctrl_orig.rateCtrl - ctrl_gen.rateCtrl)
        rate_diff_old_ctrl = np.linalg.norm(ctrl_orig.rateCtrl - ctrl_old_ctrl.rateCtrl)
        
        # if thrust_diff_gen > 0.001 or rate_diff_gen > 0.001:
        #     print(f"Control difference (Orig vs Gen) - Thrust: {thrust_diff_gen:.6f}N, Rate: {rate_diff_gen:.6f}rad/s")
        # if thrust_diff_old_ctrl > 0.001 or rate_diff_old_ctrl > 0.001:
        #     print(f"Control difference (Orig vs Old Ctrl) - Thrust: {thrust_diff_old_ctrl:.6f}N, Rate: {rate_diff_old_ctrl:.6f}rad/s")
        
        step += 1
    
    print("-" * 140)
    print(f"{'Step':<6} {'Time':<8} {'Pos Err Orig':<15} {'Pos Err Gen':<15} {'Pos Err OldCtrl':<15} {'Ctrl Diff Gen':<15} {'Ctrl Diff OldCtrl':<16} {'Dyn Diff Gen':<15} {'Dyn Diff OldCtrl':<15}")
    print(f"{'='*140}")
    
    print(f"\n\n{'='*120}")
    print(f"FINAL RESULTS AFTER {step} STEPS")
    print(f"{'='*120}")
    
    # Final positions vs desired
    final_desired_pos = sDes_orig[:3]  # Final desired position
    print(f"\nFinal Positions vs Desired:")
    print(f"{'System':<15} {'X':<12} {'Y':<12} {'Z':<12} {'Tracking Error':<15}")
    print("-" * 75)
    print(f"{'Desired':<15} {final_desired_pos[0]:<12.6f} {final_desired_pos[1]:<12.6f} {final_desired_pos[2]:<12.6f} {'--':<15}")
    print(f"{'Original':<15} {quad_orig.pos[0]:<12.6f} {quad_orig.pos[1]:<12.6f} {quad_orig.pos[2]:<12.6f} {np.linalg.norm(quad_orig.pos - final_desired_pos):<15.6f}")
    print(f"{'Generalized':<15} {quad_gen.pos[0]:<12.6f} {quad_gen.pos[1]:<12.6f} {quad_gen.pos[2]:<12.6f} {np.linalg.norm(quad_gen.pos - final_desired_pos):<15.6f}")
    print(f"{'Old Controller':<15} {quad_old_ctrl.pos[0]:<12.6f} {quad_old_ctrl.pos[1]:<12.6f} {quad_old_ctrl.pos[2]:<12.6f} {np.linalg.norm(quad_old_ctrl.pos - final_desired_pos):<15.6f}")
    
    # Error statistics table
    print(f"\nError Statistics:")
    print(f"{'Metric':<25} {'Comparison':<20} {'Mean':<12} {'Max':<12} {'RMS':<12}")
    print("-" * 85)
    
    # Position errors
    print(f"{'Position Error (m)':<25} {'Orig vs Gen':<20} {np.mean(pos_errors_gen[:step]):<12.6f} {np.max(pos_errors_gen[:step]):<12.6f} {np.sqrt(np.mean(pos_errors_gen[:step]**2)):<12.6f}")
    print(f"{'Position Error (m)':<25} {'Orig vs Old Ctrl':<20} {np.mean(pos_errors_old_ctrl[:step]):<12.6f} {np.max(pos_errors_old_ctrl[:step]):<12.6f} {np.sqrt(np.mean(pos_errors_old_ctrl[:step]**2)):<12.6f}")
    
    # Control differences  
    print(f"{'Control Diff (sum)':<25} {'Orig vs Gen':<20} {np.mean(ctrl_diff_gen[:step]):<12.6f} {np.max(ctrl_diff_gen[:step]):<12.6f} {np.sqrt(np.mean(ctrl_diff_gen[:step]**2)):<12.6f}")
    print(f"{'Control Diff (sum)':<25} {'Orig vs Old Ctrl':<20} {np.mean(ctrl_diff_old_ctrl[:step]):<12.6f} {np.max(ctrl_diff_old_ctrl[:step]):<12.6f} {np.sqrt(np.mean(ctrl_diff_old_ctrl[:step]**2)):<12.6f}")
    
    # Dynamic differences
    print(f"{'Dynamic Diff (m)':<25} {'Orig vs Gen':<20} {np.mean(dynamic_diff_gen[:step]):<12.6f} {np.max(dynamic_diff_gen[:step]):<12.6f} {np.sqrt(np.mean(dynamic_diff_gen[:step]**2)):<12.6f}")
    print(f"{'Dynamic Diff (m)':<25} {'Orig vs Old Ctrl':<20} {np.mean(dynamic_diff_old_ctrl[:step]):<12.6f} {np.max(dynamic_diff_old_ctrl[:step]):<12.6f} {np.sqrt(np.mean(dynamic_diff_old_ctrl[:step]**2)):<12.6f}")

if __name__ == "__main__":
    run_parallel_simulation()