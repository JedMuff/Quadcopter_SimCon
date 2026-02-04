# Quadcopter Simulation and Control Framework

A comprehensive Python framework for quadcopter and multi-rotor drone simulation with configurable control systems.

## Features

- **Configurable Drone Types**: Support for quadrotors, hexarotors, octorotors, and custom configurations
- **Advanced Control Systems**: Lee control, PX4-based control, and generalized allocation matrices
- **Physical Simulation**: Realistic dynamics with configurable mass, inertia, and aerodynamics
- **Trajectory Generation**: Multiple trajectory types including minimum jerk and waypoint following
- **Visualization**: 3D animation and comprehensive plotting capabilities

## Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/bobzwik/Quadcopter_SimCon.git
cd Quadcopter_SimCon

# Install in development mode
pip install -e .
```

### Dependencies

- Python >= 3.7
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0
- SymPy >= 1.8.0
- SciPy >= 1.7.0

## Quick Start

```python
from drone_sim.simulation.drone_simulator import DroneSimulator
from drone_sim.px4_ctrl.px4_based_ctrl import GeneralizedControl

# Create a standard quadrotor
drone = DroneSimulator.create_standard_drone("quad", arm_length=0.11, prop_size=5)

# Initialize controller
controller = GeneralizedControl(drone, yawType=1)

# Run simulation
drone.simulate(time_span=10.0)
```

## Examples

Run the main simulation example:

```bash
python examples/run_3D_simulation_configurable.py --type quad --time 20
```

For different drone configurations:

```bash
# Hexarotor simulation
python examples/run_3D_simulation_configurable.py --type hex --time 20

# Custom configuration from JSON
python examples/run_3D_simulation_configurable.py --config configs/custom_config.json
```

## Project Structure

```
drone_sim/
├── simulation/          # Core simulation framework
├── lee_control/         # Lee controller implementation
├── px4_ctrl/           # PX4-based control systems
├── trajectory_generation/ # Trajectory planning
└── utils/              # Utilities and visualization

examples/               # Example scripts and demos
archive/               # Legacy code and reference implementations
```

## Technical Deep Dive: Thrust Allocation Pipeline

This framework uses a sophisticated control allocation system to convert high-level control commands into individual motor speeds. Understanding this pipeline is crucial for control system development and debugging.

### Control Allocation Pipeline

The thrust allocation system follows this pipeline:

```
Controller Output → Control Allocation → Motor Commands → Physical Forces
     ↓                      ↓                  ↓              ↓
[F, Mx, My, Mz]     →   mixerFMinv      →  [w₁, w₂, ...]  →  Thrust & Moments
  (Newtons,          →   (4×N matrix)   →   (rad/s)       →   (Physics)
   Newton-meters)
```

### Pipeline Stages

1. **Controller Output** (`drone_sim/px4_ctrl/px4_based_ctrl.py:527`)
   - Thrust command: `thrust_command` (N) - upward thrust needed
   - Moment commands: `[roll_moment, pitch_moment, yaw_moment]` (N·m)
   - Combined into vector: `t = [F_thrust, M_x, M_y, M_z]`

2. **Control Allocation** (`drone_sim/px4_ctrl/px4_based_ctrl.py:555`)
   - Uses `mixerFMinv` matrix to convert commands to normalized motor speeds
   - `w²_normalized = mixerFMinv @ t`
   - Output range: [0,1] representing fraction of maximum motor capability

3. **Motor Speed Scaling** (`drone_sim/px4_ctrl/px4_based_ctrl.py:563`)
   - Converts normalized commands to actual motor speeds
   - `w²_actual = w²_normalized × w_max²`
   - Accounts for different motor specifications

4. **Physical Limiting** (`drone_sim/px4_ctrl/px4_based_ctrl.py:571`)
   - Clips motor speeds to physical limits: `[w_min, w_max]`
   - Final output: `w_cmd` in rad/s for each motor

### Matrix Construction

The mixer matrices are constructed in `drone_sim/simulation/drone_simulator.py:271`:

- **Allocation Matrices**: Pre-computed force and moment contributions
  - `Bf[i,j]`: Force from motor j in direction i at full throttle
  - `Bm[i,j]`: Moment from motor j about axis i at full throttle

- **Control Matrix**: Extracts relevant control variables
  ```python
  A_control = [thrust_row, roll_row, pitch_row, yaw_row]
  ```

- **Mixer Matrices**: 
  - `mixerFM = A_control` (forward mapping)
  - `mixerFMinv = pseudo_inverse(A_control)` (inverse mapping)

### Key Design Principles

1. **Normalized Scaling**: All allocation matrices use normalized motor commands [0,1]
2. **Physical Units**: Controller commands are in SI units (N, N·m)
3. **Robust Inversion**: Uses pseudo-inverse for over-actuated configurations
4. **Coordinate Consistency**: Body-frame moments, upward-positive thrust

### Debugging Tips

- Check mixer matrix condition number for singularity issues
- Verify motor speed ranges stay within [w_min, w_max]
- Monitor normalized commands - should stay near [0,1] for reasonable inputs
- Use debug output in controller to trace command flow

For implementation details, see the comprehensive comments in:
- `drone_sim/simulation/drone_simulator.py:271-312`
- `drone_sim/px4_ctrl/px4_based_ctrl.py:542-573`

## Documentation

For detailed documentation, examples, and API reference, see the [project wiki](https://github.com/bobzwik/Quadcopter_SimCon/wiki).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Credits

Originally developed by John Bass. Enhanced with configurable drone framework and modern Python packaging.