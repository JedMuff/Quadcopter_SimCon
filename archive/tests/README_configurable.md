# Configurable Drone Simulation Framework

This enhanced simulation framework allows you to simulate any multi-rotor drone configuration with automatic computation of physical properties and control allocation matrices from propeller placement.

## Features

- **Flexible Drone Configurations**: Support for quadrotors, hexarotors, tricopters, octorotors, and custom configurations
- **Automatic Property Computation**: Mass, center of gravity, inertia matrix, and control allocation matrices computed automatically from propeller specifications
- **Backward Compatibility**: Works with existing controller framework and trajectory planning
- **Configuration Files**: JSON-based configuration for easy drone specification
- **Command Line Interface**: Simple command-line options for quick simulation setup

## Quick Start

### Basic Usage

```bash
# Run simulation with default quadrotor
python run_3D_simulation_configurable.py

# Run with hexarotor configuration
python run_3D_simulation_configurable.py --type hex

# Run with custom arm length and propeller size
python run_3D_simulation_configurable.py --type quad --arm-length 0.15 --prop-size 6

# Show detailed drone information
python run_3D_simulation_configurable.py --type octo --info
```

### Configuration Files

#### Standard Configuration Format
```json
{
  "type": "quad",           // quad, hex, tri, octo
  "arm_length": 0.11,       // arm length in meters
  "prop_size": 5,           // propeller size in inches (4-8)
  "description": "Description of the configuration"
}
```

#### Custom Propeller Configuration
```json
{
  "propellers": [
    {
      "loc": [0.11, 0.11, 0],      // [x, y, z] position in body frame (meters)
      "dir": [0, 0, -1, "ccw"],    // [x, y, z, rotation] thrust direction and spin
      "propsize": 5                // propeller size in inches
    },
    // ... more propellers
  ],
  "description": "Custom configuration description"
}
```

## Command Line Options

- `--config FILE`: Load configuration from JSON file
- `--type TYPE`: Standard drone type (quad, hex, tri, octo)  
- `--arm-length LENGTH`: Arm length in meters
- `--prop-size SIZE`: Propeller size in inches (4-8)
- `--time TIME`: Simulation time in seconds (default: 20)
- `--dt TIMESTEP`: Time step in seconds (default: 0.005)
- `--save`: Save animation
- `--info`: Print detailed drone configuration

## Drone Configuration Examples

### Standard Quadrotor
```bash
python run_3D_simulation_configurable.py --type quad --arm-length 0.11 --prop-size 5
```

### Large Hexarotor  
```bash
python run_3D_simulation_configurable.py --type hex --arm-length 0.15 --prop-size 6
```

### Custom Configuration
Create a JSON file with specific propeller placement:
```json
{
  "propellers": [
    {"loc": [0.12, 0.12, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
    {"loc": [-0.12, 0.12, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
    {"loc": [-0.12, -0.12, 0], "dir": [0, 0, -1, "ccw"], "propsize": 5},
    {"loc": [0.12, -0.12, 0], "dir": [0, 0, -1, "cw"], "propsize": 5},
    {"loc": [0, 0.17, 0], "dir": [0, 0, -1, "cw"], "propsize": 4},
    {"loc": [0, -0.17, 0], "dir": [0, 0, -1, "ccw"], "propsize": 4}
  ]
}
```

## Propeller Specifications

The framework includes realistic propeller data for sizes 4-8 inches:

| Size | k_f (N⋅s²⋅m⁻²) | k_m (N⋅m⋅s²⋅m⁻²) | Max RPM | Mass (kg) |
|------|-------------|-------------|---------|-----------|
| 4"   | 7.24e-07    | 8.20e-09    | 3927    | 0.018     |
| 5"   | 1.08e-06    | 1.22e-08    | 3142    | 0.0196    |
| 6"   | 2.21e-06    | 2.74e-08    | 2618    | 0.0252    |
| 7"   | 4.65e-06    | 6.62e-08    | 2244    | 0.046     |
| 8"   | 7.60e-06    | 1.14e-07    | 1963    | 0.056     |

## Physical Property Computation

The framework automatically computes:

- **Mass**: Controller + propeller/motor + structural mass
- **Center of Gravity**: Weighted average of component positions
- **Inertia Matrix**: Full 6-DOF inertia with products of inertia
- **Control Allocation**: Force and moment allocation matrices for any configuration

## Integration with Existing Framework

The configurable system is fully compatible with the existing:
- PID controllers in `ctrl.py`
- Trajectory planning in `trajectory.py` 
- Visualization and animation in `utils/`
- Wind models and disturbances

## Advanced Features

### Over-Actuated Systems
The framework automatically handles over-actuated systems (>6 DOF control):
- Uses pseudo-inverse for control allocation
- Provides optimal motor command distribution
- Handles redundancy gracefully

### Configuration Analysis
Use `--info` flag to see detailed analysis:
- Mass and inertia properties
- Control authority analysis  
- Actuator configuration metrics
- Motor placement and specifications

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Configuration Errors**: Validate JSON syntax and propeller specifications
3. **Simulation Instability**: Check propeller placement and mass distribution
4. **Controller Performance**: May need PID tuning for different configurations

### Validation

The framework includes validation for:
- Propeller size availability (4-8 inches)
- Configuration format correctness
- Physical property computation
- Control allocation matrix properties

## Technical Details

### State Representation
- Position: [x, y, z] in world frame
- Velocity: [vx, vy, vz] in world frame  
- Attitude: [φ, θ, ψ] (roll, pitch, yaw)
- Angular velocity: [p, q, r] in body frame

### Control Allocation
Uses 6-DOF wrench allocation:
```
[Fx, Fy, Fz, Mx, My, Mz]ᵀ = B × [u₁, u₂, ..., uₙ]ᵀ
```

Where B is the allocation matrix computed from propeller geometry and specifications.

## Future Enhancements

- Support for tilted propellers
- Variable pitch propellers  
- Asymmetric configurations
- Real-time parameter adjustment
- Hardware-in-the-loop integration