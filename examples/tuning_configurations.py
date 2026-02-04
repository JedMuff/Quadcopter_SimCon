#!/usr/bin/env python3
"""
Configuration templates for controller parameter tuning.

This module provides predefined configurations for different drone types,
mission types, and parameter sets to facilitate systematic tuning.
"""

import numpy as np
from typing import Dict, List, Tuple, Any

# Default drone configurations for tuning
DRONE_CONFIGURATIONS = {
    'micro_quad': {
        'drone_type': 'quad',
        'arm_length': 0.065,
        'prop_size': 4,
        'description': 'Micro quadcopter (65mm, 4" props)'
    },
    'small_quad': {
        'drone_type': 'quad', 
        'arm_length': 0.11,
        'prop_size': 4,
        'description': 'Small quadcopter (110mm, 4" props)'
    },
    'racing_quad': {
        'drone_type': 'quad',
        'arm_length': 0.14,
        'prop_size': 5,
        'description': 'Racing quadcopter (140mm, 5" props)'
    },
    'standard_quad': {
        'drone_type': 'quad',
        'arm_length': 0.16,
        'prop_size': 5,
        'description': 'Standard quadcopter (160mm, 5" props)'
    },
    'large_quad': {
        'drone_type': 'quad',
        'arm_length': 0.25,
        'prop_size': 8,
        'description': 'Large quadcopter (250mm, 8" props)'
    },
    'cinema_quad': {
        'drone_type': 'quad',
        'arm_length': 0.35,
        'prop_size': 10,
        'description': 'Cinema quadcopter (350mm, 10" props)'
    },
    'compact_hex': {
        'drone_type': 'hex',
        'arm_length': 0.12,
        'prop_size': 5,
        'description': 'Compact hexarotor (120mm, 5" props)'
    },
    'standard_hex': {
        'drone_type': 'hex',
        'arm_length': 0.18,
        'prop_size': 6,
        'description': 'Standard hexarotor (180mm, 6" props)'
    },
    'large_hex': {
        'drone_type': 'hex',
        'arm_length': 0.28,
        'prop_size': 8,
        'description': 'Large hexarotor (280mm, 8" props)'
    },
    'compact_octo': {
        'drone_type': 'octo',
        'arm_length': 0.15,
        'prop_size': 5,
        'description': 'Compact octorotor (150mm, 5" props)'
    },
    'standard_octo': {
        'drone_type': 'octo',
        'arm_length': 0.22,
        'prop_size': 6,
        'description': 'Standard octorotor (220mm, 6" props)'
    },
    'heavy_lift_octo': {
        'drone_type': 'octo',
        'arm_length': 0.35,
        'prop_size': 10,
        'description': 'Heavy lift octorotor (350mm, 10" props)'
    }
}

# Mission/trajectory configurations
MISSION_CONFIGURATIONS = {
    'hover_stability': {
        'trajectory_type': 'hover',
        'waypoints': [[0, 0, -2.0], [0, 0, -2.0], [0, 0, -2.0]],
        'duration': 30.0,
        'description': 'Hover stability at 2m altitude',
        'performance_priorities': ['stability', 'efficiency'],
        'cost_weights': {
            'tracking_weight': 0.8,
            'stability_weight': 1.2,
            'efficiency_weight': 0.6
        }
    },
    'altitude_hold': {
        'trajectory_type': 'step',
        'waypoints': [[0, 0, -1.0], [0, 0, -3.0], [0, 0, -3.0]],
        'duration': 25.0,
        'description': 'Altitude change and hold',
        'performance_priorities': ['tracking', 'stability'],
        'cost_weights': {
            'tracking_weight': 1.0,
            'stability_weight': 1.0,
            'efficiency_weight': 0.4
        }
    },
    'position_step': {
        'trajectory_type': 'step',
        'waypoints': [[0, 0, -2], [3, 0, -2], [3, 0, -2]],
        'duration': 25.0,
        'description': 'Position step response in X direction',
        'performance_priorities': ['tracking', 'stability'],
        'cost_weights': {
            'tracking_weight': 1.2,
            'stability_weight': 1.0,
            'efficiency_weight': 0.3
        }
    },
    'square_pattern': {
        'trajectory_type': 'waypoint',
        'waypoints': [[1, 1, -2], [1, -1, -2], [-1, -1, -2], [-1, 1, -2], [1, 1, -2]],
        'duration': 40.0,
        'description': '2m square pattern at 2m altitude',
        'performance_priorities': ['tracking', 'efficiency'],
        'cost_weights': {
            'tracking_weight': 1.0,
            'stability_weight': 0.8,
            'efficiency_weight': 0.7
        }
    },
    'figure_eight': {
        'trajectory_type': 'smooth',
        'waypoints': [[2, 0, -2], [1, 1, -2], [0, 0, -2], [-1, -1, -2], [-2, 0, -2], 
                     [-1, 1, -2], [0, 0, -2], [1, -1, -2], [2, 0, -2]],
        'duration': 50.0,
        'description': 'Figure-8 smooth trajectory',
        'performance_priorities': ['tracking', 'efficiency'],
        'cost_weights': {
            'tracking_weight': 1.0,
            'stability_weight': 0.6,
            'efficiency_weight': 0.8
        }
    },
    'vertical_traverse': {
        'trajectory_type': 'waypoint',
        'waypoints': [[0, 0, -0.5], [0, 0, -1.5], [0, 0, -2.5], [0, 0, -3.5], [0, 0, -2.0]],
        'duration': 35.0,
        'description': 'Vertical altitude traversal',
        'performance_priorities': ['tracking', 'stability'],
        'cost_weights': {
            'tracking_weight': 1.1,
            'stability_weight': 0.9,
            'efficiency_weight': 0.4
        }
    },
    'aggressive_maneuver': {
        'trajectory_type': 'waypoint',
        'waypoints': [[0, 0, -2], [3, 3, -1], [-3, 3, -3], [-3, -3, -1], [3, -3, -3], [0, 0, -2]],
        'duration': 30.0,
        'description': 'Aggressive 3D maneuvering',
        'performance_priorities': ['tracking', 'stability'],
        'cost_weights': {
            'tracking_weight': 1.3,
            'stability_weight': 1.1,
            'efficiency_weight': 0.2
        }
    },
    'precision_landing': {
        'trajectory_type': 'waypoint',
        'waypoints': [[0, 0, -3], [0, 0, -2], [0, 0, -1], [0, 0, -0.2]],
        'duration': 25.0,
        'description': 'Precision landing sequence',
        'performance_priorities': ['tracking', 'stability'],
        'cost_weights': {
            'tracking_weight': 1.4,
            'stability_weight': 1.2,
            'efficiency_weight': 0.3
        }
    }
}

# Parameter sets for tuning
PARAMETER_SETS = {
    'position_control': {
        'parameters': ['pos_P_gain'],
        'description': 'Position loop gains only',
        'bounds': {
            'pos_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]}
        },
        'initial_range': {
            'pos_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 3.0]}
        }
    },
    'velocity_control': {
        'parameters': ['vel_P_gain', 'vel_D_gain', 'vel_I_gain'],
        'description': 'Velocity loop PID gains',
        'bounds': {
            'vel_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [20.0, 20.0, 20.0]},
            'vel_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [2.0, 2.0, 2.0]},
            'vel_I_gain': {'min': [0.1, 0.1, 0.1], 'max': [20.0, 20.0, 20.0]}
        },
        'initial_range': {
            'vel_P_gain': {'min': [2.0, 2.0, 2.0], 'max': [8.0, 8.0, 6.0]},
            'vel_D_gain': {'min': [0.1, 0.1, 0.1], 'max': [1.0, 1.0, 1.0]},
            'vel_I_gain': {'min': [1.0, 1.0, 1.0], 'max': [10.0, 10.0, 10.0]}
        }
    },
    'attitude_control': {
        'parameters': ['att_P_gain'],
        'description': 'Attitude control gains',
        'bounds': {
            'att_P_gain': {'min': [1.0, 1.0, 0.5], 'max': [30.0, 30.0, 10.0]}
        },
        'initial_range': {
            'att_P_gain': {'min': [4.0, 4.0, 1.0], 'max': [15.0, 15.0, 3.0]}
        }
    },
    'rate_control': {
        'parameters': ['rate_P_gain', 'rate_D_gain'],
        'description': 'Angular rate control gains',
        'bounds': {
            'rate_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]},
            'rate_D_gain': {'min': [0.001, 0.001, 0.001], 'max': [0.5, 0.5, 0.5]}
        },
        'initial_range': {
            'rate_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 2.0]},
            'rate_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [0.2, 0.2, 0.2]}
        }
    },
    'control_limits': {
        'parameters': ['vel_max', 'tilt_max', 'rate_max'],
        'description': 'Control authority limits',
        'bounds': {
            'vel_max': {'min': [0.5, 0.5, 0.5], 'max': [25.0, 25.0, 25.0]},
            'tilt_max': {'min': 10.0 * np.pi/180, 'max': 70.0 * np.pi/180},
            'rate_max': {'min': [30.0 * np.pi/180, 30.0 * np.pi/180, 30.0 * np.pi/180],
                        'max': [500.0 * np.pi/180, 500.0 * np.pi/180, 500.0 * np.pi/180]}
        },
        'initial_range': {
            'vel_max': {'min': [2.0, 2.0, 2.0], 'max': [10.0, 10.0, 10.0]},
            'tilt_max': {'min': 20.0 * np.pi/180, 'max': 60.0 * np.pi/180},
            'rate_max': {'min': [100.0 * np.pi/180, 100.0 * np.pi/180, 50.0 * np.pi/180],
                        'max': [300.0 * np.pi/180, 300.0 * np.pi/180, 200.0 * np.pi/180]}
        }
    },
    'full_control': {
        'parameters': ['pos_P_gain', 'vel_P_gain', 'vel_D_gain', 'vel_I_gain', 
                      'att_P_gain', 'rate_P_gain', 'rate_D_gain'],
        'description': 'All control gains (no limits)',
        'bounds': {
            'pos_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]},
            'vel_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [20.0, 20.0, 20.0]},
            'vel_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [2.0, 2.0, 2.0]},
            'vel_I_gain': {'min': [0.1, 0.1, 0.1], 'max': [20.0, 20.0, 20.0]},
            'att_P_gain': {'min': [1.0, 1.0, 0.5], 'max': [30.0, 30.0, 10.0]},
            'rate_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]},
            'rate_D_gain': {'min': [0.001, 0.001, 0.001], 'max': [0.5, 0.5, 0.5]}
        },
        'initial_range': {
            'pos_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 3.0]},
            'vel_P_gain': {'min': [2.0, 2.0, 2.0], 'max': [8.0, 8.0, 6.0]},
            'vel_D_gain': {'min': [0.1, 0.1, 0.1], 'max': [1.0, 1.0, 1.0]},
            'vel_I_gain': {'min': [1.0, 1.0, 1.0], 'max': [10.0, 10.0, 10.0]},
            'att_P_gain': {'min': [4.0, 4.0, 1.0], 'max': [15.0, 15.0, 3.0]},
            'rate_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 2.0]},
            'rate_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [0.2, 0.2, 0.2]}
        }
    },
    'complete_system': {
        'parameters': ['pos_P_gain', 'vel_P_gain', 'vel_D_gain', 'vel_I_gain',
                      'att_P_gain', 'rate_P_gain', 'rate_D_gain', 
                      'vel_max', 'tilt_max', 'rate_max'],
        'description': 'All gains and limits',
        'bounds': {
            'pos_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]},
            'vel_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [20.0, 20.0, 20.0]},
            'vel_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [2.0, 2.0, 2.0]},
            'vel_I_gain': {'min': [0.1, 0.1, 0.1], 'max': [20.0, 20.0, 20.0]},
            'att_P_gain': {'min': [1.0, 1.0, 0.5], 'max': [30.0, 30.0, 10.0]},
            'rate_P_gain': {'min': [0.1, 0.1, 0.1], 'max': [10.0, 10.0, 10.0]},
            'rate_D_gain': {'min': [0.001, 0.001, 0.001], 'max': [0.5, 0.5, 0.5]},
            'vel_max': {'min': [0.5, 0.5, 0.5], 'max': [25.0, 25.0, 25.0]},
            'tilt_max': {'min': 10.0 * np.pi/180, 'max': 70.0 * np.pi/180},
            'rate_max': {'min': [30.0 * np.pi/180, 30.0 * np.pi/180, 30.0 * np.pi/180],
                        'max': [500.0 * np.pi/180, 500.0 * np.pi/180, 500.0 * np.pi/180]}
        },
        'initial_range': {
            'pos_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 3.0]},
            'vel_P_gain': {'min': [2.0, 2.0, 2.0], 'max': [8.0, 8.0, 6.0]},
            'vel_D_gain': {'min': [0.1, 0.1, 0.1], 'max': [1.0, 1.0, 1.0]},
            'vel_I_gain': {'min': [1.0, 1.0, 1.0], 'max': [10.0, 10.0, 10.0]},
            'att_P_gain': {'min': [4.0, 4.0, 1.0], 'max': [15.0, 15.0, 3.0]},
            'rate_P_gain': {'min': [0.5, 0.5, 0.5], 'max': [3.0, 3.0, 2.0]},
            'rate_D_gain': {'min': [0.01, 0.01, 0.01], 'max': [0.2, 0.2, 0.2]},
            'vel_max': {'min': [2.0, 2.0, 2.0], 'max': [10.0, 10.0, 10.0]},
            'tilt_max': {'min': 20.0 * np.pi/180, 'max': 60.0 * np.pi/180},
            'rate_max': {'min': [100.0 * np.pi/180, 100.0 * np.pi/180, 50.0 * np.pi/180],
                        'max': [300.0 * np.pi/180, 300.0 * np.pi/180, 200.0 * np.pi/180]}
        }
    }
}

# Predefined tuning scenarios
TUNING_SCENARIOS = {
    'micro_hover': {
        'drone_config': 'micro_quad',
        'mission_config': 'hover_stability',
        'parameter_set': 'position_control',
        'description': 'Micro quad hover tuning'
    },
    'racing_agility': {
        'drone_config': 'racing_quad',
        'mission_config': 'aggressive_maneuver',
        'parameter_set': 'full_control',
        'description': 'Racing quad agility tuning'
    },
    'cinema_smooth': {
        'drone_config': 'cinema_quad',
        'mission_config': 'figure_eight',
        'parameter_set': 'complete_system',
        'description': 'Cinema quad smooth tracking'
    },
    'heavy_lift_stability': {
        'drone_config': 'heavy_lift_octo',
        'mission_config': 'precision_landing',
        'parameter_set': 'velocity_control',
        'description': 'Heavy lift precision control'
    },
    'multi_rotor_comparison': {
        'drone_configs': ['standard_quad', 'standard_hex', 'standard_octo'],
        'mission_config': 'square_pattern',
        'parameter_set': 'full_control',
        'description': 'Compare tuning across multi-rotor types'
    }
}

def get_tuning_configuration(scenario_name: str) -> Dict:
    """
    Get complete tuning configuration for a predefined scenario.
    
    Args:
        scenario_name: Name of the tuning scenario
    
    Returns:
        Dictionary containing complete configuration
    """
    if scenario_name not in TUNING_SCENARIOS:
        raise ValueError(f"Unknown tuning scenario: {scenario_name}")
    
    scenario = TUNING_SCENARIOS[scenario_name]
    config = {}
    
    # Handle single or multiple drone configurations
    if 'drone_config' in scenario:
        config['drone_config'] = DRONE_CONFIGURATIONS[scenario['drone_config']]
    elif 'drone_configs' in scenario:
        config['drone_configs'] = [DRONE_CONFIGURATIONS[name] for name in scenario['drone_configs']]
    
    config['mission_config'] = MISSION_CONFIGURATIONS[scenario['mission_config']]
    config['parameter_set'] = PARAMETER_SETS[scenario['parameter_set']]
    config['description'] = scenario['description']
    
    return config

def get_parameter_bounds(parameter_names: List[str]) -> Dict:
    """
    Get parameter bounds for a list of parameter names.
    
    Args:
        parameter_names: List of parameter names to get bounds for
    
    Returns:
        Dictionary of parameter bounds
    """
    bounds = {}
    
    for param_set in PARAMETER_SETS.values():
        for param_name in parameter_names:
            if param_name in param_set['bounds']:
                bounds[param_name] = param_set['bounds'][param_name]
                break
    
    return bounds

def create_custom_scenario(drone_config: str, mission_config: str, 
                          parameter_set: str, description: str = "") -> Dict:
    """
    Create a custom tuning scenario from existing configurations.
    
    Args:
        drone_config: Name of drone configuration
        mission_config: Name of mission configuration
        parameter_set: Name of parameter set
        description: Optional description
    
    Returns:
        Custom scenario configuration
    """
    if drone_config not in DRONE_CONFIGURATIONS:
        raise ValueError(f"Unknown drone config: {drone_config}")
    if mission_config not in MISSION_CONFIGURATIONS:
        raise ValueError(f"Unknown mission config: {mission_config}")
    if parameter_set not in PARAMETER_SETS:
        raise ValueError(f"Unknown parameter set: {parameter_set}")
    
    return {
        'drone_config': DRONE_CONFIGURATIONS[drone_config],
        'mission_config': MISSION_CONFIGURATIONS[mission_config],
        'parameter_set': PARAMETER_SETS[parameter_set],
        'description': description or f"{drone_config} {mission_config} tuning"
    }

def get_default_parameters() -> Dict:
    """
    Get default controller parameters (matching GeneralizedControl defaults).
    
    Returns:
        Dictionary of default parameter values
    """
    return {
        'pos_P_gain': np.array([1.0, 1.0, 1.0]),
        'vel_P_gain': np.array([5.0, 5.0, 4.0]),
        'vel_D_gain': np.array([0.5, 0.5, 0.5]),
        'vel_I_gain': np.array([5.0, 5.0, 5.0]),
        'att_P_gain': np.array([8.0, 8.0, 1.5]),
        'rate_P_gain': np.array([1.5, 1.5, 1.0]),
        'rate_D_gain': np.array([0.04, 0.04, 0.1]),
        'vel_max': np.array([5.0, 5.0, 5.0]),
        'tilt_max': 50.0 * np.pi/180,
        'rate_max': np.array([200.0 * np.pi/180, 200.0 * np.pi/180, 150.0 * np.pi/180])
    }

def generate_random_parameters(parameter_set_name: str, n_samples: int = 1) -> List[Dict]:
    """
    Generate random parameter sets within the initial range for a parameter set.
    
    Args:
        parameter_set_name: Name of parameter set
        n_samples: Number of random samples to generate
    
    Returns:
        List of parameter dictionaries
    """
    if parameter_set_name not in PARAMETER_SETS:
        raise ValueError(f"Unknown parameter set: {parameter_set_name}")
    
    param_set = PARAMETER_SETS[parameter_set_name]
    initial_range = param_set['initial_range']
    parameter_samples = []
    
    for _ in range(n_samples):
        params = {}
        for param_name in param_set['parameters']:
            if param_name in initial_range:
                min_val = np.array(initial_range[param_name]['min'])
                max_val = np.array(initial_range[param_name]['max'])
                
                # Generate random values within range
                random_vals = np.random.uniform(0, 1, size=min_val.shape)
                params[param_name] = min_val + random_vals * (max_val - min_val)
            else:
                # Use default value if not in initial range
                defaults = get_default_parameters()
                if param_name in defaults:
                    params[param_name] = defaults[param_name]
        
        parameter_samples.append(params)
    
    return parameter_samples