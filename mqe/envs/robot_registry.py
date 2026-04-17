"""
Robot Registry System for Heterogeneous Agent Support

This module provides a centralized registry for all robot types supported in the MQE environment.
To add a new robot:
1. Add robot files to mqe/envs/<robot_name>/
2. Register the robot in ROBOT_REGISTRY below
3. Provide URDF assets in resources/robots/<robot_name>/

Author: Claude
Date: 2026-01-15
"""

import importlib
import os
from typing import Dict, Any, Tuple


# ============================================================================
# ROBOT REGISTRY
# ============================================================================
# To add a new robot, add an entry here with:
# - 'class_path': Full import path to the robot class
# - 'config_path': Full import path to the robot config class
# - 'default_control': Default control type ('C' for hierarchical, 'P' for direct)
# ============================================================================

ROBOT_REGISTRY = {
    'go1': {
        'class_path': 'mqe.envs.go1.go1.Go1',
        'config_path': 'mqe.envs.go1.go1_config.Go1Cfg',
        'default_control': 'C',  # Hierarchical control with locomotion policy
        'num_actions': 3,  # [vx, vy, w] for mid-level control
        'description': 'Unitree Go1 quadruped robot with locomotion policy'
    },
    'anymal_c': {
        'class_path': 'mqe.envs.anymal_c.anymal_c.AnymalC',
        'config_path': 'mqe.envs.anymal_c.anymal_c_config.AnymalCCfg',
        'default_control': 'C',  # Hierarchical control with locomotion policy
        'num_actions': 3,  # [vx, vy, w] for mid-level control
        'description': 'ANYmal C quadruped robot with trained locomotion policy'
    },
    'cassie': {
        'class_path': 'mqe.envs.cassie.cassie.Cassie',
        'config_path': 'mqe.envs.cassie.cassie_config.CassieCfg',
        'default_control': 'C',  # Hierarchical control with locomotion policy
        'num_actions': 3,  # [vx, vy, w] for mid-level control
        'description': 'Agility Robotics Cassie biped robot with trained locomotion policy (12 DOF)'
    },
    # Add more robots here as they are implemented
}


def get_robot_class(robot_name: str):
    """
    Dynamically import and return the robot class.

    Args:
        robot_name (str): Name of the robot (e.g., 'go1', 'wheeled_bot')

    Returns:
        class: Robot class

    Raises:
        ValueError: If robot not found in registry
        ImportError: If robot class cannot be imported
    """
    if robot_name not in ROBOT_REGISTRY:
        available = ', '.join(ROBOT_REGISTRY.keys())
        raise ValueError(
            f"Robot '{robot_name}' not found in registry. "
            f"Available robots: {available}"
        )

    class_path = ROBOT_REGISTRY[robot_name]['class_path']
    module_path, class_name = class_path.rsplit('.', 1)

    try:
        module = importlib.import_module(module_path)
        robot_class = getattr(module, class_name)
        return robot_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import robot class '{class_path}'. "
            f"Make sure the robot is properly implemented. Error: {e}"
        )


def get_robot_config(robot_name: str):
    """
    Dynamically import and return the robot config class.

    Args:
        robot_name (str): Name of the robot (e.g., 'go1', 'wheeled_bot')

    Returns:
        class: Robot config class

    Raises:
        ValueError: If robot not found in registry
        ImportError: If robot config cannot be imported
    """
    if robot_name not in ROBOT_REGISTRY:
        available = ', '.join(ROBOT_REGISTRY.keys())
        raise ValueError(
            f"Robot '{robot_name}' not found in registry. "
            f"Available robots: {available}"
        )

    config_path = ROBOT_REGISTRY[robot_name]['config_path']
    module_path, class_name = config_path.rsplit('.', 1)

    try:
        module = importlib.import_module(module_path)
        config_class = getattr(module, class_name)
        return config_class
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import robot config '{config_path}'. "
            f"Make sure the robot is properly implemented. Error: {e}"
        )


def get_robot_info(robot_name: str) -> Dict[str, Any]:
    """
    Get all information about a robot from the registry.

    Args:
        robot_name (str): Name of the robot

    Returns:
        dict: Robot information including class paths, control type, etc.

    Raises:
        ValueError: If robot not found in registry
    """
    if robot_name not in ROBOT_REGISTRY:
        available = ', '.join(ROBOT_REGISTRY.keys())
        raise ValueError(
            f"Robot '{robot_name}' not found in registry. "
            f"Available robots: {available}"
        )

    return ROBOT_REGISTRY[robot_name].copy()


def register_robot(robot_name: str,
                   class_path: str,
                   config_path: str,
                   default_control: str = 'P',
                   num_actions: int = 3,
                   description: str = ''):
    """
    Programmatically register a new robot type.

    Args:
        robot_name (str): Unique name for the robot
        class_path (str): Full import path to robot class
        config_path (str): Full import path to robot config class
        default_control (str): Default control type ('C' or 'P')
        num_actions (int): Number of actions for this robot
        description (str): Human-readable description

    Raises:
        ValueError: If robot name already exists
    """
    if robot_name in ROBOT_REGISTRY:
        raise ValueError(
            f"Robot '{robot_name}' already registered. "
            f"Choose a different name or unregister first."
        )

    ROBOT_REGISTRY[robot_name] = {
        'class_path': class_path,
        'config_path': config_path,
        'default_control': default_control,
        'num_actions': num_actions,
        'description': description
    }

    print(f"[Robot Registry] Registered new robot: {robot_name}")


def unregister_robot(robot_name: str):
    """
    Remove a robot from the registry.

    Args:
        robot_name (str): Name of the robot to remove

    Raises:
        ValueError: If robot not found in registry
    """
    if robot_name not in ROBOT_REGISTRY:
        raise ValueError(f"Robot '{robot_name}' not found in registry")

    del ROBOT_REGISTRY[robot_name]
    print(f"[Robot Registry] Unregistered robot: {robot_name}")


def list_available_robots() -> list:
    """
    Get list of all registered robots.

    Returns:
        list: List of robot names
    """
    return list(ROBOT_REGISTRY.keys())


def validate_robot(robot_name: str) -> Tuple[bool, str]:
    """
    Validate that a robot is properly registered and can be imported.

    Args:
        robot_name (str): Name of the robot to validate

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    # Check if robot exists in registry
    if robot_name not in ROBOT_REGISTRY:
        available = ', '.join(ROBOT_REGISTRY.keys())
        return False, f"Robot '{robot_name}' not in registry. Available: {available}"

    # Try to import class
    try:
        robot_class = get_robot_class(robot_name)
    except Exception as e:
        return False, f"Failed to import robot class: {e}"

    # Try to import config
    try:
        robot_config = get_robot_config(robot_name)
    except Exception as e:
        return False, f"Failed to import robot config: {e}"

    return True, f"Robot '{robot_name}' is valid and ready to use"


def print_robot_registry():
    """
    Print a formatted list of all registered robots.
    """
    print("\n" + "="*70)
    print("REGISTERED ROBOTS")
    print("="*70)

    if not ROBOT_REGISTRY:
        print("No robots registered.")
        return

    for robot_name, info in ROBOT_REGISTRY.items():
        print(f"\n[{robot_name}]")
        print(f"  Class:       {info['class_path']}")
        print(f"  Config:      {info['config_path']}")
        print(f"  Control:     {info['default_control']}")
        print(f"  Actions:     {info['num_actions']}")
        print(f"  Description: {info['description']}")

    print("\n" + "="*70 + "\n")


# ============================================================================
# Command-line interface for testing
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Robot Registry System")
    print_robot_registry()

    # Test validation
    print("\nValidating registered robots...")
    for robot_name in list_available_robots():
        is_valid, message = validate_robot(robot_name)
        status = "✓" if is_valid else "✗"
        print(f"  {status} {robot_name}: {message}")
