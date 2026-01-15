"""
Heterogeneous Agent Configuration Helper Functions

This module provides utilities for handling configurations when using
multiple robot types in the same environment.

Author: Claude
Date: 2026-01-15
"""

import os
from typing import List, Dict, Any, Tuple
from copy import deepcopy
from mqe import LEGGED_GYM_ROOT_DIR
from mqe.envs.robot_registry import (
    get_robot_config,
    get_robot_info,
    validate_robot,
    list_available_robots
)


def validate_hetero_agents(agent_types: List[str]) -> Tuple[bool, str]:
    """
    Validate that all specified agent types are available and compatible.

    Args:
        agent_types (list): List of robot names (e.g., ['go1', 'wheeled_bot'])

    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if not agent_types:
        return False, "Agent types list is empty"

    if len(agent_types) < 2:
        return False, f"Need at least 2 agents, got {len(agent_types)}"

    # Validate each robot type
    for i, robot_name in enumerate(agent_types):
        is_valid, message = validate_robot(robot_name)
        if not is_valid:
            return False, f"Agent {i} ({robot_name}): {message}"

    return True, f"All {len(agent_types)} agent types are valid"


def get_hetero_asset_paths(agent_types: List[str]) -> List[str]:
    """
    Get URDF asset paths for each agent type.

    Args:
        agent_types (list): List of robot names

    Returns:
        list: List of URDF file paths (with LEGGED_GYM_ROOT_DIR placeholder)

    Example:
        >>> get_hetero_asset_paths(['go1', 'wheeled_bot'])
        ['{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf',
         '{LEGGED_GYM_ROOT_DIR}/resources/robots/wheeled_bot/urdf/wheeled_bot.urdf']
    """
    asset_paths = []

    for robot_name in agent_types:
        config_class = get_robot_config(robot_name)
        # Get the URDF path from the config
        asset_file = config_class.asset.file
        asset_paths.append(asset_file)

    return asset_paths


def get_hetero_action_dims(agent_types: List[str]) -> List[int]:
    """
    Get action dimensions for each agent type.

    Args:
        agent_types (list): List of robot names

    Returns:
        list: List of action dimensions

    Example:
        >>> get_hetero_action_dims(['go1', 'wheeled_bot'])
        [3, 2]  # go1 has 3 actions, wheeled_bot has 2
    """
    action_dims = []

    for robot_name in agent_types:
        info = get_robot_info(robot_name)
        action_dims.append(info['num_actions'])

    return action_dims


def get_max_action_dim(agent_types: List[str]) -> int:
    """
    Get the maximum action dimension across all agent types.
    Used for padding action spaces to uniform size.

    Args:
        agent_types (list): List of robot names

    Returns:
        int: Maximum action dimension
    """
    action_dims = get_hetero_action_dims(agent_types)
    return max(action_dims)


def get_hetero_control_types(agent_types: List[str]) -> List[str]:
    """
    Get default control types for each agent.

    Args:
        agent_types (list): List of robot names

    Returns:
        list: List of control types ('C' for hierarchical, 'P' for direct)
    """
    control_types = []

    for robot_name in agent_types:
        info = get_robot_info(robot_name)
        control_types.append(info['default_control'])

    return control_types


def merge_hetero_configs(base_config_class, agent_types: List[str]) -> Any:
    """
    Create a merged configuration that supports heterogeneous agents.

    This function takes a base config (usually from the primary agent like Go1)
    and extends it to support multiple agent types.

    Args:
        base_config_class: Base configuration class (e.g., Go1PushMidCfg)
        agent_types (list): List of robot names

    Returns:
        class: New config class with hetero support

    Example:
        >>> from task.cuboid.config import Go1PushMidCfg
        >>> HeteroCfg = merge_hetero_configs(Go1PushMidCfg, ['go1', 'wheeled_bot'])
    """
    # Validate agents first
    is_valid, message = validate_hetero_agents(agent_types)
    if not is_valid:
        raise ValueError(f"Invalid hetero agents: {message}")

    # Create a new config class that inherits from base
    class HeteroConfig(base_config_class):
        """Dynamically generated heterogeneous config"""
        pass

    # Add hetero-specific attributes
    HeteroConfig.hetero_agent_types = agent_types
    HeteroConfig.hetero_num_agents = len(agent_types)

    # Get asset paths for each agent
    asset_paths = get_hetero_asset_paths(agent_types)

    # Update asset configuration
    class HeteroAsset(base_config_class.asset):
        # Store original file as primary agent's file
        file_agent0 = base_config_class.asset.file
        # Add files for all agents
        hetero_files = asset_paths
        # Flag to indicate heterogeneous mode
        is_hetero = True

    HeteroConfig.asset = HeteroAsset

    # Get action dimensions
    action_dims = get_hetero_action_dims(agent_types)

    # Update env configuration
    class HeteroEnv(base_config_class.env):
        hetero_action_dims = action_dims
        max_action_dim = max(action_dims)
        # Store original num_actions for reference
        num_actions_per_agent = action_dims

    HeteroConfig.env = HeteroEnv

    # Get control types
    control_types = get_hetero_control_types(agent_types)

    # Update control configuration
    class HeteroControl(base_config_class.control):
        hetero_control_types = control_types

    HeteroConfig.control = HeteroControl

    return HeteroConfig


def create_hetero_config(base_config_class,
                        agent_types: List[str],
                        validate: bool = True) -> Any:
    """
    High-level function to create heterogeneous configuration.

    Args:
        base_config_class: Base config class (e.g., Go1PushMidCfg)
        agent_types (list): List of robot names (e.g., ['go1', 'wheeled_bot'])
        validate (bool): Whether to validate agents before creating config

    Returns:
        class: Heterogeneous configuration class

    Raises:
        ValueError: If validation fails

    Example:
        >>> from task.cuboid.config import Go1PushMidCfg
        >>> config = create_hetero_config(Go1PushMidCfg, ['go1', 'wheeled_bot'])
        >>> env = make_hetero_env('go1push_mid', args, config)
    """
    if validate:
        is_valid, message = validate_hetero_agents(agent_types)
        if not is_valid:
            raise ValueError(f"Hetero config validation failed: {message}")

    return merge_hetero_configs(base_config_class, agent_types)


def print_hetero_config_summary(agent_types: List[str]):
    """
    Print a summary of the heterogeneous configuration.

    Args:
        agent_types (list): List of robot names
    """
    print("\n" + "="*70)
    print("HETEROGENEOUS AGENT CONFIGURATION")
    print("="*70)

    # Validate
    is_valid, message = validate_hetero_agents(agent_types)
    if not is_valid:
        print(f"\n❌ INVALID: {message}\n")
        return

    print(f"\n✓ Configuration is valid")
    print(f"\nNumber of agents: {len(agent_types)}")

    # Print each agent's info
    for i, robot_name in enumerate(agent_types):
        info = get_robot_info(robot_name)
        asset_paths = get_hetero_asset_paths(agent_types)

        print(f"\n--- Agent {i}: {robot_name} ---")
        print(f"  Actions:     {info['num_actions']} DOF")
        print(f"  Control:     {info['default_control']}")
        print(f"  URDF:        {asset_paths[i]}")
        print(f"  Description: {info['description']}")

    # Print summary
    action_dims = get_hetero_action_dims(agent_types)
    max_action = get_max_action_dim(agent_types)

    print(f"\nAction Dimensions: {action_dims}")
    print(f"Max Action Dim (for padding): {max_action}")

    print("\n" + "="*70 + "\n")


def is_homogeneous(agent_types: List[str]) -> bool:
    """
    Check if all agents are of the same type.

    Args:
        agent_types (list): List of robot names

    Returns:
        bool: True if all agents are identical
    """
    if not agent_types:
        return True

    first_type = agent_types[0]
    return all(robot == first_type for robot in agent_types)


def get_hetero_info_dict(agent_types: List[str]) -> Dict[str, Any]:
    """
    Get comprehensive information about heterogeneous setup.

    Args:
        agent_types (list): List of robot names

    Returns:
        dict: Information dictionary with all relevant hetero info
    """
    return {
        'agent_types': agent_types,
        'num_agents': len(agent_types),
        'is_homogeneous': is_homogeneous(agent_types),
        'action_dims': get_hetero_action_dims(agent_types),
        'max_action_dim': get_max_action_dim(agent_types),
        'control_types': get_hetero_control_types(agent_types),
        'asset_paths': get_hetero_asset_paths(agent_types),
    }


# ============================================================================
# Command-line interface for testing
# ============================================================================

if __name__ == "__main__":
    import sys

    print("Heterogeneous Configuration Helper\n")

    # Test with homogeneous agents
    print("Test 1: Homogeneous (2x go1)")
    print("-" * 70)
    agent_types_homo = ['go1', 'go1']
    print_hetero_config_summary(agent_types_homo)

    # Test with heterogeneous agents (will fail until we add another robot)
    print("\nTest 2: Heterogeneous (go1 + wheeled_bot)")
    print("-" * 70)
    agent_types_hetero = ['go1', 'wheeled_bot']
    print_hetero_config_summary(agent_types_hetero)

    print("\nAvailable robots:")
    for robot in list_available_robots():
        print(f"  - {robot}")
