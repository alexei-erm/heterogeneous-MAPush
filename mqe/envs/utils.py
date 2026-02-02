# environments
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.go1.go1 import Go1
from mqe.envs.npc.go1_object import Go1Object

# configs
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg
from mqe.envs.configs.go1_push_upper_config import Go1PushUpperCfg


# wrappers
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from mqe.envs.wrappers.go1_push_mid_wrapper import Go1PushMidWrapper
from mqe.envs.wrappers.go1_push_upper_wrapper import Go1PushUpperWrapper

from mqe.utils import make_env

from typing import Tuple

ENV_DICT = {
    "go1push_mid": {
        "class": Go1Object,
        "config": Go1PushMidCfg,
        "wrapper": Go1PushMidWrapper
    },
    "go1push_upper": {
        "class": Go1Object,
        "config": Go1PushUpperCfg,
        "wrapper": Go1PushUpperWrapper
    },
}

def make_mqe_env(env_name: str, args=None, custom_cfg=None) -> Tuple[LeggedRobotField, LeggedRobotFieldCfg]:
    env_dict = ENV_DICT[env_name]

    if callable(custom_cfg):
        env_dict["config"] = custom_cfg(env_dict["config"])

    env, env_cfg = make_env(env_dict["class"], env_dict["config"], args)

    env = env_dict["wrapper"](env)

    return env, env_cfg


def make_hetero_env(env_name: str, agent_types: list, args=None, custom_cfg=None) -> Tuple[LeggedRobotField, LeggedRobotFieldCfg]:
    """
    Create a multi-agent environment with specified robot types.

    If both agents are the same type, creates a homogeneous environment using
    that robot's native class. If agents differ, creates a heterogeneous
    environment using HeteroRobot.

    Args:
        env_name: Name of the environment (e.g., 'go1push_mid')
        agent_types: List of robot type names (e.g., ['go1', 'cassie'])
        args: Environment arguments
        custom_cfg: Custom configuration function

    Returns:
        env: Wrapped environment
        env_cfg: Environment configuration

    Example:
        >>> env, cfg = make_hetero_env('go1push_mid', ['go1', 'cassie'], args)  # Hetero
        >>> env, cfg = make_hetero_env('go1push_mid', ['cassie', 'cassie'], args)  # Homogeneous Cassie
    """
    from mqe.envs.robot_registry import get_robot_class, get_robot_config

    env_dict = ENV_DICT[env_name]
    base_config = env_dict["config"]
    base_task_class = env_dict["class"]

    # Check if homogeneous (both agents same type)
    is_homogeneous = (len(set(agent_types)) == 1)
    robot_type = agent_types[0]

    # For Go1, use native Go1Object class directly (true homogeneous)
    # For other robots, use HeteroRobot with same-robot config (pseudo-homogeneous)
    # This avoids MRO conflicts since Go1Object inherits from Go1
    use_native_class = (is_homogeneous and robot_type == 'go1')

    if use_native_class:
        # TRUE HOMOGENEOUS: Go1 only - use Go1Object directly
        task_class = base_task_class
        merged_config = base_config

        # Apply custom config if provided
        if callable(custom_cfg):
            merged_config = custom_cfg(merged_config)

        env, env_cfg = make_env(task_class, merged_config, args)

        # Apply wrapper
        env = env_dict["wrapper"](env)

        print(f"[make_hetero_env] Created HOMOGENEOUS environment:")
        print(f"  Task: {env_name}")
        print(f"  Robot: {robot_type} x 2")
        print(f"  Class: {task_class.__name__}")

    else:
        # HETEROGENEOUS: Use HeteroRobot
        from mqe.utils.hetero_config import create_hetero_config
        from mqe.envs.base.hetero_robot import HeteroRobot

        # Create heterogeneous configuration
        hetero_config = create_hetero_config(base_config, agent_types)

        # Apply custom config if provided
        if callable(custom_cfg):
            hetero_config = custom_cfg(hetero_config)

        # Create a dynamic class that inherits from both the base task class and HeteroRobot
        class HeteroTask(HeteroRobot, base_task_class):
            """Dynamically created heterogeneous task class."""
            pass

        env, env_cfg = make_env(HeteroTask, hetero_config, args)

        # Apply wrapper
        env = env_dict["wrapper"](env)

        print(f"[make_hetero_env] Created HETEROGENEOUS environment:")
        print(f"  Task: {env_name}")
        print(f"  Agents: {agent_types}")
        print(f"  Class: HeteroRobot + {base_task_class.__name__}")

    return env, env_cfg


def _merge_configs_for_homogeneous(base_config, robot_config_class, robot_type):
    """
    Merge base task config with robot-specific config for homogeneous environments.

    Strategy: Inherit from base_config (has all task settings like NPC, rewards, terrain)
    and override only the robot-specific parts (control, asset URDF path, init heights).

    For Go1, the base config is already correct (Go1PushMidCfg inherits from Go1Cfg).
    For other robots, we inherit from base_config and override robot-specific parts.
    """
    # For Go1, use base config directly (it's already Go1-based)
    if robot_type == 'go1':
        return base_config

    # For other robots, inherit from base_config and override robot-specific settings
    class MergedConfig(base_config):
        """Merged config: inherits task settings from base, overrides robot-specific parts."""
        pass

    # Override control settings from robot config
    if hasattr(robot_config_class, 'control'):
        class MergedControl(base_config.control):
            pass
        # Copy robot-specific control attributes
        for attr in dir(robot_config_class.control):
            if not attr.startswith('_'):
                try:
                    setattr(MergedControl, attr, getattr(robot_config_class.control, attr))
                except (AttributeError, TypeError):
                    pass
        MergedConfig.control = MergedControl

    # Override asset settings (URDF path, etc.) from robot config
    if hasattr(robot_config_class, 'asset'):
        class MergedAsset(base_config.asset):
            pass
        # Copy robot-specific asset attributes (file path, foot_name, etc.)
        robot_asset_attrs = ['file', 'name', 'foot_name', 'penalize_contacts_on',
                            'terminate_after_contacts_on', 'self_collisions',
                            'flip_visual_attachments', 'fix_base_link', 'collapse_fixed_joints']
        for attr in robot_asset_attrs:
            if hasattr(robot_config_class.asset, attr):
                try:
                    setattr(MergedAsset, attr, getattr(robot_config_class.asset, attr))
                except (AttributeError, TypeError):
                    pass
        MergedConfig.asset = MergedAsset

    # Override init_state heights for the robot (but keep NPC init states from base)
    if hasattr(robot_config_class, 'init_state'):
        class MergedInitState(base_config.init_state):
            pass
        # Copy robot init height/pos from robot config
        if hasattr(robot_config_class.init_state, 'pos'):
            MergedInitState.pos = robot_config_class.init_state.pos
        if hasattr(robot_config_class.init_state, 'default_joint_angles'):
            MergedInitState.default_joint_angles = robot_config_class.init_state.default_joint_angles

        # Update init_states for the robot agents (adjust heights)
        if hasattr(base_config.init_state, 'init_states') and hasattr(robot_config_class.init_state, 'pos'):
            robot_height = robot_config_class.init_state.pos[2]
            # Create new init_states with robot-appropriate heights
            new_init_states = []
            for state in base_config.init_state.init_states:
                new_state = type(state)(
                    pos=[state.pos[0], state.pos[1], robot_height],
                    rot=state.rot,
                    lin_vel=state.lin_vel,
                    ang_vel=state.ang_vel
                )
                new_init_states.append(new_state)
            MergedInitState.init_states = new_init_states

        MergedConfig.init_state = MergedInitState

    return MergedConfig

def custom_cfg(args, individualized_rewards=False, shared_gated_rewards=False, cooperation_rewards=False, mapush_og_rewards_teamified=False, reward_scale_testing=False, collaboration_rewards=False, positive_approachtobox_reward=False, agent0='go1', agent1='go1', baseline_mappo_rewards=False):

    def fn(cfg:LeggedRobotFieldCfg):

        if getattr(args, "num_envs", None) is not None:
            cfg.env.num_envs = args.num_envs

        cfg.env.record_video = args.record_video

        # Enable heterogeneous agents if agent types differ
        is_hetero = (agent0 != agent1)
        if is_hetero and hasattr(cfg, 'hetero'):
            cfg.hetero.use_hetero = True
            cfg.hetero.hetero_agent_types = [agent0, agent1]
            print(f"[custom_cfg] Enabled hetero mode: agent0={agent0}, agent1={agent1}")

        # MAPPO BASELINE MODE: Use original MAPush rewards only
        # When enabled, disables ALL HAPPO-specific rewards and uses original scales
        if baseline_mappo_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.baseline_mappo_rewards = True
            # Override scales to original MAPush values
            cfg.rewards.scales.target_reward_scale = 0.00325
            cfg.rewards.scales.approach_reward_scale = 0.00075
            cfg.rewards.scales.collision_punishment_scale = -0.0025
            cfg.rewards.scales.push_reward_scale = 0.0015
            cfg.rewards.scales.ocb_reward_scale = 0.004  # Original was 0.004, not 0.01
            cfg.rewards.scales.reach_target_reward_scale = 10
            cfg.rewards.scales.exception_punishment_scale = -5
            # Disable HAPPO-specific reward scales
            if hasattr(cfg.rewards.scales, 'proximity_penalty_scale'):
                cfg.rewards.scales.proximity_penalty_scale = 0.0
            print(f"[custom_cfg] BASELINE MAPPO REWARDS MODE: Using original 7 rewards with original scales")

        # Enable individualized rewards for HAPPO if requested
        if individualized_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.individualized_rewards = True

        # Iter8: Enable gated shared rewards
        if shared_gated_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.shared_gated_rewards = True

        # CRITIC12: Enable three-tier cooperation bonuses
        if cooperation_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.cooperation_rewards = True

        # Original MAPush rewards (teamified) - 7 original rewards converted to team rewards
        if mapush_og_rewards_teamified and hasattr(cfg, 'rewards'):
            cfg.rewards.mapush_og_rewards_teamified = True

        # CRITIC15 v3: Reward scale testing (currently unused)
        if reward_scale_testing and hasattr(cfg, 'rewards'):
            cfg.rewards.reward_scale_testing = True

        # CRITIC15 v4: Collaboration rewards - dual pushing bonus
        if collaboration_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.collaboration_rewards = True

        # CRITIC17: Positive approach_to_box reward (inverse distance instead of quadratic penalty)
        if positive_approachtobox_reward and hasattr(cfg, 'rewards'):
            cfg.rewards.positive_approachtobox_reward = True

        return cfg

    return fn

