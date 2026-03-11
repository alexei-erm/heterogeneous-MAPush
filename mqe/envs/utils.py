# environments
from mqe.envs.field.legged_robot_field import LeggedRobotField
from mqe.envs.go1.go1 import Go1
from mqe.envs.npc.go1_object import Go1Object

# configs
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg
from mqe.envs.configs.go1_push_mid_config import Go1PushMidCfg
from mqe.envs.configs.go1_push_upper_config import Go1PushUpperCfg
from mqe.envs.configs.go1_push_vel_config import Go1PushVelCfg


# wrappers
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from mqe.envs.wrappers.go1_push_mid_wrapper import Go1PushMidWrapper
from mqe.envs.wrappers.go1_push_upper_wrapper import Go1PushUpperWrapper
from mqe.envs.wrappers.go1_push_vel_wrapper import Go1PushVelWrapper

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
    "go1push_vel": {
        "class": Go1Object,
        "config": Go1PushVelCfg,
        "wrapper": Go1PushVelWrapper
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

def custom_cfg(args, individualized_rewards=False, shared_gated_rewards=False, cooperation_rewards=False, mapush_og_rewards_teamified=False, reward_scale_testing=False, collaboration_rewards=False, positive_approachtobox_reward=False, agent0='go1', agent1='go1', baseline_mappo_rewards=False, mappo_heavybox_rewards=False, require_both_contact_for_success=False, contact_force_gating=False, contact_force_gating_alpha=0.3, vel_speed_min=None, vel_speed_max=None, vel_tracking_scale=None, vel_angular_penalty_scale=None, box_mass=None):

    def fn(cfg:LeggedRobotFieldCfg):

        if getattr(args, "num_envs", None) is not None:
            cfg.env.num_envs = args.num_envs

        cfg.env.record_video = args.record_video

        # Enable heterogeneous agents if any agent is non-Go1
        # (even same-type pairs like anymal_c+anymal_c need the hetero config path)
        is_hetero = (agent0 != agent1) or (agent0 != 'go1')
        if is_hetero and hasattr(cfg, 'hetero'):
            cfg.hetero.use_hetero = True
            cfg.hetero.hetero_agent_types = [agent0, agent1]
            print(f"[custom_cfg] Enabled hetero mode: agent0={agent0}, agent1={agent1}")

        # MAPPO BASELINE MODE: Use TRUE ORIGINAL MAPush rewards only
        # When enabled, disables ALL HAPPO-specific rewards and uses ORIGINAL scales
        # This is the actual baseline from the original MAPush paper/repo
        if baseline_mappo_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.baseline_mappo_rewards = True
            # TRUE ORIGINAL baseline values from MAPush
            cfg.rewards.scales.target_reward_scale = 0.00325     # original
            cfg.rewards.scales.approach_reward_scale = 0.00075   # original
            cfg.rewards.scales.collision_punishment_scale = -0.0025  # original
            cfg.rewards.scales.push_reward_scale = 0.0015        # original
            cfg.rewards.scales.ocb_reward_scale = 0.004          # original
            cfg.rewards.scales.reach_target_reward_scale = 10    # original
            cfg.rewards.scales.exception_punishment_scale = -5   # original
            # Disable HAPPO-specific reward scales
            if hasattr(cfg.rewards.scales, 'proximity_penalty_scale'):
                cfg.rewards.scales.proximity_penalty_scale = 0.0
            print(f"[custom_cfg] BASELINE MAPPO REWARDS MODE: Using TRUE ORIGINAL 7 rewards with original scales")
            print(f"  target=0.00325, approach=0.00075, collision=-0.0025, push=0.0015, ocb=0.004, reach=10, exception=-5")

        # MAPPO HEAVY BOX MODE: Adjusted scales for 8kg box (from heavy_cuboid_testing_mappo.md Exp 7)
        # Use this when training with heavier box that requires collaboration
        # Key changes: 3x target, 2.5x push, 2x OCB, distance penalty term REMOVED
        if mappo_heavybox_rewards and hasattr(cfg, 'rewards'):
            cfg.rewards.mappo_heavybox_rewards = True
            # Exp 7 scales from heavy_cuboid_testing_mappo.md
            cfg.rewards.scales.target_reward_scale = 0.01        # 3x increase (penalty term removed in wrapper)
            cfg.rewards.scales.approach_reward_scale = 0.00075   # original
            cfg.rewards.scales.collision_punishment_scale = -0.0025  # original
            cfg.rewards.scales.push_reward_scale = 0.004         # 2.5x increase
            cfg.rewards.scales.ocb_reward_scale = 0.008          # 2x increase
            cfg.rewards.scales.reach_target_reward_scale = 10    # original
            cfg.rewards.scales.exception_punishment_scale = -5   # original
            # Disable HAPPO-specific reward scales
            if hasattr(cfg.rewards.scales, 'proximity_penalty_scale'):
                cfg.rewards.scales.proximity_penalty_scale = 0.0
            print(f"[custom_cfg] MAPPO HEAVY BOX REWARDS MODE: Using Exp 7 scales for 8kg box")
            print(f"  target=0.01 (3x), push=0.004 (2.5x), ocb=0.008 (2x), distance penalty REMOVED")

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

        # COLLABORATION ENFORCEMENT: Only give reach_target_reward if BOTH agents are in contact
        if require_both_contact_for_success and hasattr(cfg, 'rewards'):
            cfg.rewards.require_both_contact_for_success = True

        # Contact-force gating: gate push_reward and target_reward by contact-force balance
        if contact_force_gating and hasattr(cfg, 'rewards'):
            cfg.rewards.contact_force_gating = True
            cfg.rewards.contact_force_gating_alpha = contact_force_gating_alpha
            print(f"[custom_cfg] Contact-force gating ENABLED: alpha={contact_force_gating_alpha}")

        # Box mass override
        if box_mass is not None and hasattr(cfg, 'asset'):
            cfg.asset.npc_mass_override = box_mass
            print(f"[custom_cfg] Box mass override: {box_mass} kg")

        # Velocity task overrides
        if hasattr(cfg, 'velocity_command'):
            if vel_speed_min is not None:
                cfg.velocity_command.speed_range[0] = vel_speed_min
            if vel_speed_max is not None:
                cfg.velocity_command.speed_range[1] = vel_speed_max
            if vel_tracking_scale is not None and hasattr(cfg, 'rewards'):
                cfg.rewards.scales.velocity_tracking_scale = vel_tracking_scale
            if vel_angular_penalty_scale is not None and hasattr(cfg, 'rewards'):
                cfg.rewards.scales.angular_velocity_penalty_scale = vel_angular_penalty_scale

        return cfg

    return fn

