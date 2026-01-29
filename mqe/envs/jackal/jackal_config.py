"""
Jackal Robot Configuration

Configuration for Clearpath Robotics Jackal differential drive robot.
High-level action space: [vx, vy, vyaw] (same as Go1)
Low-level differential drive controller converts to wheel velocities

Author: Claude
Date: 2026-01-15
"""

from mqe.envs.base.legged_robot_config import LeggedRobotCfg
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg


class JackalCfg(LeggedRobotFieldCfg):
    """Configuration for Jackal wheeled robot."""

    class env(LeggedRobotFieldCfg.env):
        use_lin_vel = True
        num_envs = 256
        num_observations = 235  # Will match Go1 for now
        num_privileged_obs = None
        num_actions = 3  # [vx, vy, vyaw] - same as Go1 for unified action space
        env_spacing = 3.0
        send_timeouts = True
        episode_length_s = 5

        # Recording configs
        record_video = True
        record_actor_id = 0
        recording_width_px = 360 * 3
        recording_height_px = 360 * 3
        recording_mode = "COLOR"

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/jackal/urdf/jackal.urdf"
        name = "jackal"
        foot_name = "caster"  # Jackal has caster wheels, not feet
        penalize_contacts_on = ["chassis"]
        terminate_after_contacts_on = []  # Jackal is robust, no termination on contact
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 1  # VELOCITY control (stable for wheeled robots!)
                                     # Was 3 (EFFORT/torque) which caused physics instability
        self_collisions = 0

        replace_cylinder_with_capsule = True
        flip_visual_attachments = False

        # Jackal physical properties
        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 20.0
        max_linear_velocity = 10.0
        armature = 0.0
        thickness = 0.01

    class terrain(LeggedRobotFieldCfg.terrain):
        mesh_type = 'plane'  # Jackal works best on flat terrain
        measure_heights = False
        static_friction = 0.6  # Rubber wheels on ground
        dynamic_friction = 0.5
        restitution = 0.0

    class init_state(LeggedRobotFieldCfg.init_state):
        # CRITICAL FIX: Proper ground clearance for wheeled robot
        # Wheel radius = 0.098m, wheel center 0.10m above base_link (0.0655+0.0345)
        # Wheel bottom = base_link + 0.10 - 0.098 = base_link + 0.002m
        # For wheels to touch ground (bottom at z=0): base_link_z = -0.002m ≈ 0.0m
        pos = [0.0, 0.0, 0.0]  # base_link at ground level (wheels touch ground)
        rot = [0.0, 0.0, 0.0, 1.0]  # Quaternion [x, y, z, w]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        # Jackal DOF initial states (2 wheels)
        default_joint_angles = {
            "left_wheel_joint": 0.0,
            "right_wheel_joint": 0.0,
        }

    class control(LeggedRobotFieldCfg.control):
        # Direct velocity control (no hierarchical policy)
        control_type = 'P'  # Position/velocity control (not 'C' hierarchical)

        # PD gains for wheel velocity control
        stiffness = {'wheel': 0.0}  # Velocity control doesn't need stiffness
        damping = {'wheel': 1.0}    # Damping for smooth velocity tracking

        # Action scale
        action_scale = 10.0  # Scale actions to rad/s

        # Limits for wheel velocities
        clip_actions = 1.0  # Clip to [-1, 1] before scaling

    class normalization(LeggedRobotFieldCfg.normalization):
        clip_observations = 100.0
        clip_actions = 1.0

        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05

    class noise(LeggedRobotFieldCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05

    class domain_rand(LeggedRobotFieldCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.4, 0.8]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]  # Jackal can carry payloads
        push_robots = False
        push_interval_s = 4
        max_push_vel_xy = 0.5

    class commands(LeggedRobotFieldCfg.commands):
        """Not used for MAPush (environment provides commands)"""
        pass

    class rewards(LeggedRobotFieldCfg.rewards):
        """Jackal-specific rewards (if needed)"""
        pass
