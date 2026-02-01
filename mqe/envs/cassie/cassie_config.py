# Cassie Biped Robot Configuration for MAPush
#
# CRITICAL: All parameters MUST match the legged_gym training config
# Training command: python legged_gym/scripts/train.py --task=cassie_flat --headless
# Training config: legged_gym/legged_gym/envs/cassie/cassie_flat_config.py
#
# Key differences from quadrupeds:
# - 12 DOFs (6 per leg): hip_abduction, hip_rotation, hip_flexion, thigh_joint, ankle_joint, toe_joint
# - Biped requires different spawn height and stability considerations
# - Uses standard legged_gym observation structure (48 dims for flat terrain)

from mqe.envs.base.legged_robot_config import LeggedRobotCfg
from mqe.envs.field.legged_robot_field_config import LeggedRobotFieldCfg


class CassieCfg(LeggedRobotFieldCfg):

    class env(LeggedRobotFieldCfg.env):
        use_lin_vel = True
        num_envs = 256
        num_observations = 235  # Will be computed by MAPush
        use_lin_vel = True
        num_privileged_obs = None
        num_actions = 12  # 6 DOF per leg x 2 legs
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 5

        # Recording cfgs
        record_video = True
        record_actor_id = 0
        recording_width_px = 360 * 3
        recording_height_px = 360 * 3
        recording_mode = "COLOR"

    class asset:
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf"
        files = [
            "{LEGGED_GYM_ROOT_DIR}/resources/robots/cassie/urdf/cassie.urdf",
        ]
        name = "cassie"
        foot_name = "toe"  # Cassie uses 'toe' as foot contact point
        penalize_contacts_on = ["pelvis"]  # Terminate if pelvis contacts ground
        terminate_after_contacts_on = ["pelvis"]
        disable_gravity = False
        collapse_fixed_joints = True
        fix_base_link = False
        default_dof_drive_mode = 3  # Effort mode for actuator network control
        self_collisions = 1  # 1 to disable - matches training config
        replace_cylinder_with_capsule = True
        flip_visual_attachments = False  # MUST match training config (False for Cassie)

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class init_state(LeggedRobotFieldCfg.init_state):
        # Cassie is taller than quadrupeds - spawn at 1.0m
        pos = [0.0, 0.0, 1.0]  # x,y,z [m] - MUST match training config

        # Default joint angles - MUST match training config EXACTLY
        # From legged_gym/legged_gym/envs/cassie/cassie_config.py
        default_joint_angles = {
            # Left leg
            'hip_abduction_left': 0.1,
            'hip_rotation_left': 0.,
            'hip_flexion_left': 1.,
            'thigh_joint_left': -1.8,
            'ankle_joint_left': 1.57,
            'toe_joint_left': -1.57,
            # Right leg
            'hip_abduction_right': -0.1,
            'hip_rotation_right': 0.,
            'hip_flexion_right': 1.,
            'thigh_joint_right': -1.8,
            'ankle_joint_right': 1.57,
            'toe_joint_right': -1.57,
        }

    class normalization(LeggedRobotFieldCfg.normalization):
        clip_actions = 10.

    class control(LeggedRobotFieldCfg.control):
        control_type = 'C'  # Command control (hierarchical with locomotion policy)

        # PD gains - MUST match training config from cassie_config.py
        # Pattern matching for joint names
        stiffness = {
            'hip_abduction': 100.0,
            'hip_rotation': 100.0,
            'hip_flexion': 200.,
            'thigh_joint': 200.,
            'ankle_joint': 200.,
            'toe_joint': 40.
        }
        damping = {
            'hip_abduction': 3.0,
            'hip_rotation': 3.0,
            'hip_flexion': 6.,
            'thigh_joint': 6.,
            'ankle_joint': 6.,
            'toe_joint': 1.
        }

        # Action scale - MUST match training config
        action_scale = 0.5

        # Torque limits per joint type (repeated for both legs)
        # Order: hip_abd, hip_rot, hip_flex, thigh, ankle, toe (x2 for left/right)
        torque_limits = [100., 100., 200., 200., 200., 40.] * 2

        computer_clip_torque = True
        motor_clip_torque = False
        decimation = 4  # MUST match training config

        # Locomotion policy path (to be populated after training)
        locomotion_policy_dir = "./resources/robots/cassie/policy"
        # Cassie may use PD control without actuator network, or could use one
        # Set to None if not using actuator network
        actuator_network_path = None

        class default_command:
            lin_vel_x = 0.5
            lin_vel_y = 0.0
            ang_vel = 0.0
            body_height = 0.0
            gait_freq = 2.0
            gait = "trotting"
            footswing_height = 0.08
            body_pitch = 0.0
            body_roll = 0.0
            stance_width = 0.25
            stance_length = 0.4
            aux_reward = 0.0

        class obs_scales:
            # Standard legged_gym observation scales
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            body_height = 2.0
            gait_phase = 1.0
            gait_freq = 1.0
            footswing_height = 0.15
            body_pitch = 0.3
            body_roll = 0.3
            aux_reward = 1.0
            compliance = 1.0
            stance_width = 1.0
            stance_length = 1.0

    class command:
        gaits = {
            "pronking": [0, 0, 0],
            "trotting": [0.5, 0, 0],
            "bounding": [0, 0.5, 0],
            "pacing": [0, 0, 0.5]
        }

        curriculum = False
        max_curriculum = 1.
        num_commands = 4
        resampling_time = 10.
        heading_command = True

        class cfg:
            vel = False
            body_height = False
            body_pose = False
            gait_freq = False
            gait = False
            footswing_height = False
            stance_width = False
            stance_length = False
            aux_reward = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # m/s
            lin_vel_y = [-0.5, 0.5]  # m/s - reduced for biped stability
            ang_vel_yaw = [-1.0, 1.0]  # rad/s
            heading = [-3.14, 3.14]

    class termination:
        termination_terms = [
            "roll",
            "pitch",
            "z_wave",
            "collision",
        ]

        roll_kwargs = dict(
            threshold=0.8,  # [rad] - bipeds are less stable
        )
        pitch_kwargs = dict(
            threshold=1.0,  # [rad] - tighter constraint for biped
        )
        z_wave_kwargs = dict(
            threshold=1.5,  # [m] - adjusted for Cassie's height
        )
        collision_kwargs = dict(
            threshold=0.15,  # [m]
        )
        far_away_kwargs = dict(
            threshold_agent=3.,  # [m]
            threshold_box=5.,
        )

    class domain_rand(LeggedRobotFieldCfg.domain_rand):
        randomize_com = False
        class com_range:
            x = [-0.05, 0.15]
            y = [-0.1, 0.1]
            z = [-0.05, 0.05]

        randomize_motor = False
        leg_motor_strength_range = [0.9, 1.1]

        randomize_base_mass = False
        added_mass_range = [-1.0, 3.0]

        randomize_friction = False
        friction_range = [0.05, 4.5]

        randomize_lag_timesteps = False
        lag_timesteps = 6

        init_base_pos_range = dict(
            x=[0.1, 0.1],
            y=[-0.1, 0.1],
        )

        init_dof_pos_ratio_range = None

        init_npc_base_pos_range = dict(
            x=[-0.2, 0.2],
            y=[-0.2, 0.2],
        )

        push_robots = False

    class obs:
        class cfgs:
            base_pos = True
            base_quat = True
            dof_pos = True
            dof_vel = True
            lin_vel = True
            ang_vel = True
            projected_gravity = True
            base_rpy = True
            contact_states = False
            command = True
            height_command = False
            gait_commands = False
            timing_parameter = False
            clock_inputs = False
            last_action = True
            last_last_action = True
            imu = False
            depth_image = False
            rgb_image = False
            env_info = True

        class scales:
            base_pos = 1.0
            base_quat = 1.0
            segmentation_image = 1.0
            rgb_image = 1.0
            depth_image = 1.0

        def keys(self):
            key_dict = dir(self.cfgs)
            key_list = []
            for key in key_dict:
                if getattr(self.cfgs, key) == True and key:
                    key_list.append(key)
            return key_list

    class privileged_obs:
        class cfgs:
            pass

        def keys(self):
            key_dict = dir(self.cfgs)
            print("===Available Observations===")
            for key in key_dict:
                if getattr(self.cfgs, key) == True and key:
                    print(key, end=" ")
            print("\n===========================")

    class rewards(LeggedRobotFieldCfg.rewards):
        soft_dof_pos_limit = 0.95
        base_height_target = 0.9  # Cassie is tall
        class scales(LeggedRobotFieldCfg.rewards.scales):
            torques = -0.0002
            dof_pos_limits = -10.0

    class viewer(LeggedRobotFieldCfg.viewer):
        pos = [0., 11., 5.]  # [m]
        lookat = [4., 11., 0.]  # [m]
