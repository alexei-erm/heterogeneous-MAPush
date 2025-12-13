import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1PushUpperCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1push"
        num_envs = 10
        num_agents = 2
        num_obs = 2
        num_npcs = 3 + num_obs
        episode_length_s = 160
        hie = True
        record_video = True

    # config of the robot 
    class asset(Go1Cfg.asset):
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/MiddleBox.urdf"
        vertex_list = [[-0.50,-0.25],[ 0.50,-0.25],
                       [ 0.25, 0.75],[-0.25, 0.75],]
        name_npc = "box"
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
        # target area
        _terminate_after_contacts_on = []
        _file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder.urdf"
        _name_npc = "target_area"
        _npc_collision = False
        _fix_npc_base_link = True
        _npc_gravity = True
        # final area
        terminate_after_contacts_on_final = []
        file_npc_final = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cylinder_blue.urdf"
        name_npc_final = "final_target"
        npc_collision_final = False
        fix_npc_base_link_final = True
        npc_gravity_final = True
        # obstacle
        obs_terminate_after_contacts_on = []
        obs_file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/obstacle.urdf"
        obs_name_npc = "target_area"
        obs_npc_collision = True
        obs_fix_npc_base_link = True
        obs_npc_gravity = True

    # config of the terrain
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "wall",
            ],
            # wall_thickness= 0.2,
            track_width = 15.0,
            init = dict(
                block_length = 1.0,
                room_size = (1.0, 7.5),
                border_width = 0.0,
                offset = (0, 0),
            ),
            plane = dict(
                block_length = 13.0,
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, # Change this to False for real terrain
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
       ))
    
    # velocity control
    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    # velocity control
    class control(Go1Cfg.control):
        control_type = 'C'
        command_network_path = "./resources/command_nets/1.2x1.2.pt"

    # termination conditions
    class termination(Go1Cfg.termination):
        # additional factors that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        termination_terms = [
        #    "roll",
        #    "pitch",
        #    "z_wave",
        #    "collision",
        #    "far_away"
        ]

    # viewer setting
    class viewer(Go1Cfg.viewer):
        pos = [7., 7.5, 10.]  # [m]
        lookat = [8., 7.5, 0.]  # [m]

    # rewards weight setting
    class rewards(Go1Cfg.rewards):
        class scales:
            target_reward_scale = 0.3
            trajectory_rewards_scale = 0.5
            reach_target_reward_scale = 2
            exception_punishment_scale = -0.5
            obstacle_reward_scale = -0.1

    # goal setting
    class goal:
        # static goal pos
        static_goal_pos = False
        goal_pos = [ 5.0,-3.0, 0.1]
        # random goal pos
        random_goal_pos = False
        random_goal_distance_from_init = [1.5 , 3.0]                                  # target_pos_randomlization
        random_goal_theta_from_init = [-0.5 , 0.5] # [min, max]                     # target_theta_randomlization
        # receive goal pos from the user or high layer
        received_goal_pos = True
        received_final_pos = [ 9.0, 0.0, 0.1]
        # use only for test mode(middle layer sequential task)
        sequential_goal_pos = False
        goal_poses = [
            [ 3.0, 0.0, 0.1],
            [ 4.0, 0.0, 0.1],
            [ 5.0, 0.0, 0.1],
            [ 6.0, 0.0, 0.1],
            [ 7.0, 0.0, 0.1]]
        THRESHOLD=1.0
        # check the goal setting
        check_setting = [static_goal_pos, random_goal_pos, received_goal_pos, sequential_goal_pos]
        if check_setting.count(True) != 1:
            raise ValueError("Only one of static_goal_pos, random_goal_pos, received_goal_pos, sequential_goal_pos can be True")

    class obstacle_state:
        # static goal pos
        static_obs_pos = False
        init_state_class = Go1Cfg.init_state
        states_obs = [
            init_state_class(
                pos = [ 4.0, 1.0, 0.1],
                rot = [ 0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [ 6.5, -2, 0.1],
                rot = [ 0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        # random obs pos
        random_obs_pos = True
        random_obs_x_range = [ 3.0, 14.0]                                  
        random_obs_y_range = [-7.5, 7.5]
        random_obs_rpy_range = dict(                                     # box_yaw_randomlization
                        r= [-0.01, 0.01],
                        p= [-0.01, 0.01],
                        y= [-0.01, 2 * np.pi],
                    )  
        obs1_pos = [0.0, 0.0, 0.0]
        obs2_pos = [0.0, 0.0, 0.0]
        check_setting = [static_obs_pos, random_obs_pos]
        if check_setting.count(True) != 1:
            raise ValueError("Only one of static_obs_pos, random_obs_pos can be True")

    # init state of robot sysytem
    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        init_states = [
            init_state_class(
                pos = [0.5, 0.45, 0.4],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [0.5, -0.45, 0.4],
                rot = [0.0, 0.0, 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]
        init_states_npc = [
            # physical box 
            init_state_class(
                pos = [1.6, 0.0, 0.55],                                          # box_pos_origin
                # rot = [0.0, 0.0, 0.0, 1.0],
                rot = [0.0, 0.0, 0.3826834, 0.9238795],
                # rot = [0.0, 0, 0.3827, 0.9238],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            # target area
            init_state_class(
                pos = [8.0, 0.0, 0.1],
                rot = [0.0, 0.0, 0.0, 1.0],
                # rot = [0.0, 0, 0.3827, 0.9238],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            # final target
            init_state_class(
                pos = [8.5, 1.0, 0.1],
                rot = [0.0, 0.0, 0.0, 1.0],
                # rot = [0.0, 0, 0.3827, 0.9238],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
        ]

    # domain randomization
    class domain_rand(Go1Cfg.domain_rand):
        # use for non-virtual training（add-noise）
        push_robots = False 
        random_base_init_state = False                                      # agent_pos_randomlization
        init_base_pos_range = dict(                                         
            r= [1.2, 1.3],
            theta=[-0.01, 2 * np.pi],
        )
        init_base_rpy_range = dict(                                       # agent_yaw_randomlization
            r= [-0.01, 0.01],
            p= [-0.01, 0.01],
            y= [-0.01, 2 * np.pi],
        )
        init_base_tiny_pos_range = dict(                                     # agent_pos_randomlization
            x= [-0.05, 0.05],
            y= [-0.03, 0.03],
        )
        init_npc_base_pos_range = dict(                                     # box_pos_randomlization
            x= [ 1.4, 1.6],
            y= [-0.1, 0.1],
        )
        init_npc_base_rpy_range = dict(                                     # box_yaw_randomlization
            r= [-0.01, 0.01],
            p= [-0.01, 0.01],
            y= [-np.pi/2-0.10,-np.pi/2+0.10],
        )
        friction_raige = [0.6, 0.7]

    class generalize_obsersation:
        rotate_obs = True 
        net_origin = [ 5, 0, 0]

    class test_metrics:
        collision_threshold = 1.8
        collaboration_threshold = 1.45