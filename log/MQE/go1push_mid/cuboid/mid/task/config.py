import numpy as np
from mqe.utils.helpers import merge_dict
from mqe.envs.go1.go1 import Go1Cfg

class Go1PushMidCfg(Go1Cfg):

    class env(Go1Cfg.env):
        env_name = "go1push"
        num_envs = 10 # the number of environments, which is overriden, so it is not used
        num_agents = 2 
        num_npcs = 2 # object + target
        episode_length_s = 20  # the length of the episode in seconds
    
    # config of the object
    class asset(Go1Cfg.asset):
        # physical box
        terminate_after_contacts_on = []
        file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/cuboid/SmallBox.urdf"
        vertex_list = [[-0.60, -0.60], [ 0.60,-0.60],
                       [ 0.60,  0.60], [-0.60, 0.60]]
        name_npc = "box"
        npc_collision = True
        fix_npc_base_link = False
        npc_gravity = True
        # target area
        _terminate_after_contacts_on = []
        _file_npc = "{LEGGED_GYM_ROOT_DIR}/resources/objects/target.urdf"
        _name_npc = "target_area"
        _npc_collision = False
        _fix_npc_base_link = True
        _npc_gravity = True

    # config of the terrain
    class terrain(Go1Cfg.terrain):

        num_rows = 1
        num_cols = 1

        map_size = [24.0, 24.0] # the size of the map
        BarrierTrack_kwargs = merge_dict(Go1Cfg.terrain.BarrierTrack_kwargs, dict(
            options = [
                "init",
                "plane",
                "wall",
            ],
            track_width = map_size[1],
            init = dict(
                block_length = 0.1,
                room_size = (0.1, map_size[1]),
                border_width = 0.0,
                offset = (0, 0),
            ),
            plane = dict(
                block_length = map_size[0],
            ),
            wall = dict(
                block_length = 0.1
            ),
            wall_height= 0.5,
            virtual_terrain = False, 
            no_perlin_threshold = 0.06,
            add_perlin_noise = False,
            static_friction = 0.1, # static friction coefficient of the terrain, which will be overriden if friction_range is given in domain_rand
            dynamic_friction = 0.1, # dynamic friction coefficient of the terrain, which will be overriden if friction_range is given in domain_rand
       ))
    
    # velocity control
    class command(Go1Cfg.command):

        class cfg(Go1Cfg.command.cfg):
            vel = True         # lin_vel, ang_vel

    # velocity control
    class control(Go1Cfg.control):
        control_type = 'C'

    # termination conditions
    class termination(Go1Cfg.termination):
        # additional exceptions that determines whether to terminates the episode
        check_obstacle_conditioned_threshold = False
        z_wave_kwargs = dict(threshold= 0.35) # if the change of z position is larger than this threshold, the episode will be terminated
        collision_kwargs = dict(threshold= 0.25) # if the distance of two agents are small than this threshold, the episode will be terminated
        termination_terms = [
            "roll",
            "pitch",
            "z_wave",
            "collision",
        #    "far_away"
        ]

    # viewer setting
    class viewer(Go1Cfg.viewer):
        pos = [12., 12., 10.]  # [m]
        lookat = [13., 12., 0.]  # [m]

    # rewards weight setting
    class rewards(Go1Cfg.rewards):
        expanded_ocb_reward = False # if True, the reward will be given based on Circular Arc Interpolation Trajectory
        class scales:
            target_reward_scale = 0.00325                
            approach_reward_scale = 0.00075
            collision_punishment_scale = -0.0025
            push_reward_scale = 0.0015
            ocb_reward_scale = 0.004
            reach_target_reward_scale = 10
            exception_punishment_scale = -5

    # goal setting
    class goal:
        # static goal pos
        static_goal_pos = False
        goal_pos = [ 12.1, 0.0, 0.1]
        goal_rpy = [  0.0, 0.0, 0.0]
        # random goal pos
        random_goal_pos = True
        random_goal_distance_from_init = [1.5 , 3.0]                    # target_pos_randomlization
        random_goal_theta_from_init = [0, 2 * np.pi] # [min, max]       # target_theta_randomlization
        random_goal_rpy_range = dict(                                   # target_yaw_randomlization
                                r= [-0.01, 0.01],
                                p= [-0.01, 0.01],
                                y= [-0.01, 0.01],
                            )
        # receive goal pos from the user or high layer
        received_goal_pos = False
        received_final_pos = [ 9.0, 0.0, 0.1]
        # use only for test mode(middle layer sequential task)
        sequential_goal_pos = False
        goal_poses = [
            [ 3.0, 0.0, 0.1],
            [ 4.0, 0.0, 0.1],
            [ 5.0, 0.0, 0.1],
            [ 6.0, 0.0, 0.1],
            [ 7.0, 0.0, 0.1]]
        general_dist = False   # if True, orientation of the goal
        yaw_active = True      # if True, the general_dist will calculated based on the yaw angle
        THRESHOLD = 1.0        # the threshold of completion
        # check the goal setting
        check_setting = [static_goal_pos, random_goal_pos, received_goal_pos, sequential_goal_pos]
        if check_setting.count(True) != 1:
            raise ValueError("Only one of static_goal_pos, random_goal_pos, received_goal_pos, sequential_goal_pos can be True")


    # init state of robot sysytem
    class init_state(Go1Cfg.init_state):
        multi_init_state = True
        init_state_class = Go1Cfg.init_state
        # initial state of the robot, which will be overriden if random_base_init_state is True
        # make sure len(init_states) == num_agents
        init_states = [
            init_state_class(
                pos = [11.0,-1.0, 0.45],
                rot =  [0., 0., 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],
            ),
            init_state_class(
                pos = [11.0, 1.0, 0.45],
                rot =  [0., 0., 0.0, 1.0],
                lin_vel = [0.0, 0.0, 0.0],
                ang_vel = [0.0, 0.0, 0.0],      
            ),

        ] 
        # initial state of the npc, which will be overriden if random_npc_pos_range or init_npc_rpy_range is given in domain_rand
        # make sure len(init_states_npc) == num_npcs
        init_states_npc = [
            # physical box 
            init_state_class(
                pos = [12.0, 0.0, 0.30],                  
                rot = [0.0, 0.0, 0.0, 1.0],
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
        ]

    # domain randomization
    class domain_rand(Go1Cfg.domain_rand):
        # use for non-virtual training（add-noise）
        push_robots = False 
        random_base_init_state = True                                    
        init_base_pos_range = dict(                                       # agent_pos_randomlization
            r= [1.2, 1.3],
            theta=[-0.01, 2 * np.pi],
        )
        init_base_rpy_range = dict(                                       # agent_yaw_randomlization
            r= [-0.01, 0.01],
            p= [-0.01, 0.01],
            y= [-0.01, 2 * np.pi],
        )
        # init_npc_pos_range = dict(                                      # box_pos_randomlization
        #     x= [-0.1, 0.1],
        #     y= [-0.1, 0.1],
        # )
        init_npc_rpy_range = dict(                                        # box_yaw_randomlization
            r= [-0.01, 0.01],
            p= [-0.01, 0.01],
            y= [-0.01, 2 * np.pi],
        )
        friction_range = [0.5, 0.6]                                       # friction_randomlization

