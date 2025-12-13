from collections import OrderedDict, defaultdict
import itertools
import numpy as np
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz, quat_from_euler_xyz, tf_apply
from isaacgym import gymtorch, gymapi, gymutil
from copy import deepcopy
import torch

from mqe.envs.base.legged_robot import LeggedRobot
from mqe.utils.terrain import get_terrain_cls
from ..base.legged_robot_config import LeggedRobotCfg
from ..go1.go1_config import Go1Cfg

class LeggedRobotField(LeggedRobot):
    """ NOTE: Most of this class implementation does not depend on the terrain. Check where
    `check_BarrierTrack_terrain` is called to remove the dependency of BarrierTrack terrain.
    """
    def __init__(self, cfg: Go1Cfg, sim_params, physics_engine, sim_device, headless):
        print("Using LeggedRobotField.__init__, num_obs and num_privileged_obs will be computed instead of assigned.")
        # cfg.terrain.measure_heights = True # force height measurement that have full obs from parent class implementation.
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

        # calculate metrics(only for testing--calculator)
        self.init_reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.init_finished_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.init_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.collision_degree_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.collaboration_degree_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.init_episode_length_buf = torch.tensor(self.max_episode_length, dtype=torch.long, device=self.device, requires_grad=False).repeat(self.num_envs)

        self.last_init_finished_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    ##### adds-on with sensors #####
    def _create_sensors(self, env_handle=None, actor_handle= None):
        sensor_handle_dict = super()._create_sensors(
            env_handle= env_handle,
            actor_handle= actor_handle,
        )
        if self.cfg.obs.cfgs.depth_image or self.cfg.obs.cfgs.rgb_image:
            camera_handle = self._create_onboard_camera(env_handle, actor_handle, "forward_camera")
            sensor_handle_dict["forward_camera"] = camera_handle

        return sensor_handle_dict

    def _create_onboard_camera(self, env_handle, actor_handle, sensor_name):
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.height = getattr(self.cfg.sensor, sensor_name).resolution[0]
        camera_props.width = getattr(self.cfg.sensor, sensor_name).resolution[1]
        if hasattr(getattr(self.cfg.sensor, sensor_name), "horizontal_fov"):
            camera_props.horizontal_fov = np.random.uniform(
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[0],
                getattr(self.cfg.sensor, sensor_name).horizontal_fov[1],
            ) if isinstance(getattr(self.cfg.sensor, sensor_name).horizontal_fov, (tuple, list)) else getattr(self.cfg.sensor, sensor_name).horizontal_fov
            # vertical_fov = horizontal_fov * camera_props.height / camera_props.width
        camera_handle = self.gym.create_camera_sensor(env_handle, camera_props)
        local_transform = gymapi.Transform()
        if isinstance(getattr(self.cfg.sensor, sensor_name).position, dict):
            # allow domain randomization across robots.
            # sample from "mean" and "std" attributes.
            # each must be a list of 3 elements.
            cam_x = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][0],
                getattr(self.cfg.sensor, sensor_name).position["std"][0],
            )
            cam_y = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][1],
                getattr(self.cfg.sensor, sensor_name).position["std"][1],
            )
            cam_z = np.random.normal(
                getattr(self.cfg.sensor, sensor_name).position["mean"][2],
                getattr(self.cfg.sensor, sensor_name).position["std"][2],
            )
            local_transform.p = gymapi.Vec3(cam_x, cam_y, cam_z)
        else:
            local_transform.p = gymapi.Vec3(*getattr(self.cfg.sensor, sensor_name).position)
        if isinstance(getattr(self.cfg.sensor, sensor_name).rotation, dict):
            # allow domain randomization across robots
            # sample from "lower" and "upper" attributes.
            # each must be a list of 3 elements (in radian).
            cam_roll = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][0] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][0]
            cam_pitch = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][1] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][1]
            cam_yaw = np.random.uniform(0, 1) * (
                getattr(self.cfg.sensor, sensor_name).rotation["upper"][2] - \
                getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            ) + getattr(self.cfg.sensor, sensor_name).rotation["lower"][2]
            local_transform.r = gymapi.Quat.from_euler_zyx(cam_yaw, cam_pitch, cam_roll)
        else:
            local_transform.r = gymapi.Quat.from_euler_zyx(*getattr(self.cfg.sensor, sensor_name).rotation)
        self.gym.attach_camera_to_body(
            camera_handle,
            env_handle,
            actor_handle,
            local_transform,
            gymapi.FOLLOW_TRANSFORM,
        )
        
        return camera_handle

    ##### Working on simulation steps #####
    def pre_physics_step(self, actions):
        actions_preprocessed = False
        if isinstance(self.cfg.normalization.clip_actions, (tuple, list)):
            self.cfg.normalization.clip_actions = torch.tensor(
                self.cfg.normalization.clip_actions,
                device= self.device,
            )
        if getattr(self.cfg.normalization, "clip_actions_method", None) == "tanh":
            clip_actions = self.cfg.normalization.clip_actions
            self.actions = (torch.tanh(actions) * clip_actions).to(self.device)
            actions_preprocessed = True
        if getattr(self.cfg.normalization, "clip_actions_delta", None) is not None:
            self.actions = torch.clip(
                self.actions,
                self.last_actions - self.cfg.normalization.clip_actions_delta,
                self.last_actions + self.cfg.normalization.clip_actions_delta,
            )
        
        if not actions_preprocessed:
            return super().pre_physics_step(actions)
    
    def post_physics_step(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        return super().post_physics_step()
    
    def check_termination(self):
        return_ = super().check_termination()
        if not hasattr(self.cfg, "termination"): return return_
        
        r, p, y = get_euler_xyz(self.base_quat)
        r[r > np.pi] -= np.pi * 2 # to range (-pi, pi)
        p[p > np.pi] -= np.pi * 2 # to range (-pi, pi)
        z = self.root_states[:, 2] - self.agent_origins.reshape(-1, 3)[:, 2]

        base_init_state = deepcopy(self.base_init_state[:, :3].reshape(self.num_envs,self.num_agents, -1))
        base_init_state_npc = deepcopy(self.base_init_state_npc[:, :3].reshape(self.num_envs,self.num_npcs, -1))
        base_state = (self.root_states[:, :3]-self.env_origins.repeat(self.num_agents, 1)).reshape(self.num_envs, self.num_agents, -1)
        base_state_npc = (self.root_states_npc[:, :3]-self.env_origins.repeat(self.num_npcs, 1)).reshape(self.num_envs, self.num_npcs, -1)

        self.exception_buf = torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)

        if "roll" in self.cfg.termination.termination_terms:
            self.r_term_buff = (torch.abs(r) > self.cfg.termination.roll_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
            self.exception_buf |= self.r_term_buff

        if "pitch" in self.cfg.termination.termination_terms:
            self.p_term_buff = (torch.abs(p) > self.cfg.termination.pitch_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
            self.exception_buf |= self.p_term_buff

        if "z_wave" in self.cfg.termination.termination_terms:
            # initialize buff
            self.z_wave_term_buff = torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)
            for i in range(self.num_agents):
                self.z_wave_term_buff |= (torch.abs(base_state[:,i,2]-base_init_state[:,i,2]) > self.cfg.termination.z_wave_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
            for j in range(self.num_npcs):
                self.z_wave_term_buff |= (torch.abs(base_state_npc[:,j,2]-base_init_state_npc[:,j,2]) > self.cfg.termination.z_wave_kwargs["threshold"]).reshape(self.num_envs, -1).sum(1).to(torch.bool)
            self.exception_buf |= self.z_wave_term_buff

        if "collision" in self.cfg.termination.termination_terms:
            self.collision_term_buff = torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)
            # if two agent in a same env, check collision between them
            for i, j in itertools.combinations(range(self.num_agents), 2):
                self.collision_term_buff |= torch.norm(base_state[:,i]-base_state[:,j], dim=1) < self.cfg.termination.collision_kwargs["threshold"]
            self.exception_buf |= self.collision_term_buff
        
        if "far_away" in self.cfg.termination.termination_terms:
            self.far_away_term_buff = torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)
            for i in range(self.num_agents):
                self.far_away_term_buff |= torch.norm(base_state[:,i]-base_state_npc[:,0], dim=1) > self.cfg.termination.far_away_kwargs["threshold_agent"]
            # target pos and box pos should not be too far away from each other
            self.far_away_term_buff |= torch.norm(base_state_npc[:,0]-base_state_npc[:,1], dim=1) > self.cfg.termination.far_away_kwargs["threshold_box"]
            self.exception_buf |= self.far_away_term_buff
        
        if "out_of_area" in self.cfg.termination.termination_terms:
            self.out_of_area_term_buff = torch.zeros(self.num_envs, dtype= torch.bool, device= self.device)
            # transfer agent_state to box coordinate
            box_pos = base_state_npc[:,0,:2]
            box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
            box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
            agent_pos_relative_to_box = base_state[:,:,:2] - box_pos.unsqueeze(1).repeat(1,self.num_agents,1)
            # rotate agent_pos_relative_to_box to box coordinate
            agent_pos_relative_to_box = torch.stack([agent_pos_relative_to_box[:,:,0]*torch.cos(box_rpy[:,2].unsqueeze(1)) + agent_pos_relative_to_box[:,:,1]*torch.sin(box_rpy[:,2].unsqueeze(1)),
                                                     -agent_pos_relative_to_box[:,:,0]*torch.sin(box_rpy[:,2].unsqueeze(1)) + agent_pos_relative_to_box[:,:,1]*torch.cos(box_rpy[:,2].unsqueeze(1))], dim=2)
            agent_x_relative_to_box = agent_pos_relative_to_box[:,:,0]
            agent_y_relative_to_box = agent_pos_relative_to_box[:,:,1]
            # check if agent is out of area
            self.out_of_area_term_buff |= torch.sum(agent_x_relative_to_box < self.cfg.termination.out_of_area_kwargs["threshold_x"][0], dim=1).to(torch.bool)
            self.out_of_area_term_buff |= torch.sum(agent_x_relative_to_box > self.cfg.termination.out_of_area_kwargs["threshold_x"][1], dim=1).to(torch.bool)  
            self.out_of_area_term_buff |= torch.sum(agent_y_relative_to_box < self.cfg.termination.out_of_area_kwargs["threshold_y"][0], dim=1).to(torch.bool)
            self.out_of_area_term_buff |= torch.sum(agent_y_relative_to_box > self.cfg.termination.out_of_area_kwargs["threshold_y"][1], dim=1).to(torch.bool)
            self.exception_buf |= self.out_of_area_term_buff


            
        self.reset_buf |= self.exception_buf

        # calc success rate and finished time
        self.init_finished_buf |= ( ~ self.init_reset_buf & self.finished_buf)
        recent_success = self.init_finished_buf & ~ self.last_init_finished_buf
        self.init_episode_length_buf[recent_success] = self.episode_length_buf[recent_success]
        

        self.last_init_finished_buf = deepcopy(self.init_finished_buf)

        # calc collaboration degree and collision degree
        obs_pos  = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:,2:2+self.num_obs,:2]
        box_pos  = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:,0,:2]
        base_pos = self.root_states.reshape(self.num_envs, self.num_agents, -1)[:,:,:2]

        if self.num_obs != 0:
            collision_threshold = getattr(self.cfg.goal, "collision_threshold", 1.0)
            collision_dist,_ = torch.min(torch.norm(obs_pos - box_pos.unsqueeze(1).repeat(1,self.num_obs,1),dim=2),dim=1)
            collision_check = collision_dist < collision_threshold
            collision_buf = collision_check & ~self.init_reset_buf
            self.collision_degree_buf[collision_buf] +=1 

        collaboration_threshold = getattr(self.cfg.goal, "collaboration_threshold", 1.0)
        dist_to_box = torch.norm(base_pos - box_pos.unsqueeze(1).repeat(1,self.num_agents,1),dim=2) < collaboration_threshold
        collaboration_check = torch.sum(dist_to_box,dim=1) >= self.num_agents
        collaboration_buf = collaboration_check & ~self.init_reset_buf
        self.collaboration_degree_buf[collaboration_buf] +=1

        # deactivate finished case and exception case
        self.init_reset_buf |= self.reset_buf
        return return_

    def _fill_extras(self, env_ids):
        return_ = super()._fill_extras(env_ids)

        self.extras["episode"]["max_pos_x"] = 0.
        self.extras["episode"]["min_pos_x"] = 0.
        self.extras["episode"]["max_pos_y"] = 0.
        self.extras["episode"]["min_pos_y"] = 0.
        # self.extras["episode"]["n_obstacle_passed"] = 0.
        with torch.no_grad():
            pos_x = self.root_states[env_ids, 0] - self.env_origins[env_ids, 0]
            self.extras["episode"]["pos_x"] = pos_x
            # if self.check_BarrierTrack_terrain():
            #     self.extras["episode"]["n_obstacle_passed"] = None
        
        return return_

    def _post_physics_step_callback(self):
        return_ = super()._post_physics_step_callback()

        with torch.no_grad():
            pos_x = self.root_states[:, 0] - self.agent_origins.reshape(-1, 3)[:, 0]
            pos_y = self.root_states[:, 1] - self.agent_origins.reshape(-1, 3)[:, 1]
            self.extras["episode"]["max_pos_x"] = max(self.extras["episode"]["max_pos_x"], torch.max(pos_x).cpu())
            self.extras["episode"]["min_pos_x"] = min(self.extras["episode"]["min_pos_x"], torch.min(pos_x).cpu())
            self.extras["episode"]["max_pos_y"] = max(self.extras["episode"]["max_pos_y"], torch.max(pos_y).cpu())
            self.extras["episode"]["min_pos_y"] = min(self.extras["episode"]["min_pos_y"], torch.min(pos_y).cpu())
            # if self.check_BarrierTrack_terrain():
            #     self.extras["episode"]["n_obstacle_passed"] = None

        return return_
    
    def _compute_torques(self, actions):
        if hasattr(self, "motor_strength"):
            actions = self.motor_strength * actions
        return super()._compute_torques(actions)
    
    ##### Dealing with observations #####
    def _init_buffers(self):
        # update obs_scales components incase there will be one-by-one scaling
        # for k in self.all_obs_components:
        #     if isinstance(getattr(self.obs_scales, k, None), (tuple, list)):
        #         setattr(
        #             self.obs_scales,
        #             k,
        #             torch.tensor(getattr(self.obs_scales, k, 1.), dtype= torch.float32, device= self.device)
        #         )
        
        super()._init_buffers()
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.all_rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        # add sensor dict, which will be filled during create sensor
        self.sensor_tensor_dict = defaultdict(list)

        for env_i, env_handle in enumerate(self.envs):
            if self.cfg.obs.cfgs.depth_image:
                env_sensor_tensors = []
                for agent_i in range(self.num_agents):
                    env_sensor_tensors.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.sensor_handles[env_i][agent_i]["forward_camera"],
                            gymapi.IMAGE_DEPTH,
                    )))
                self.sensor_tensor_dict["forward_depth"].append(torch.stack(env_sensor_tensors))
            if self.cfg.obs.cfgs.rgb_image:
                env_sensor_tensors = []
                for agent_i in range(self.num_agents):
                    env_sensor_tensors.append(gymtorch.wrap_tensor(
                        self.gym.get_camera_image_gpu_tensor(
                            self.sim,
                            env_handle,
                            self.sensor_handles[env_i][agent_i]["forward_camera"],
                            gymapi.IMAGE_COLOR,
                    )))
                self.sensor_tensor_dict["forward_color"].append(torch.stack(env_sensor_tensors))

    def compute_observations(self):
        for key in self.sensor_handles[0][0].keys():
            if "camera" in key:
                # NOTE: Different from the documentation and examples from isaacgym
                # gym.fetch_results() must be called before gym.start_access_image_tensors()
                # refer to https://forums.developer.nvidia.com/t/camera-example-and-headless-mode/178901/10
                self.gym.fetch_results(self.sim, True)
                self.gym.step_graphics(self.sim)
                self.gym.render_all_camera_sensors(self.sim)
                self.gym.start_access_image_tensors(self.sim)
                break
        add_noise = self.add_noise; self.add_noise = False
        return_ = super().compute_observations() # currently self.obs_buf is a mess
        self.obs_super_impl = self.obs_buf
        self.add_noise = add_noise

        # actor obs
        self.obs_buf = self._get_obs_from_components(
            self.cfg.env.obs_components,
            privileged= False,
        )
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

        # critic obs
        if not self.num_privileged_obs is None:
            self.privileged_obs_buf[:] = self._get_obs_from_components(
                self.cfg.env.privileged_obs_components,
                privileged= getattr(self.cfg.env, "privileged_obs_gets_privilege", False),
            )
        # fixing linear velocity in proprioception observation
        if "proprioception" in getattr(self.cfg.env, "privileged_obs_components", []) \
            and getattr(self.cfg.env, "privileged_use_lin_vel", False):
            # NOTE: according to self.get_obs_segment_from_components, "proprioception" observation
            # is always the first part of this flattened observation. check super().compute_observations
            # and self.cfg.env.use_lin_vel for the reason of this if branch.
            self.privileged_obs_buf[:, :3] = self.base_lin_vel * self.obs_scales.lin_vel

        for key in self.sensor_handles[0][0].keys():
            if "camera" in key:
                self.gym.end_access_image_tensors(self.sim)
                break
        return return_

    def _get_noise_scale_vec(self, cfg):

        raise NotImplementedError

    ##### adds-on with building the environment #####
    def _create_terrain(self):
        """ Using cfg.terrain.selected to identify terrain class """
        if not isinstance(self.cfg.terrain.selected, str):
            return super()._create_terrain()
        terrain_cls = self.cfg.terrain.selected
        self.terrain = get_terrain_cls(terrain_cls)(self.cfg.terrain, self.num_envs, self.num_agents)
        self.terrain.add_terrain_to_sim(self.gym, self.sim, self.device)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        if self.cfg.domain_rand.randomize_motor:
            self.motor_strength = torch_rand_float(
                self.cfg.domain_rand.leg_motor_strength_range[0],
                self.cfg.domain_rand.leg_motor_strength_range[1],
                (self.num_envs, 12 * self.num_agents),
                device=self.device,
            )
        return super()._create_envs()
    
    # def _process_rigid_shape_props(self, props, env_id):
    #     props = super()._process_rigid_shape_props(props, env_id)
    #     if env_id == 0:
    #         all_obs_components = self.all_obs_components
    #         if "robot_config" in all_obs_components:
    #             # all_obs_components
    #             self.robot_config_buffer = torch.empty(
    #                 self.num_envs * self.num_agents, 1 + 3 + 1 + 12,
    #                 dtype= torch.float32,
    #                 device= self.device,
    #             )
        
    #     if hasattr(self, "robot_config_buffer"):
    #         self.robot_config_buffer[env_id, 0] = props[0].friction
    #     return props

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            if hasattr(self.cfg.control, "torque_limits"):
                if not isinstance(self.cfg.control.torque_limits, (tuple, list)):
                    self.torque_limits = torch.ones(self.num_actuated_dof, dtype= torch.float, device= self.device, requires_grad= False)
                    self.torque_limits *= self.cfg.control.torque_limits
                else:
                    assert not self.num_actuated_dof % len(self.cfg.control.torque_limits), "torque_limits does not fit num_dof"
                    self.torque_limits = torch.tensor(self.cfg.control.torque_limits * (self.num_actuated_dof // len(self.cfg.control.torque_limits)), dtype= torch.float, device= self.device, requires_grad= False)
        return props

    def _process_rigid_body_props(self, props, env_id):
        props = super()._process_rigid_body_props(props, env_id)

        if self.cfg.domain_rand.randomize_com:
            rng_com_x = self.cfg.domain_rand.com_range.x
            rng_com_y = self.cfg.domain_rand.com_range.y
            rng_com_z = self.cfg.domain_rand.com_range.z
            rand_com = np.random.uniform(
                [rng_com_x[0], rng_com_y[0], rng_com_z[0]],
                [rng_com_x[1], rng_com_y[1], rng_com_z[1]],
                size=(3,),
            )
            props[0].com += gymapi.Vec3(*rand_com)

        agent_id = self.env_agent_indices[env_id].reshape(-1)

        if hasattr(self, "robot_config_buffer"):
            self.robot_config_buffer[agent_id, 1] = props[0].com.x
            self.robot_config_buffer[agent_id, 2] = props[0].com.y
            self.robot_config_buffer[agent_id, 3] = props[0].com.z
            self.robot_config_buffer[agent_id, 4] = props[0].mass
            self.robot_config_buffer[agent_id, 5 : 5 + 12] = self.motor_strength[env_id].reshape(self.num_agents, -1) if hasattr(self, "motor_strength") else 1.
        return props

    def _get_env_origins(self):
        super()._get_env_origins()
        self.custom_origins = True

    def _draw_sensor_vis(self, env_h, sensor_hd):
        for sensor_name, sensor_h in sensor_hd.items():
            if "camera" in sensor_name:
                camera_transform = self.gym.get_camera_transform(self.sim, env_h, sensor_h)
                cam_axes = gymutil.AxesGeometry(scale= 0.1)
                gymutil.draw_lines(cam_axes, self.gym, self.viewer, env_h, camera_transform)

    def _draw_debug_vis(self):
        if not "height_measurements" in self.all_obs_components:
            measure_heights_tmp = self.terrain.cfg.measure_heights
            self.terrain.cfg.measure_heights = False
            return_ = super()._draw_debug_vis()
            self.terrain.cfg.measure_heights = measure_heights_tmp
        else:
            return_ = super()._draw_debug_vis()

        for env_h, sensor_hd in zip(self.envs, self.sensor_handles):
            self._draw_sensor_vis(env_h, sensor_hd)
        return return_

    def check_BarrierTrack_terrain(self):
        if getattr(self.cfg.terrain, "pad_unavailable_info", False):
            return self.cfg.terrain.selected == "BarrierTrack"
        assert self.cfg.terrain.selected == "BarrierTrack", "This implementation is only for BarrierTrack terrain"
        return True
