import gym
from gym import spaces
import numpy
import torch
from copy import copy,deepcopy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper
from mqe.envs.wrappers.utils.trajectory import TrajectoryPlanner
from openrl.modules.ppo_module import PPOModule
from mqe.envs.wrappers.utils.rrt import KinodynamicRRT,TwoDVisualizer

from isaacgym.torch_utils import *

# convert tensor to numpy
def _t2n(tensor):
    return tensor.detach().cpu().numpy()

# tensor type
def rotation_matrix_2D(theta):
    theta = theta.float()
    cos_theta = torch.cos(theta)  
    sin_theta = torch.sin(theta)  

    rotation_matrices = torch.stack([
        torch.stack([cos_theta, -sin_theta], dim=1),
        torch.stack([sin_theta, cos_theta], dim=1)
    ], dim=1)

    return rotation_matrices

def euler_to_quaternion_tensor(euler_angles):
    roll = euler_angles[:, 0]
    pitch = euler_angles[:, 1]
    yaw = euler_angles[:, 2]

    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qx, qy, qz, qw], dim=1)
    return quaternion

def normalize_rpy(box_rpy):

    box_rpy = box_rpy % (2 * torch.pi)
    
    return box_rpy

def interpolate_trajectory(trajectory, target_points=13):
    num_envs, current_points = trajectory.shape[0], trajectory.shape[1] // 2
    new_trajectory = torch.zeros((num_envs, target_points * 2), device='cuda:0')
    
    for i in range(num_envs):
        x = trajectory[i, 0::2].cpu().numpy()
        y = trajectory[i, 1::2].cpu().numpy()
        
        if current_points < target_points:
            interp_x = np.interp(
                np.linspace(0, current_points - 1, target_points),
                np.arange(current_points), x
            )
            interp_y = np.interp(
                np.linspace(0, current_points - 1, target_points),
                np.arange(current_points), y
            )
        else:
            arc_length = np.cumsum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
            arc_length = np.insert(arc_length, 0, 0)
            interp_arc = np.linspace(0, arc_length[-1], target_points)
            interp_x = np.interp(interp_arc, arc_length, x)
            interp_y = np.interp(interp_arc, arc_length, y)
        
        new_trajectory[i, 0::2] = torch.tensor(interp_x)
        new_trajectory[i, 1::2] = torch.tensor(interp_y)
    
    return new_trajectory

class Go1PushUpperWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(26,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float)     # should be revised in openrl_ws/utils.py
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)
        self.net_origin = torch.tensor(self.cfg.generalize_obsersation.net_origin).to(self.device)
        
        self.planning = True
        self.reset_count = 0

        self.obs1_pos = torch.randn(self.num_envs, 3, device="cuda")
        self.obs2_pos = torch.randn(self.num_envs, 3, device="cuda")
      
        self.target_reward_scale = self.cfg.rewards.scales.target_reward_scale
        self.reach_target_reward_scale = self.cfg.rewards.scales.reach_target_reward_scale
        self.trajectory_rewards_scale = self.cfg.rewards.scales.trajectory_rewards_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale
        self.obstacle_reward_scale = self.cfg.rewards.scales.obstacle_reward_scale

        self.reward_buffer = {
            "distance_to_target_reward": 0,
            "exception_punishment": 0,
            "obstacle_reward_scale": 0,
            "reach_target_reward":0,
            "trajectory_rewards":0,
            "step_count": 0,
        }

        # init command policy
        self._prepare_command_policy()

    def _prepare_command_policy(self):
        assert self.cfg.control.command_network_path != None, "No command policy provided."

        self.rnn_states_command = np.zeros((self.num_envs * self.num_agents, 1, 64))
        self.mask_command = np.ones((self.num_envs * self.num_agents, 1))

        self.command_observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3 + 3 * self.num_agents,), dtype=float)
        self.command_action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        self.command_module = torch.load(self.cfg.control.command_network_path, map_location=self.device)

    def _init_extras(self, obs):
        return

    def reset_target_positions(self, env_ids):
        new_positions = torch.randn(len(env_ids), 3, device="cuda")
        new_positions[:, 0] = new_positions[:, 0].abs() + 9.5
        new_positions[:, 1] = new_positions[:, 1] + 2 * torch.sign(new_positions[:, 1])
        new_positions[:, 2] = 0.1

        self.final_target_pos[env_ids] = new_positions

    def set_target_pos(self, target_pos):
        self.cfg.goal.received_final_pos = target_pos

    def reset(self,next_target_pos=None):
        self.reset_count += 1
        if getattr(self.cfg.goal, "received_goal_pos",False):
            if next_target_pos == None:
                pass
                # raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.next_target_pos = next_target_pos

        obs_buf = self.env.reset()
        
        # extract npc pos from self.root_states_npc\
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        
        # init obstacles position
        self.obs1_pos = npc_pos[:,3,:] - self.env.env_origins
        self.obs2_pos = npc_pos[:,4,:] - self.env.env_origins
        # (optional)
        # self.obs1_pos[:, 0] = torch.FloatTensor(self.num_envs).uniform_(0, 14).to("cuda")
        # self.obs2_pos[:, 0] = torch.FloatTensor(self.num_envs).uniform_(0, 14).to("cuda")

        # self.obs1_pos[:, 1] = torch.FloatTensor(self.num_envs).uniform_(-7, 7).to("cuda")
        # self.obs2_pos[:, 1] = torch.FloatTensor(self.num_envs).uniform_(-7, 7).to("cuda")

        # self.obs1_pos[:, 2] = 0.1
        # self.obs2_pos[:, 2] = 0.1
        self.cfg.obstacle_state.obs1_pos = self.obs1_pos
        self.cfg.obstacle_state.obs2_pos = self.obs2_pos
        
        # init final goal position
        self.final_target_pos = torch.randn(self.num_envs, 3, device="cuda")
        self.final_target_pos[:, 0] = self.final_target_pos[:, 0].abs() + 9.5
        # self.final_target_pos[:, 1] = torch.rand(self.final_target_pos[:, 1].shape) * 10 - 5  # random y freely
        self.final_target_pos[:, 1] = self.final_target_pos[:, 1] + 3 * torch.sign(self.final_target_pos[:, 1])  #  farther y
        self.final_target_pos[:, 2] = 0.1
        self.set_target_pos(self.final_target_pos)

        # generate observation
        base_pos = obs_buf.base_pos.view(self.num_envs, -1)
        base_rpy = obs_buf.base_rpy.view(self.num_envs, -1)
        base_info = torch.cat([base_pos, base_rpy], dim=1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        box_rot = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        target_pos = npc_pos[:,1,:] - self.env.env_origins
        self.Planner = TrajectoryPlanner(self.num_envs, box_pos, self.final_target_pos)
        self.trajectory = self.Planner.get_trajectory()

        # initialize RRT
        if self.num_obs > 0 and self.planning == True and self.reset_count == 4:  
            x_lim = (0, 14)
            y_lim = (-7, 7)
            self.rrt = KinodynamicRRT(x_lim=x_lim, y_lim=y_lim)
            vis = TwoDVisualizer()

            obs1_pos_2d = self.cfg.obstacle_state.obs1_pos[:, :2]  
            obs2_pos_2d = self.cfg.obstacle_state.obs2_pos[:, :2] 
            obs_combined = torch.cat([obs1_pos_2d.unsqueeze(1), obs2_pos_2d.unsqueeze(1)], dim=1) 
            start= box_pos[:, :2]
            end = self.final_target_pos[:, :2]

            for i in range(self.num_envs):
                if vis is not None:
                    vis.clear()
                    vis.set_bounds(x_lim, y_lim)
                    vis.draw_state(start[i], color='k', s=20, alpha=1)
                    vis.draw_state(end[i], color='g', s=20, alpha=1)
                    vis.draw_obstacle(obs_combined[i], s=20, alpha=1)
                planned_trace = self.rrt.plan(start=start[i], goal=end[i] , visualizer=vis, obstacle_states=obs_combined[i], 
                                                                    action_scale=0.75, 
                                                                    timeout=60, 
                                                                    goal_sample_prob=0.75, 
                                                                    REACH_THERESHOLD=1.5, 
                                                                    COLLISION_THERESHOLD=0.3)
                if planned_trace is not None:
                    concatenated_trace = torch.cat(planned_trace).unsqueeze(0)
                    self.trajectory[i, :] = interpolate_trajectory(concatenated_trace)
                else:
                    print("Fail to plane")

        next_planning_position = self.Planner.update_next_planning_position(box_pos, self.trajectory)  
        obs = torch.cat([base_info, target_pos[:, :2], box_pos[:, :2], box_rot, self.obs1_pos[:, :2], self.obs2_pos[:, :2], next_planning_position], dim=1).unsqueeze(1)
        return obs

    def step(self, action):
        sub_goals = action.clone()
        sub_goals = torch.clip(sub_goals, -1, 1)
        sub_goals = sub_goals.squeeze(1)
        sub_goals[:, 0] = (sub_goals[:, 0] + 1) * 7         # adjust to your map size 
        sub_goals[:, 1] = sub_goals[:, 1] * 5.5             # example: mow x~[0, 7*2] y~[-5.5, 5.5]
        fill_tensor = torch.full((sub_goals.size(0), 1), 0.15, dtype=sub_goals.dtype, device=sub_goals.device)
        sub_goals = torch.cat([sub_goals, fill_tensor], dim=1)

        self.env.next_target_pos = sub_goals
        
        # organize middle-layer observation
        # get agent state
        base_pos = deepcopy(self.obs_buf.base_pos) 
        base_rpy = deepcopy(self.obs_buf.base_rpy)  
        # get box state and target pos
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins 
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1 , 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        # rotate box state and target pos to agent's local state
        box_pos = box_pos.repeat_interleave(self.num_agents, dim=0)
        target_pos = target_pos.repeat_interleave(self.num_agents, dim=0)
        box_rpy = box_rpy.repeat_interleave(self.num_agents, dim=0)
        target_rpy = target_rpy.repeat_interleave(self.num_agents, dim=0)
        rotated_box_pos = torch.stack([(box_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (box_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                       (box_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (box_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                      box_pos[:, 2]], dim=1)
        rotated_target_pos = torch.stack([(target_pos[:, 0] - base_pos[:, 0]) * torch.cos(-base_rpy[:, 2]) - (target_pos[:, 1] - base_pos[:, 1]) * torch.sin(-base_rpy[:, 2]),
                                          (target_pos[:, 0] - base_pos[:, 0]) * torch.sin(-base_rpy[:, 2]) + (target_pos[:, 1] - base_pos[:, 1]) * torch.cos(-base_rpy[:, 2]),
                                         target_pos[:, 2]], dim=1)
        rotated_box_rpy = deepcopy(box_rpy)
        rotated_box_rpy[:,2] = box_rpy[:,2] - base_rpy[:,2]
        rotated_box_rpy = normalize_rpy(rotated_box_rpy)
        rotated_target_rpy = deepcopy(target_rpy)
        rotated_target_rpy[:,2] = target_rpy[:,2] - base_rpy[:,2]
        rotated_target_rpy = normalize_rpy(rotated_target_rpy)
        rotated_box_pos = rotated_box_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_box_rpy = rotated_box_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_pos = rotated_target_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        rotated_target_rpy = rotated_target_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # rotate other agents' state to agent's local state
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_info = torch.cat([base_pos, base_rpy], dim=2)
        all_base_info = []
        if self.num_agents != 1:
            for i in range(1, self.env.num_agents):
                other_base_info = deepcopy(torch.roll(base_info, i, dims=1))
                # roate other agents' state to agent's local state
                other_base_pos = torch.stack([(other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.cos(-base_rpy[:, :, 2]) - (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.sin(-base_rpy[:, :, 2]),
                                              (other_base_info[:, :, 0] - base_pos[:, :, 0]) * torch.sin(-base_rpy[:, :, 2]) + (other_base_info[:, :, 1] - base_pos[:, :, 1]) * torch.cos(-base_rpy[:, :, 2]),
                                              other_base_info[:, :, 2]], dim=2)
                other_base_rpy = deepcopy(other_base_info[:, :, 3:6])
                other_base_rpy[:, :, 2] = other_base_info[:, :, 5] - base_rpy[:, :, 2]
                other_base_rpy = normalize_rpy(other_base_rpy)
                other_base_info = torch.cat([other_base_pos[:,:,:2], other_base_rpy[:,:,2].unsqueeze(2)], dim=2)
                all_base_info.append(other_base_info)
            all_base_info = torch.cat(all_base_info, dim=2)

        # observation for middle layer
        command_obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)

        # remove nan and inf in obs
        command_obs[torch.isnan(command_obs)] = 0
        command_obs[torch.isinf(command_obs)] = 0

        # fit for openrl
        command_obs = command_obs.cpu().numpy()
        command_obs = np.concatenate(command_obs, axis=0)

        # middle layer policy
        command_action,_ = self.command_module.act(command_obs, self.rnn_states_command, self.mask_command, deterministic=True)
        
        # fit for openrl
        command_action = np.array(np.split(_t2n(command_action), self.num_envs))
        command_action = torch.from_numpy(0.5 * command_action).cuda().clip(-1, 1)
        command_action = torch.clip(command_action, -1, 1)

        obs_buf, _, termination, info = self.env.step((command_action * self.action_scale).reshape(-1, self.command_action_space.shape[0]))

        #organize obs
        #extract target position and box pos from self.root_states_npc\
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins

        # generate observation
        base_pos = obs_buf.base_pos.view(self.num_envs, -1)
        base_rpy = obs_buf.base_rpy.view(self.num_envs, -1)
        base_info = torch.cat([base_pos, base_rpy], dim=1)

        box_rot = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]

        # env reset
        reset_envs = (self.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
        if len(reset_envs) > 0:
            self.reset_target_positions(reset_envs)
            self.set_target_pos(self.final_target_pos)
            self.Planner.reset_trajectory(reset_envs, self.final_target_pos)

            if self.num_obs > 0:
                self.obs1_pos[reset_envs, :] = npc_pos[reset_envs , 3, :] - self.env.env_origins[reset_envs, :]
                self.obs2_pos[reset_envs, :] = npc_pos[reset_envs , 4, :] - self.env.env_origins[reset_envs, :]
            else:
                self.obs1_pos[reset_envs, 0] = torch.FloatTensor(len(reset_envs)).uniform_(0, 9).to("cuda")
                self.obs2_pos[reset_envs, 0] = torch.FloatTensor(len(reset_envs)).uniform_(0, 9).to("cuda")

                self.obs1_pos[reset_envs, 1] = torch.FloatTensor(len(reset_envs)).uniform_(-5, 5).to("cuda")
                self.obs2_pos[reset_envs, 1] = torch.FloatTensor(len(reset_envs)).uniform_(-5, 5).to("cuda")

                self.obs1_pos[reset_envs, 2] = 0.1
                self.obs2_pos[reset_envs, 2] = 0.1

        if self.planning == False:
            self.trajectory = self.Planner.get_trajectory(self.episode_length_buf)

        if len(reset_envs) > 0 and self.num_obs > 0:    # reset your obstacles' position here
            # one way to reset obstacles(optional)
            # third_point = self.trajectory[reset_envs, 4:6]
            # seventh_point = self.trajectory[reset_envs, 12:14]

            # y_offset = (torch.rand(len(reset_envs), 1, device='cuda:0') * (3 - 2) + 2)  
            # sign = torch.sign(torch.rand(len(reset_envs), 1, device='cuda:0') - 0.5)  

            # third_point[:, 1] = third_point[:, 1] + (y_offset * sign).squeeze()
            # seventh_point[:, 1] = seventh_point[:, 1] - (y_offset * sign).squeeze()

            # self.cfg.obstacle_state.obs1_pos[reset_envs, :] = torch.cat((third_point, torch.full((len(reset_envs), 1), 0.1, device='cuda:0')), dim=1)
            # self.cfg.obstacle_state.obs2_pos[reset_envs, :] = torch.cat((seventh_point, torch.full((len(reset_envs), 1), 0.1, device='cuda:0')), dim=1)
            self.cfg.obstacle_state.obs1_pos[reset_envs, 0] = torch.FloatTensor(len(reset_envs)).uniform_(0, 14).to("cuda")
            self.cfg.obstacle_state.obs2_pos[reset_envs, 0] = torch.FloatTensor(len(reset_envs)).uniform_(0, 14).to("cuda")

            self.cfg.obstacle_state.obs1_pos[reset_envs, 1] = torch.FloatTensor(len(reset_envs)).uniform_(-7, 7).to("cuda")
            self.cfg.obstacle_state.obs2_pos[reset_envs, 1] = torch.FloatTensor(len(reset_envs)).uniform_(-7, 7).to("cuda")

            self.obs1_pos = self.cfg.obstacle_state.obs1_pos
            self.obs2_pos = self.cfg.obstacle_state.obs2_pos

            self.trajectory = self.Planner.get_trajectory(self.episode_length_buf)          

        next_planning_position = self.Planner.update_next_planning_position(box_pos, self.trajectory)  
        obs = torch.cat([base_info, target_pos[:, :2], box_pos[:, :2], box_rot, self.obs1_pos[:, :2], self.obs2_pos[:, :2], next_planning_position], dim=1).unsqueeze(1)
        
        # remove nan and inf in obs and reward
        self.value_exception_buf = torch.isnan(obs).any(dim=2).any(dim=1) \
                                | torch.isinf(obs).any(dim=2).any(dim=1) \
                                
        obs[torch.isnan(obs)] = 0
        obs[torch.isinf(obs)] = 0

        # calculate reward 
        base_pos = obs_buf.base_pos     # (env_num, agent_num, 3)
        base_vel = obs_buf.lin_vel      # (env_num, agent_num, 3)
        base_rpy = obs_buf.base_rpy     # (env_num, agent_num, 3)
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_vel = base_vel.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # occlude nan or inf
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        box_rpy[torch.isnan(box_rpy)] = 0
        box_rpy[torch.isinf(box_rpy)] = 0
        
        self.reward_buffer["step_count"] += 1

        reward = torch.zeros([self.env.num_envs, 1], device=self.env.device)

        # calculate reach target reward and set finish task termination
        if self.reach_target_reward_scale != 0:
            reward[self.finished_buf, :] += self.reach_target_reward_scale
            self.reward_buffer["reach_target_reward"] += self.reach_target_reward_scale * self.finished_buf.sum().item()
        
        # calculate exception punishment
        if self.exception_punishment_scale != 0:
            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            self.reward_buffer["exception_punishment"] += self.exception_punishment_scale * self.exception_buf.sum().item()

        # calculate distance from current_box_pos to target_box_pos reward
        if self.target_reward_scale != 0:
            target_distance = torch.norm(box_pos[:, :2] - self.final_target_pos[:, :2], dim=1)
            target_distance = 1 / (1 + target_distance)
            target_distance *= self.target_reward_scale
            target_distance = target_distance.unsqueeze(1)
            reward[:, :] += target_distance
            self.reward_buffer["distance_to_target_reward"] += torch.sum(target_distance).cpu()

        if self.obstacle_reward_scale != 0:
            obs1_distance = torch.norm(sub_goals[:, :2] - self.obs1_pos[:, :2], dim=1)
            obs2_distance = torch.norm(sub_goals[:, :2] - self.obs2_pos[:, :2], dim=1)
            obs1_distance = 1 / (1 + obs1_distance)
            obs2_distance = 1 / (1 + obs2_distance)
            obs1_reward = self.obstacle_reward_scale * obs1_distance
            obs2_reward = self.obstacle_reward_scale * obs2_distance
            obstacle_reward = obs1_reward + obs2_reward
            obstacle_reward = obstacle_reward.unsqueeze(1)
            reward[:, :] += obstacle_reward
            self.reward_buffer["obstacle_reward_scale"] += torch.sum(obstacle_reward).cpu()

        # calculate trajectory rewards
        if self.trajectory_rewards_scale != 0:
            next_planne_distances = torch.norm(sub_goals[:, :2] - next_planning_position, dim=1)
            next_planne_reward = 1 / (1 + next_planne_distances)
            next_planne_reward *= self.trajectory_rewards_scale
            next_planne_reward = next_planne_reward.unsqueeze(1)
            reward[:, :] += next_planne_reward
            self.reward_buffer["trajectory_rewards"] += torch.sum(next_planne_reward).cpu()

        return obs, reward, termination, info
