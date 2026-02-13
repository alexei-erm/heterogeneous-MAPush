import gym
from gym import spaces
import numpy
import torch
from copy import copy,deepcopy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import *

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

class Go1PushMidWrapper(EmptyWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Check if heterogeneous mode is enabled
        self.is_hetero = getattr(self.cfg.hetero, 'use_hetero', False) if hasattr(self.cfg, 'hetero') else False

        # Action space: Both Go1 and Jackal use [vx, vy, vyaw] (3 DOF)
        # Difference is in the low-level controller (locomotion policy vs differential drive)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5],],], device="cuda").repeat(self.num_envs, self.num_agents, 1)

        if self.is_hetero:
            self.hetero_agent_types = self.cfg.hetero.hetero_agent_types
            print(f"[Go1PushMidWrapper] Heterogeneous mode enabled:")
            print(f"  Agent types: {self.hetero_agent_types}")
            print(f"  Both agents use 3 DOF [vx, vy, vyaw] action space")
            # Identify control types for each agent
            ctrl_descriptions = []
            for agent_type in self.hetero_agent_types:
                if agent_type in ['go1', 'anymal_c']:  # Quadrupeds use locomotion policy
                    ctrl_descriptions.append(f"{agent_type}: Locomotion policy")
                elif agent_type == 'jackal':  # Wheeled robots use differential drive
                    ctrl_descriptions.append(f"{agent_type}: Differential drive")
                else:
                    ctrl_descriptions.append(f"{agent_type}: Unknown")
            print(f"  Control types: {', '.join(ctrl_descriptions)}")

        # Initialize physics exception buffer for NaN detection
        self.physics_exception_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Observation space remains same for both modes
        if getattr(self.cfg.goal, "general_dist",False):
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(3 + 3 * self.num_agents,), dtype=float)
            pass
        else:
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2 + 3 * self.num_agents,), dtype=float)
        
        # for hard setting of reward scales (not recommended)
        
        self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
        self.target_reward_scale = self.cfg.rewards.scales.target_reward_scale
        self.reach_target_reward_scale = self.cfg.rewards.scales.reach_target_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale
        self.ocb_reward_scale = self.cfg.rewards.scales.ocb_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale
        self.proximity_penalty_scale = getattr(self.cfg.rewards.scales, "proximity_penalty_scale", 0.002)

        self.reward_buffer = {
            "distance_to_target_reward": 0,
            "exception_punishment": 0,
            "approach_to_box_reward": 0,
            "collision_punishment":0,
            "reach_target_reward":0,
            "push_reward":0,
            "ocb_reward":0,
            "goal_push_bonus": 0,
            # Iter6 new rewards (old, not used)
            "engagement_bonus": 0,
            "cooperation_bonus": 0,
            "same_side_bonus": 0,
            "blocking_penalty": 0,
            # Iter8 gated rewards
            "gating_factor": 0,
            # CRITIC12: Three-tier cooperation bonuses
            "dual_engagement_bonus": 0,
            "synchronized_contact_bonus": 0,
            "bilateral_push_bonus": 0,
            "proximity_penalty": 0,
            # CRITIC15 v4: Collaboration rewards
            "dual_push_bonus": 0,
            # CRITIC17 v2: Gaussian proximity bonus
            "gaussian_proximity_bonus": 0,
            "step_count": 0,
        }

        # DEBUG: Exception tracking counters
        # self.debug_step_counter = 0
        # self.debug_sim_exception_count = 0
        # self.debug_nan_exception_count = 0

        # Contact threshold for individualized rewards (distance to box center)
        # Agents within this distance are considered "in contact" with box
        self.contact_threshold = getattr(self.cfg.rewards, "contact_threshold", 0.8)

        # COLLABORATION ENFORCEMENT: Only give reach_target_reward if BOTH agents are in contact
        # This forces collaboration - solo pushing to goal gives NO success reward
        # Box is 1.2m x 1.2m, so contact_threshold of 0.8m means agent is touching the edge
        self.require_both_contact_for_success = getattr(self.cfg.rewards, "require_both_contact_for_success", False)
        if self.require_both_contact_for_success:
            print(f"[Go1PushMidWrapper] COLLABORATION ENFORCEMENT ENABLED")
            print(f"  reach_target_reward ONLY given if BOTH agents within {self.contact_threshold}m of box at success")
            print(f"  Solo pushing to goal = NO reward!")

        # Whether to use individualized rewards (for HAPPO)
        self.individualized_rewards = getattr(self.cfg.rewards, "individualized_rewards", False)
        # Iter8: Gated shared rewards - all rewards multiplied by min engagement of both agents
        # This prevents freeloading: if one agent runs away, BOTH get reduced rewards
        self.shared_gated_rewards = getattr(self.cfg.rewards, "shared_gated_rewards", False)
        self.gating_threshold = getattr(self.cfg.rewards, "gating_threshold", 2.0)  # meters

        # CRITIC12: Three-tier cooperation bonuses (prevents freeloading through positive rewards)
        # v2: Relaxed thresholds to make Tier 2 & 3 achievable
        self.cooperation_rewards = getattr(self.cfg.rewards, "cooperation_rewards", False)
        self.dual_engagement_threshold = 2.0  # meters - both agents within this distance of box (was 1.5)
        self.contact_threshold_sync = 1.2  # meters - both agents in contact with box (was 0.8, +50%)
        self.push_force_threshold = 0.03  # minimum useful push force magnitude (was 0.1, -70%)

        # CRITIC12 v7b: Directional push reward - rewards box velocity toward goal, punishes away
        # Increased from 0.003 to 0.01 to strongly discourage pushing in wrong direction

        # Original MAPush rewards (teamified) - uses vanilla reward structure
        # When enabled: 7 original rewards, goal_push_bonus and proximity_penalty disabled
        # approach_to_box uses AVERAGE (not sum), collision uses -0.0025, OCB uses ±0.004
        self.mapush_og_rewards_teamified = getattr(self.cfg.rewards, "mapush_og_rewards_teamified", False)

        # CRITIC15 v3: Reward scale testing - uses adjusted scales from heavy_cuboid Exp 7
        # IMPORTANT: For HAPPO, distance_to_target_reward is DISABLED (found to hurt performance)
        # See claude_summaries/heavy_cuboid_testing_happo.md for details
        self.reward_scale_testing = getattr(self.cfg.rewards, "reward_scale_testing", False)
        self.goal_push_bonus_scale = getattr(self.cfg.rewards, "goal_push_bonus_scale", 0.01)

        # When reward_scale_testing is enabled:
        # - Applies adjusted scales from heavy_cuboid Exp 7 (best MAPPO results)
        # - Does NOT enable mapush_og_rewards_teamified (keeps distance_to_target DISABLED for HAPPO)
        # - HAPPO Exp 2/3 showed >95% SR without distance_to_target, Exp 4 with it was worse
        if self.reward_scale_testing:
            # NOTE: Do NOT set mapush_og_rewards_teamified = True here!
            # This keeps distance_to_target_reward disabled, which works better for HAPPO
            # See heavy_cuboid_testing_happo.md Exp 4 results

            # Apply Exp 7 scales:
            # - push_reward_scale: 0.004 (2.5x from 0.0015) - encourages pushing
            # - ocb_reward_scale: 0.008 (2x from 0.004) - optimal contact positioning
            # - reach_target_reward_scale: 10 (1x) - original sparse success bonus
            # - approach_reward_scale: 0.00075 (1x) - original
            # - collision_punishment_scale: -0.0025 (1x) - original
            # - exception_punishment_scale: -5 (1x) - original
            # - target_reward_scale: SET but distance_to_target_reward NOT COMPUTED (disabled)
            self.target_reward_scale = 0.01      # 3x (but reward disabled for HAPPO)
            self.push_reward_scale = 0.004       # 2.5x increase
            self.ocb_reward_scale = 0.008        # 2x increase
            self.reach_target_reward_scale = 10  # 1x (original)
            self.approach_reward_scale = 0.00075 # 1x (original)
            self.collision_punishment_scale = -0.0025  # 1x (original)
            self.exception_punishment_scale = -5 # 1x (original)

            # Disable HAPPO-specific reward additions
            self.proximity_penalty_scale = 0.0
            self.goal_push_bonus_scale = 0.0
            self.dual_push_bonus_scale = 0.0

            print(f"[REWARD_SCALE_TESTING] Using heavy_cuboid Exp 7 scales (HAPPO optimized)")
            print(f"  push_reward_scale: {self.push_reward_scale} (2.5x)")
            print(f"  ocb_reward_scale: {self.ocb_reward_scale} (2x)")
            print(f"  reach_target_reward_scale: {self.reach_target_reward_scale} (1x)")
            print(f"  approach_reward_scale: {self.approach_reward_scale} (1x)")
            print(f"  collision_punishment_scale: {self.collision_punishment_scale} (1x)")
            print(f"  exception_punishment_scale: {self.exception_punishment_scale} (1x)")
            print(f"  distance_to_target_reward: DISABLED (hurts HAPPO performance)")

        # CRITIC15 v4: Collaboration rewards - dual pushing bonus
        # Works in conjunction with mapush_og_rewards_teamified
        self.collaboration_rewards = getattr(self.cfg.rewards, "collaboration_rewards", False)
        self.dual_push_velocity_threshold = 0.05  # m/s - minimum velocity toward goal to count as "pushing"
        self.dual_push_bonus_scale = 0.005  # per step when both agents pushing toward goal

        # BASELINE MAPPO REWARDS MODE: Uses ONLY original 7 MAPush rewards with TRUE ORIGINAL scales
        # Disables ALL HAPPO-specific rewards (proximity_penalty, goal_push_bonus, cooperation, etc.)
        # Uses ORIGINAL distance_to_target formula WITH penalty term
        self.baseline_mappo_rewards = getattr(self.cfg.rewards, "baseline_mappo_rewards", False)
        if self.baseline_mappo_rewards:
            # Force disable all HAPPO-specific reward flags
            self.individualized_rewards = False
            self.shared_gated_rewards = False
            self.cooperation_rewards = False
            self.collaboration_rewards = False
            self.reward_scale_testing = False
            self.positive_approachtobox_reward = False
            # Force HAPPO-specific scales to 0
            self.proximity_penalty_scale = 0.0
            self.goal_push_bonus_scale = 0.0
            self.engagement_bonus_scale = 0.0
            self.cooperation_bonus_scale = 0.0
            self.same_side_bonus_scale = 0.0
            self.blocking_penalty_scale = 0.0
            self.directional_progress_scale = 0.0
            self.dual_push_bonus_scale = 0.0
            self.gaussian_proximity_amplitude = 0.0
            print(f"[Go1PushMidWrapper] BASELINE MAPPO REWARDS MODE ACTIVE (TRUE ORIGINAL)")
            print(f"  Using ONLY 7 original rewards with ORIGINAL scales and ORIGINAL distance formula (with penalty)")
            print(f"  Scales: target={self.target_reward_scale}, approach={self.approach_reward_scale}, "
                  f"collision={self.collision_punishment_scale}, push={self.push_reward_scale}, "
                  f"ocb={self.ocb_reward_scale}, reach={self.reach_target_reward_scale}, exception={self.exception_punishment_scale}")

        # MAPPO HEAVY BOX REWARDS MODE: Uses adjusted scales for 8kg box training
        # Uses 6 rewards (distance_to_target COMPLETELY DISABLED)
        # See claude_summaries/heavy_cuboid_testing_mappo.md for details
        self.mappo_heavybox_rewards = getattr(self.cfg.rewards, "mappo_heavybox_rewards", False)
        if self.mappo_heavybox_rewards:
            # Force disable all HAPPO-specific reward flags
            self.individualized_rewards = False
            self.shared_gated_rewards = False
            self.cooperation_rewards = False
            self.collaboration_rewards = False
            self.reward_scale_testing = False
            self.positive_approachtobox_reward = False
            # Force HAPPO-specific scales to 0
            self.proximity_penalty_scale = 0.0
            self.goal_push_bonus_scale = 0.0
            self.engagement_bonus_scale = 0.0
            self.cooperation_bonus_scale = 0.0
            self.same_side_bonus_scale = 0.0
            self.blocking_penalty_scale = 0.0
            self.directional_progress_scale = 0.0
            self.dual_push_bonus_scale = 0.0
            self.gaussian_proximity_amplitude = 0.0
            print(f"[Go1PushMidWrapper] MAPPO HEAVY BOX REWARDS MODE ACTIVE")
            print(f"  Using 6 rewards (distance_to_target DISABLED), Exp 7 scales")
            print(f"  Scales: push={self.push_reward_scale} (2.5x), ocb={self.ocb_reward_scale} (2x), "
                  f"approach={self.approach_reward_scale}, collision={self.collision_punishment_scale}, "
                  f"reach={self.reach_target_reward_scale}, exception={self.exception_punishment_scale}")

        # CRITIC17: Positive approach_to_box bonus (Gaussian with steep falloff)
        # When enabled: ADDS positive Gaussian bonus on top of original negative penalty
        # Net reward = negative penalty + positive bonus
        # Bonus has VERY STEEP falloff after 1.5m (essentially 0 beyond that)
        self.positive_approachtobox_reward = getattr(self.cfg.rewards, "positive_approachtobox_reward", False)
        # Gaussian parameters: bonus = amplitude * exp(-(distance/sigma)^2)
        self.gaussian_proximity_sigma = 0.4  # Steepness (smaller = steeper falloff)
        self.gaussian_proximity_amplitude = 0.006  # Max bonus at distance=0

        # Iter6: New reward scales - scaled down to match codebase (old rewards are ~0.001-0.004)
        self.engagement_bonus_scale = getattr(self.cfg.rewards, "engagement_bonus_scale", 0.0004)  # Near box
        self.cooperation_bonus_scale = getattr(self.cfg.rewards, "cooperation_bonus_scale", 0.0002)  # Both near box
        self.same_side_bonus_scale = getattr(self.cfg.rewards, "same_side_bonus_scale", 0.0004)  # Both on push side
        self.blocking_penalty_scale = getattr(self.cfg.rewards, "blocking_penalty_scale", -0.001)  # Between box and goal
        self.directional_progress_scale = getattr(self.cfg.rewards, "directional_progress_scale", 0.003)  # Shared: box toward goal

    def _init_extras(self, obs):
        return
        # self.gate_pos = obs.env_info["gate_deviation"]
        # self.gate_pos[:, 0] += self.BarrierTrack_kwargs["init"]["block_length"] + self.BarrierTrack_kwargs["gate"]["block_length"] / 2
        # self.gate_pos = self.gate_pos.unsqueeze(1).repeat(1, self.num_agents, 1)
        # self.gate_distance = self.gate_pos.reshape(-1, 2)[:, 0]


    def calc_normal_vector_for_obc_reward(self, vertex_list, pos_tensor):
        pos_tensor = pos_tensor.to(self.device)
        vertices = torch.tensor(vertex_list, device=self.device).float()
        num_vertices = vertices.shape[0]

        edges = torch.roll(vertices, -1, dims=0) - vertices
        vp = pos_tensor[:, None, :] - vertices[None, :, :]

        edges_expanded = edges[None, :, :].repeat(pos_tensor.shape[0], 1, 1)
        edge_lengths = torch.norm(edges_expanded, dim=2, keepdim=True)
        edge_unit = edges_expanded / edge_lengths
        edge_normals = torch.stack([-edge_unit[:,:,1], edge_unit[:,:,0]], dim=2)

        cross_prod = torch.abs(vp[:,:,0] * edge_unit[:,:,1] - vp[:,:,1] * edge_unit[:,:,0])
        dot_product1 = (vp * edges_expanded).sum(dim=2)
        dot_product2 = (torch.roll(vp, -1, dims=1) * edges_expanded).sum(dim=2)

        on_segment = (dot_product1 >= 0) & (dot_product2 <= 0)
        dist_to_line = torch.where(on_segment, cross_prod, torch.tensor(float('inf'), device=self.device))

        dist_to_vertex1 = torch.norm(vp, dim=2)
        dist_to_vertex2 = torch.norm(pos_tensor[:, None, :] - torch.roll(vertices, -1, dims=0)[None, :, :], dim=2)

        min_dist_each_edge, indices = torch.min(torch.stack([dist_to_line, dist_to_vertex1, dist_to_vertex2], dim=-1), dim=2)
        min_dist, indices = torch.min(min_dist_each_edge,dim=1)
        selected_normals = edge_normals[0][indices]

        return selected_normals
    
    def reset(self,next_target_pos=None):
        if getattr(self.cfg.goal, "received_goal_pos",False):
            if next_target_pos == None:
                pass
                # raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.next_target_pos = next_target_pos

        obs_buf = self.env.reset()

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos) 
        base_rpy = deepcopy(obs_buf.base_rpy) 
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

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat([rotated_target_pos[:,:,:2], rotated_target_rpy[:,:,2].unsqueeze(2), rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2)], dim=2)
            else:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        self.last_box_state = None
        return obs

    def step(self, action, next_target_pos=None):
        if next_target_pos is not None:
            assert next_target_pos.shape == (self.num_envs, 3)
            assert self.cfg.generalize_obsersation.rotate_obs
            assert self.cfg.goal.received_goal_pos

        if getattr(self.cfg.goal, "received_goal_pos",False):
            if next_target_pos is None:
                raise ValueError("next_target_pos is required when received_goal_pos is True")
            self.env.next_target_pos = next_target_pos

        action = torch.clip(action, -1.0, 1.0)
        if getattr(self.cfg.goal, "received_goal_pos",False):
            if torch.any(self.env.stop_buf):
                action[self.env.stop_buf] = torch.tensor([0., 0., 0.], device=self.device).repeat(self.stop_buf.sum().item(), self.num_agents, 1)
        # set static action
        # action = torch.tensor([[1.0, 0.0, 0.0]], device="cuda").repeat(self.num_envs, 1, 1)
        obs_buf, _, termination, info = self.env.step((action * self.action_scale).reshape(-1, self.action_space.shape[0]))

        # get agent state
        base_pos = deepcopy(obs_buf.base_pos) 
        base_rpy = deepcopy(obs_buf.base_rpy) 
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

        if getattr(self.cfg.goal, "general_dist", False):
            obs = torch.cat([rotated_target_pos[:,:,:2], rotated_target_rpy[:,:,2].unsqueeze(2), rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)
        else:
            if all_base_info == []:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2)], dim=2)
            else:
                obs = torch.cat([rotated_target_pos[:,:,:2], rotated_box_pos[:,:,:2], rotated_box_rpy[:,:,2].unsqueeze(2), all_base_info], dim=2)

        # get env_id which should be reseted, because of nan or inf in obs, reward, or physics states
        obs_nan_mask = torch.isnan(obs).any(dim=2).any(dim=1) | torch.isinf(obs).any(dim=2).any(dim=1)

        # Combine observation NaN with physics NaN (from earlier detection)
        self.value_exception_buf = obs_nan_mask | self.physics_exception_buf \
                                
        # remove nan and inf in obs and reward
        obs[torch.isnan(obs)] = 0
        obs[torch.isinf(obs)] = 0

        # calculate reward
        box_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0]
        target_state = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1]
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:,0,:] - self.env.env_origins
        target_pos = npc_pos[:,1,:] - self.env.env_origins 
        box_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0 , 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_qyaternion), dim=1)
        target_qyaternion = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1 , 3:7]
        target_rpy = torch.stack(get_euler_xyz(target_qyaternion), dim=1)

        base_pos = obs_buf.base_pos # (env_num, agent_num, 3)
        base_vel = obs_buf.lin_vel # (env_num, agent_num, 3)
        base_rpy = obs_buf.base_rpy # (env_num, agent_num, 3)
        # Reshape all robot states before NaN detection
        base_pos = base_pos.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_vel = base_vel.reshape([self.env.num_envs, self.env.num_agents, -1])
        base_rpy = base_rpy.reshape([self.env.num_envs, self.env.num_agents, -1])

        # CRITICAL FIX: Detect and handle NaN/Inf from physics states
        # Physics simulation can produce NaN in unstable states (e.g., Jackal instability)
        # Mark environments with NaN for reset via value_exception_buf
        # This triggers episode reset + exception punishment (-5)

        # Detect which environments have NaN/Inf in physics states
        # Use chained any() calls for PyTorch 1.13 compatibility (doesn't support dim=tuple)
        nan_in_box = torch.isnan(box_pos).any(dim=1) | torch.isinf(box_pos).any(dim=1)
        nan_in_base_pos = torch.isnan(base_pos).any(dim=2).any(dim=1) | torch.isinf(base_pos).any(dim=2).any(dim=1)
        nan_in_base_vel = torch.isnan(base_vel).any(dim=2).any(dim=1) | torch.isinf(base_vel).any(dim=2).any(dim=1)
        physics_nan_mask = nan_in_box | nan_in_base_pos | nan_in_base_vel

        # Mark environments with physics NaN for reset (will be combined with obs NaN later)
        self.physics_exception_buf = physics_nan_mask

        # Log physics NaN events (less verbose than before)
        if physics_nan_mask.any():
            num_nan_envs = physics_nan_mask.sum().item()
            if not hasattr(self, '_last_nan_log_step'):
                self._last_nan_log_step = 0
            # Log only every 100 steps to avoid spam
            if self.reward_buffer["step_count"] - self._last_nan_log_step >= 100:
                print(f"[Physics NaN] {num_nan_envs} environments with NaN in physics (will reset)")
                self._last_nan_log_step = self.reward_buffer["step_count"]

        # Clean NaN for current step to prevent immediate crash
        # Environments will reset on next step via exception mechanism
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        base_vel[torch.isnan(base_vel)] = 0
        base_vel[torch.isinf(base_vel)] = 0

        # occlude nan or inf
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        target_pos[torch.isnan(target_pos)] = 0
        target_pos[torch.isinf(target_pos)] = 0
        base_pos[torch.isnan(base_pos)] = 0
        base_pos[torch.isinf(base_pos)] = 0
        box_rpy[torch.isnan(box_rpy)] = 0
        box_rpy[torch.isinf(box_rpy)] = 0

        self.reward_buffer["step_count"] += 1
        reward = torch.zeros([self.env.num_envs, self.num_agents], device=self.env.device)

        # ============================================================
        # ITER8: Gated Shared Rewards
        # Calculate gating factor based on min engagement of all agents
        # If either agent is far from box, ALL rewards are reduced
        # ============================================================
        if self.shared_gated_rewards:
            # Calculate engagement for each agent (0 = far, 1 = close)
            agent_engagements = []
            for i in range(self.num_agents):
                dist = torch.norm(box_pos[:, :2] - base_pos[:, i, :2], dim=1)
                engagement = torch.clamp(1.0 - dist / self.gating_threshold, 0.0, 1.0)
                agent_engagements.append(engagement)
            agent_engagements = torch.stack(agent_engagements, dim=1)  # (num_envs, num_agents)

            # Gating factor = minimum engagement across all agents
            # If ANY agent is far, gating drops for everyone
            gating_factor = torch.min(agent_engagements, dim=1)[0]  # (num_envs,)
            self.reward_buffer["gating_factor"] += torch.mean(gating_factor).cpu().item()
        else:
            gating_factor = None

        # =================================================================
        # CRITIC14: ALL TEAM REWARDS (centralized critic compatibility)
        # Key Change: Converted ALL rewards to team rewards for centralized critic
        # Previously approach_to_box was individual → caused V(s) mismatch
        #
        # Active Rewards (8) - ALL TEAM:
        #   1. reach_target_reward (10) - sparse success [TEAM]
        #   2. approach_to_box_reward (0.00075) - sum of both agents' penalties [TEAM - FIXED]
        #   3. push_reward (0.0015) - curriculum learning [TEAM]
        #   4. goal_push_bonus (0.01) - directional velocity [TEAM]
        #   5. ocb_reward (+0.01/-0.004) - positioning (asymmetric) [TEAM]
        #   6. proximity_penalty (0.002) - quadratic distance penalty [TEAM]
        #   7. collision_punishment (-0.0025) - safety [TEAM - FIXED]
        #   8. exception_punishment (-5) - safety [TEAM]
        # Disabled: distance_to_target, cooperation bonuses
        # =================================================================

        # calculate reach target reward and set finish task termination
        # Iter5: INDIVIDUAL - only agents near box get credit for completion
        # Iter8: GATED - reward multiplied by min engagement (both must be near)
        # COLLABORATION ENFORCEMENT: Only reward if BOTH agents in contact at success time
        if self.reach_target_reward_scale != 0:
            # Determine which environments get success reward
            if self.require_both_contact_for_success:
                # COLLABORATION MODE: Both agents must be within contact_threshold of box
                dist_agent0 = torch.norm(box_pos[:, :2] - base_pos[:, 0, :2], dim=1)
                dist_agent1 = torch.norm(box_pos[:, :2] - base_pos[:, 1, :2], dim=1)
                both_in_contact = (dist_agent0 < self.contact_threshold) & (dist_agent1 < self.contact_threshold)

                # Only give reward if task finished AND both agents contributed
                valid_success = self.finished_buf & both_in_contact

                # Log collaboration stats periodically
                if self.finished_buf.sum() > 0:
                    solo_successes = (self.finished_buf & ~both_in_contact).sum().item()
                    collab_successes = valid_success.sum().item()
                    if solo_successes > 0:
                        print(f"[COLLAB] Blocked {solo_successes} solo successes, rewarded {collab_successes} collaborative successes")
            else:
                # Original: all finished environments get reward
                valid_success = self.finished_buf

            if self.individualized_rewards:
                for i in range(self.num_agents):
                    agent_box_dist = torch.norm(box_pos - base_pos[:, i, :], dim=1)
                    contact_weight = torch.clamp(1.0 - (agent_box_dist - self.contact_threshold) / self.contact_threshold, 0.0, 1.0)
                    reward[valid_success, i] += self.reach_target_reward_scale * contact_weight[valid_success]
                self.reward_buffer["reach_target_reward"] += self.reach_target_reward_scale * valid_success.sum().item()
            elif self.shared_gated_rewards:
                # Iter8: Gated - only get full reward if BOTH agents engaged
                gated_reward = self.reach_target_reward_scale * gating_factor[valid_success]
                reward[valid_success, :] += gated_reward.unsqueeze(1)
                self.reward_buffer["reach_target_reward"] += torch.sum(gated_reward).cpu().item()
            else:
                # Original: shared
                reward[valid_success, :] += self.reach_target_reward_scale
                self.reward_buffer["reach_target_reward"] += self.reach_target_reward_scale * valid_success.sum().item()
        
        # calculate exception punishment
        if self.exception_punishment_scale != 0:
            # DEBUG: Track exception counts
            if not hasattr(self, 'debug_exception_step_counter'):
                self.debug_exception_step_counter = 0
                self.debug_sim_exception_count = 0
                self.debug_nan_exception_count = 0

            num_sim_exceptions = self.exception_buf.sum().item()
            num_nan_exceptions = self.value_exception_buf.sum().item()
            self.debug_sim_exception_count += num_sim_exceptions
            self.debug_nan_exception_count += num_nan_exceptions
            self.debug_exception_step_counter += 1

            # Log every 500 steps
            if self.debug_exception_step_counter % 500 == 0:
                total_exceptions = self.debug_sim_exception_count + self.debug_nan_exception_count
                print(f"[DEBUG Exception Step {self.debug_exception_step_counter}] Over last 500 steps:")
                print(f"  Simulation exceptions (roll/pitch/z/collision): {self.debug_sim_exception_count}")
                print(f"  NaN/Inf exceptions: {self.debug_nan_exception_count}")
                print(f"  Total: {total_exceptions}")
                if hasattr(self.env, 'cfg') and hasattr(self.env.cfg, 'termination'):
                    print(f"  Config has termination: YES, terms={self.env.cfg.termination.termination_terms}")
                else:
                    print(f"  Config has termination: NO!!!")
                # Reset counters
                self.debug_sim_exception_count = 0
                self.debug_nan_exception_count = 0

            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            # reward[self.time_out_buf, :] += self.exception_punishment_scale
            self.reward_buffer["exception_punishment"] += self.exception_punishment_scale * \
                    (self.exception_buf.sum().item() + self.value_exception_buf.sum().item())

        # distance_to_target_reward: Progress shaping based on box-to-goal distance
        # CRITIC13 v2: Disabled for HAPPO experiments (redundant with goal_push_bonus)
        # mapush_og_rewards_teamified: RE-ENABLED with original formula
        # baseline_mappo_rewards: ENABLED with ORIGINAL formula (includes penalty term)
        # mappo_heavybox_rewards: DISABLED - distance reward hurts heavy box training
        # reward_scale_testing: DISABLED - distance reward hurts HAPPO (see heavy_cuboid_testing_happo.md)
        #
        # IMPORTANT: For heavy box (8kg) training, distance_to_target_reward should be COMPLETELY DISABLED
        # Both MAPPO (mappo_heavybox_rewards) and HAPPO (reward_scale_testing) perform better without it
        # See heavy_cuboid_testing_happo.md Exp 2 vs Exp 4 results
        skip_distance_reward = self.mappo_heavybox_rewards or self.reward_scale_testing

        if not skip_distance_reward and (self.mapush_og_rewards_teamified or self.baseline_mappo_rewards) and self.target_reward_scale != 0:
            if self.last_box_state is None:
                self.last_box_state = copy(box_state)
            past_distance = self.env.dist_calculator.cal_dist(self.last_box_state, target_state)
            distance = self.env.dist_calculator.cal_dist(box_state, target_state)

            # ORIGINAL MAPush formula: progress shaping + distance penalty
            distance_reward = self.target_reward_scale * 100 * (2 * (past_distance - distance) - 0.01 * distance)

            # Iter8: Apply gating if enabled (NOT for baseline_mappo_rewards)
            if self.shared_gated_rewards and not self.baseline_mappo_rewards:
                distance_reward = distance_reward * gating_factor
            # Shared reward (both agents get same reward - original behavior)
            reward[:, :] += distance_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["distance_to_target_reward"] += torch.sum(distance_reward).cpu()

        # calculate distance from each robot to box reward
        # baseline_mappo_rewards: ORIGINAL per-agent reward (each agent gets own distance penalty)
        # CRITIC14: Converted to TEAM reward for centralized critic compatibility
        # CRITIC17 v2: Original negative penalty + Gaussian positive bonus (when flag enabled)
        # mapush_og_rewards_teamified: AVERAGE of both rewards (preserves original magnitude)
        # Default (CRITIC14): SUM of both rewards (2x magnitude)
        if self.approach_reward_scale != 0:
            if self.baseline_mappo_rewards:
                # ORIGINAL MAPPO: Per-agent reward - each agent gets its own distance penalty
                reward_logger = []
                for i in range(self.num_agents):
                    distance = torch.norm(box_pos - base_pos[:, i, :], dim=1, keepdim=True)
                    distance_reward = (-(distance + 0.5)**2) * self.approach_reward_scale
                    reward_logger.append(torch.sum(distance_reward).cpu())
                    reward[:, i] += distance_reward.squeeze(-1)
                self.reward_buffer["approach_to_box_reward"] += np.sum(np.array(reward_logger))
            else:
                # HAPPO: Team reward (sum or average)
                total_approach_reward = torch.zeros(self.num_envs, device=self.device)
                total_gaussian_bonus = torch.zeros(self.num_envs, device=self.device)

                for i in range(self.num_agents):
                    distance = torch.norm(box_pos - base_pos[:, i, :], dim=1)

                    # Always compute original negative quadratic penalty
                    distance_penalty = (-(distance + 0.5)**2) * self.approach_reward_scale
                    total_approach_reward += distance_penalty

                    # CRITIC17 v2: ADD Gaussian bonus on top (steep falloff after 1.5m)
                    if self.positive_approachtobox_reward:
                        # Gaussian: bonus = amplitude * exp(-(distance/sigma)^2)
                        # sigma=0.4 gives ULTRA STEEP falloff
                        # At d=0m: bonus=0.006, at d=1.5m: bonus~0 (0.00001)
                        gaussian_bonus = self.gaussian_proximity_amplitude * torch.exp(
                            -(distance / self.gaussian_proximity_sigma)**2
                        )
                        total_gaussian_bonus += gaussian_bonus

                # mapush_og_rewards_teamified: AVERAGE to preserve designed scale magnitude
                if self.mapush_og_rewards_teamified:
                    total_approach_reward = total_approach_reward / self.num_agents
                    if self.positive_approachtobox_reward:
                        total_gaussian_bonus = total_gaussian_bonus / self.num_agents

                # Add Gaussian bonus to the penalty (net reward)
                if self.positive_approachtobox_reward:
                    total_approach_reward = total_approach_reward + total_gaussian_bonus

                # Team reward: both agents get the (avg or sum) of rewards
                reward[:, :] += total_approach_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["approach_to_box_reward"] += torch.sum(total_approach_reward).cpu().item()

                # Track Gaussian bonus separately for monitoring
                if self.positive_approachtobox_reward:
                    self.reward_buffer["gaussian_proximity_bonus"] += torch.sum(total_gaussian_bonus).cpu().item() 

        # calculate collision punishment
        # baseline_mappo_rewards: ORIGINAL per-agent reward (both agents in pair get punishment)
        # CRITIC14: Explicit TEAM reward for centralized critic compatibility
        # mapush_og_rewards_teamified: Uses original scale -0.0025 (hardcoded)
        if self.collision_punishment_scale != 0:
            if self.baseline_mappo_rewards:
                # ORIGINAL MAPPO: Per-agent collision punishment (both agents in pair get it)
                punishment_logger = []
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        distance = torch.norm(base_pos[:, i, :] - base_pos[:, j, :], dim=1, keepdim=True)
                        collision_punishment = (1 / (0.02 + distance/3)) * self.collision_punishment_scale
                        punishment_logger.append(torch.sum(collision_punishment).cpu())
                        reward[:, i] += collision_punishment.squeeze(-1)
                        reward[:, j] += collision_punishment.squeeze(-1)
                self.reward_buffer["collision_punishment"] += np.sum(np.array(punishment_logger))
            else:
                # HAPPO: Team reward
                # Distance between agents (only 2 agents, so single pair)
                agent_distance = torch.norm(base_pos[:, 0, :] - base_pos[:, 1, :], dim=1)
                # Use original scale when mapush_og_rewards_teamified, otherwise from config
                if self.mapush_og_rewards_teamified:
                    collision_scale = -0.0025  # Original MAPush scale
                else:
                    collision_scale = self.collision_punishment_scale  # From config
                collision_punishment = (1 / (0.02 + agent_distance / 3)) * collision_scale
                # Team reward: both agents get the same punishment
                reward[:, :] += collision_punishment.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["collision_punishment"] += torch.sum(collision_punishment).cpu().item()

        # CRITIC13 v2: Re-enabled push_reward - curriculum learning (weak signal, then directional takes over)
        # calculate push reward for each agent
        # Iter9: Contact-weighted - only agents near box get push credit
        if self.push_reward_scale != 0:
            # Check if box is moving (velocity > 0.1)
            box_moving = torch.norm(self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9], dim=1) > 0.1

            if self.individualized_rewards:
                # Iter9: Individual push reward - weighted by contact with box
                reward_logger = []
                for i in range(self.num_agents):
                    agent_box_dist = torch.norm(box_pos[:, :2] - base_pos[:, i, :2], dim=1)
                    contact_weight = torch.clamp(1.0 - (agent_box_dist - self.contact_threshold) / self.contact_threshold, 0.0, 1.0)
                    push_reward = self.push_reward_scale * box_moving.float() * contact_weight
                    reward[:, i] += push_reward
                    reward_logger.append(torch.sum(push_reward).cpu())
                self.reward_buffer["push_reward"] += np.sum(np.array(reward_logger))
            else:
                # Original: shared push reward
                push_reward = torch.zeros((self.env.num_envs,), device=self.env.device)
                push_reward[box_moving] = self.push_reward_scale
                # Iter8: Apply gating if enabled
                if self.shared_gated_rewards:
                    push_reward = push_reward * gating_factor
                reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["push_reward"] += torch.sum(push_reward).cpu()

        # =================================================================
        # CRITIC15 v4: Dual Pushing Bonus (Collaboration Rewards)
        # Rewards when BOTH agents are pushing toward goal simultaneously
        # Use with mapush_og_rewards_teamified for clean baseline + collaboration signal
        # Prevents exploit: both agents must move TOWARD goal (not opposite sides)
        # =================================================================
        if self.collaboration_rewards:
            # Direction from box to goal (normalized)
            box_to_goal = target_pos[:, :2] - box_pos[:, :2]  # (num_envs, 2)
            box_to_goal_norm = box_to_goal / (torch.norm(box_to_goal, dim=1, keepdim=True) + 1e-6)

            # Project each agent's velocity onto goal direction
            agent0_vel_toward_goal = torch.sum(base_vel[:, 0, :2] * box_to_goal_norm, dim=1)  # (num_envs,)
            agent1_vel_toward_goal = torch.sum(base_vel[:, 1, :2] * box_to_goal_norm, dim=1)  # (num_envs,)

            # Both agents must be pushing toward goal (prevents opposite-side pushing exploit)
            both_pushing = (agent0_vel_toward_goal > self.dual_push_velocity_threshold) & \
                          (agent1_vel_toward_goal > self.dual_push_velocity_threshold)

            # Binary bonus: either 0.005 or 0
            dual_push_bonus = torch.zeros(self.num_envs, device=self.device)
            dual_push_bonus[both_pushing] = self.dual_push_bonus_scale

            # Iter8: Apply gating if enabled (for consistency with other rewards)
            if self.shared_gated_rewards:
                dual_push_bonus = dual_push_bonus * gating_factor

            # Add as shared team reward (both agents get same bonus)
            reward[:, :] += dual_push_bonus.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["dual_push_bonus"] += torch.sum(dual_push_bonus).cpu().item()

        # =================================================================
        # CRITIC12 v7: Directional Push Reward
        # Rewards box velocity TOWARD goal, penalizes velocity away from goal
        # This directly incentivizes pushing in the correct direction
        # mapush_og_rewards_teamified: DISABLED (not in original MAPush)
        # =================================================================
        if self.goal_push_bonus_scale != 0 and not self.mapush_og_rewards_teamified:
            # Get box velocity (linear velocity from root states)
            box_velocity = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 7:9]  # (num_envs, 2)

            # Direction from box to goal (normalized)
            box_to_goal = target_pos[:, :2] - box_pos[:, :2]  # (num_envs, 2)
            box_to_goal_norm = box_to_goal / (torch.norm(box_to_goal, dim=1, keepdim=True) + 1e-6)

            # Project box velocity onto goal direction
            # Positive = moving toward goal, Negative = moving away
            velocity_toward_goal = torch.sum(box_velocity * box_to_goal_norm, dim=1)  # (num_envs,)

            # Scale the reward (continuous, not binary)
            directional_push_reward = self.goal_push_bonus_scale * velocity_toward_goal

            # Iter8: Apply gating if enabled
            if self.shared_gated_rewards:
                directional_push_reward = directional_push_reward * gating_factor

            # Add as shared team reward
            reward[:, :] += directional_push_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["goal_push_bonus"] += torch.sum(directional_push_reward).cpu().item()

        # =================================================================
        # CRITIC12 v5: Joint OCB Reward (binary)
        # Only reward positively when BOTH agents are on correct pushing side
        # Punish (weaker) when any agent is on wrong side
        # This fixes the issue where per-agent OCB cancels out in team reward
        # =================================================================
        if self.ocb_reward_scale != 0:
            if getattr(self.cfg.rewards,"expanded_ocb_reward",False):
                original_target_direction=(target_pos[:, :2] - box_pos[:, :2])/(torch.norm((target_pos[:, :2] - box_pos[:, :2]+0.01),dim=1,keepdim=True))
                delta_yaw = target_rpy[:, 2] - box_rpy[:, 2]
                # delta_yaw -->(-pi, pi)
                delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi
                # rotate target direction by delta_yaw/2
                target_direction = torch.stack([original_target_direction[:, 0] * torch.cos(-delta_yaw/2) - original_target_direction[:, 1] * torch.sin(-delta_yaw/2),
                                                original_target_direction[:, 0] * torch.sin(-delta_yaw/2) + original_target_direction[:, 1] * torch.cos(-delta_yaw/2)], dim=1)
            else:
                target_direction = (target_pos[:, :2] - box_pos[:, :2])/(torch.norm((target_pos[:, :2] - box_pos[:, :2]),dim=1,keepdim=True))

            vertex_list = self.cfg.asset.vertex_list

            # Compute raw OCB for each agent (dot product with target direction)
            # Positive = correct side (behind box), Negative = wrong side (in front)
            raw_ocb_list = []
            for i in range(self.num_agents):
                gf_pos = base_pos[:, i, :2] - box_pos[:, :2]
                rotation_matrix = rotation_matrix_2D(-box_rpy[:, 2])
                box_relative_pos = torch.bmm(rotation_matrix, gf_pos.unsqueeze(2)).squeeze(2)
                normal_vector = self.calc_normal_vector_for_obc_reward(vertex_list, box_relative_pos)
                rotation_matrix = rotation_matrix_2D(box_rpy[:, 2])
                normal_vector = torch.bmm(rotation_matrix, normal_vector.to(rotation_matrix.device).unsqueeze(2)).squeeze(2)
                raw_ocb = torch.sum(target_direction * normal_vector, dim=1)
                raw_ocb_list.append(raw_ocb)

            # baseline_mappo_rewards: ORIGINAL per-agent OCB reward
            if self.baseline_mappo_rewards:
                # ORIGINAL MAPPO: Per-agent OCB reward
                reward_logger = []
                for i in range(self.num_agents):
                    ocb_reward = raw_ocb_list[i] * self.ocb_reward_scale
                    reward[:, i] += ocb_reward
                    reward_logger.append(torch.sum(ocb_reward).cpu())
                self.reward_buffer["ocb_reward"] += np.sum(np.array(reward_logger))
            elif self.mapush_og_rewards_teamified:
                # CRITIC15 v2: Continuous averaged OCB (teamified for HAPPO)
                # Each agent gets partial immediate credit, doesn't rely solely on critic
                total_ocb = torch.zeros(self.num_envs, device=self.device)
                for i in range(self.num_agents):
                    # raw_ocb ranges from -1 (maximally wrong) to +1 (perfectly positioned)
                    ocb_reward = raw_ocb_list[i] * 0.004  # Original scale
                    total_ocb += ocb_reward

                # Average to preserve scale magnitude (like approach_to_box_reward)
                joint_ocb_reward = total_ocb / self.num_agents

                # Iter8: Apply gating if enabled
                if self.shared_gated_rewards:
                    joint_ocb_reward = joint_ocb_reward * gating_factor

                # Add as TEAM reward (same for both agents)
                reward[:, :] += joint_ocb_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["ocb_reward"] += torch.sum(joint_ocb_reward).cpu().item()
            else:
                # CRITIC14: Joint binary OCB - both agents must be on correct side
                both_correct = (raw_ocb_list[0] > 0) & (raw_ocb_list[1] > 0)
                # Asymmetric +0.01/-0.004
                joint_ocb_reward = torch.where(
                    both_correct,
                    torch.full_like(raw_ocb_list[0], self.ocb_reward_scale),   # Both correct: +0.01
                    torch.full_like(raw_ocb_list[0], -0.004)                   # Any wrong: -0.004 (fixed)
                )

                # Iter8: Apply gating if enabled
                if self.shared_gated_rewards:
                    joint_ocb_reward = joint_ocb_reward * gating_factor

                # Add as TEAM reward (same for both agents)
                reward[:, :] += joint_ocb_reward.unsqueeze(1).repeat(1, self.num_agents)
                self.reward_buffer["ocb_reward"] += torch.sum(joint_ocb_reward).cpu().item()

        # =================================================================
        # CRITIC13 v3: Robot Proximity Penalty (quadratic)
        # Encourages agents to stay within optimal distance (~1.2m = box side length)
        # Penalty is 0 when within range, grows quadratically when too far apart
        # mapush_og_rewards_teamified: DISABLED (not in original MAPush)
        # =================================================================
        if self.proximity_penalty_scale != 0 and not self.mapush_og_rewards_teamified:
            # Distance between the two agents (XY plane)
            agent_distance = torch.norm(base_pos[:, 0, :2] - base_pos[:, 1, :2], dim=1)  # (num_envs,)

            # Optimal distance = box side length (1.2m)
            optimal_distance = 1.2

            # Excess distance beyond optimal (clamped to 0 if within range)
            excess_distance = torch.clamp(agent_distance - optimal_distance, min=0)

            # Quadratic penalty: grows slowly near optimal, faster when very far
            proximity_penalty = -self.proximity_penalty_scale * (excess_distance ** 2)  # (num_envs,)

            # Iter8: Apply gating if enabled
            if self.shared_gated_rewards:
                proximity_penalty = proximity_penalty * gating_factor

            # Add as TEAM reward (same for both agents)
            reward[:, :] += proximity_penalty.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["proximity_penalty"] += torch.sum(proximity_penalty).cpu().item()

        # =================================================================
        # CRITIC13: Cooperation bonuses DISABLED - redundant with other signals
        # CRITIC12: Two-Tier Cooperation Bonuses (v6)
        # Positive reward shaping to encourage both agents to work together
        # All bonuses are SHARED (same reward for both agents)
        # Dual Engagement REMOVED - redundant with approach_reward, too loose
        # =================================================================
        if False and self.cooperation_rewards:
            # Calculate distances from each agent to box
            dist_agent0 = torch.norm(box_pos[:, :2] - base_pos[:, 0, :2], dim=1)  # (num_envs,)
            dist_agent1 = torch.norm(box_pos[:, :2] - base_pos[:, 1, :2], dim=1)  # (num_envs,)

            # Bonus 1: Synchronized Contact - Both agents touching box (+0.008/step)
            both_in_contact = (dist_agent0 < self.contact_threshold_sync) & (dist_agent1 < self.contact_threshold_sync)
            synchronized_contact_bonus = torch.zeros(self.num_envs, device=self.device)
            synchronized_contact_bonus[both_in_contact] = 0.008
            reward[:, :] += synchronized_contact_bonus.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["synchronized_contact_bonus"] += torch.sum(synchronized_contact_bonus).cpu().item()

            # Bonus 2: Bilateral Push - Both agents pushing in useful directions (+0.01/step)
            # Direction from box to goal
            box_to_goal = target_pos[:, :2] - box_pos[:, :2]  # (num_envs, 2)
            box_to_goal_norm = box_to_goal / (torch.norm(box_to_goal, dim=1, keepdim=True) + 1e-6)  # normalized

            # For each agent, compute push contribution (velocity projected onto goal direction)
            # We use agent velocity as a proxy for push force
            agent0_vel_toward_goal = torch.sum(base_vel[:, 0, :2] * box_to_goal_norm, dim=1)  # (num_envs,)
            agent1_vel_toward_goal = torch.sum(base_vel[:, 1, :2] * box_to_goal_norm, dim=1)  # (num_envs,)

            # Both pushing if both have velocity toward goal above threshold
            both_pushing = (agent0_vel_toward_goal > self.push_force_threshold) & \
                           (agent1_vel_toward_goal > self.push_force_threshold) & \
                           both_in_contact  # Must also be in contact
            bilateral_push_bonus = torch.zeros(self.num_envs, device=self.device)
            bilateral_push_bonus[both_pushing] = 0.01
            reward[:, :] += bilateral_push_bonus.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["bilateral_push_bonus"] += torch.sum(bilateral_push_bonus).cpu().item()

        self.last_box_state = deepcopy(box_state)

        # =================================================================
        # HAPPO/MAPPO TEAM REWARD: Sum per-agent rewards into team reward
        # All agents receive IDENTICAL team reward for proper CTDE training
        # Credit assignment happens via HAPPO's sequential importance weighting
        # =================================================================
        # Sum across agents: team_reward = sum of all agent rewards
        team_reward = reward.sum(dim=1, keepdim=True)  # (num_envs, 1)

        # CRITICAL: Clean NaN/Inf from team_reward BEFORE expanding
        # Must clean before expand() because expand() creates a view (not modifiable)
        team_reward[torch.isnan(team_reward)] = 0
        team_reward[torch.isinf(team_reward)] = 0

        # Give identical team reward to ALL agents
        # Use repeat() instead of expand() to create a proper copy (not a view)
        reward = team_reward.repeat(1, self.num_agents)  # (num_envs, num_agents)

        return obs, reward, termination, info
