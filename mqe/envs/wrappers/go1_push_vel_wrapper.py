"""Velocity-MAPush wrapper.

Task: Push box in a commanded direction at a commanded speed for the full episode.
Cooperation mechanism: Angular velocity penalty makes single-agent torque costly,
while two agents from complementary positions cancel torques.

Observation space (9 dims for 2 agents):
    [cos_theta_local, sin_theta_local, speed,       # velocity command in agent's local frame
     box_x, box_y, box_yaw,                         # box relative to agent (egocentric)
     other_x, other_y, other_yaw]                    # other agent relative to this agent

Reward terms (all team rewards):
    1. velocity_tracking_reward   — cosine_similarity(box_vel, desired_vel) * exp(-|speed_error|)
    2. angular_velocity_penalty   — -|box_angular_vel_z| (COOPERATION KEY)
    3. approach_reward            — -(distance_to_box + 0.5)^2 per agent, averaged
    4. collision_punishment       — 1/(0.02 + agent_dist/3)
    5. push_reward                — reward when box is moving
    6. exception_punishment       — NaN/physics failure penalty
"""
import gym
from gym import spaces
import numpy as np
import torch
from copy import copy, deepcopy
from mqe.envs.wrappers.empty_wrapper import EmptyWrapper

from isaacgym.torch_utils import get_euler_xyz
from isaacgym import gymtorch


def normalize_angle(angle):
    """Normalize angle to [0, 2*pi)."""
    return angle % (2 * torch.pi)


class Go1PushVelWrapper(EmptyWrapper):
    """Wrapper for Velocity-MAPush task.

    Does NOT inherit from Go1PushMidWrapper — clean implementation with
    velocity-specific observations, rewards, and arrow marker logic.
    """

    def __init__(self, env):
        super().__init__(env)

        # Action space: [vx, vy, vyaw] same as mid task
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)
        self.action_scale = torch.tensor([[[0.5, 0.5, 0.5]]], device="cuda").repeat(
            self.num_envs, self.num_agents, 1
        )

        # Observation space: 3 (vel cmd) + 3 (box) + 3*(num_agents-1) (other agents) = 9 for 2 agents
        obs_dim = 3 + 3 * self.num_agents
        self.observation_space = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(obs_dim,), dtype=float
        )

        # Physics exception buffer for NaN detection
        self.physics_exception_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Velocity command buffers (per env)
        self.cmd_direction = torch.zeros(self.num_envs, device=self.device)  # theta in [0, 2pi]
        self.cmd_speed = torch.zeros(self.num_envs, device=self.device)      # speed in m/s

        # Load velocity command config
        vel_cfg = self.cfg.velocity_command
        self.speed_range = vel_cfg.speed_range
        self.direction_range = vel_cfg.direction_range
        self.arrow_offset = vel_cfg.arrow_offset

        # Reward scales
        self.velocity_tracking_scale = self.cfg.rewards.scales.velocity_tracking_scale
        self.angular_velocity_penalty_scale = self.cfg.rewards.scales.angular_velocity_penalty_scale
        self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale

        # Reward buffer for tensorboard logging
        self.reward_buffer = {
            "velocity_tracking_reward": 0,
            "angular_velocity_penalty": 0,
            "approach_to_box_reward": 0,
            "collision_punishment": 0,
            "push_reward": 0,
            "exception_punishment": 0,
            # Metrics (not rewards, but tracked for monitoring)
            "avg_direction_error": 0,
            "avg_speed_error": 0,
            "avg_box_angular_vel": 0,
            "step_count": 0,
        }

        print(f"[Go1PushVelWrapper] Velocity-MAPush task initialized")
        print(f"  Speed range: {self.speed_range}")
        print(f"  Direction range: {self.direction_range}")
        print(f"  Arrow offset: {self.arrow_offset}")
        print(f"  Obs dim: {obs_dim}")
        print(f"  Reward scales: vel_track={self.velocity_tracking_scale}, "
              f"ang_vel_pen={self.angular_velocity_penalty_scale}, "
              f"approach={self.approach_reward_scale}, "
              f"collision={self.collision_punishment_scale}, "
              f"push={self.push_reward_scale}, "
              f"exception={self.exception_punishment_scale}")

    def _sample_velocity_commands(self, env_ids):
        """Sample new velocity commands for the given environment IDs."""
        n = len(env_ids)
        if n == 0:
            return

        # Sample random direction in [direction_range[0], direction_range[1]]
        self.cmd_direction[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.direction_range[1] - self.direction_range[0])
            + self.direction_range[0]
        )

        # Sample random speed in [speed_range[0], speed_range[1]]
        self.cmd_speed[env_ids] = (
            torch.rand(n, device=self.device)
            * (self.speed_range[1] - self.speed_range[0])
            + self.speed_range[0]
        )

    def _update_arrow_marker(self):
        """Reposition the target NPC (index 1) to act as direction arrow.

        Places the target marker at box_pos + [cos(theta), sin(theta)] * arrow_offset
        in world frame, using set_actor_root_state_tensor_indexed for efficiency.
        """
        npc_states = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)

        # Box position in world frame (root_states_npc includes env_origins)
        box_pos_world = npc_states[:, 0, :3].clone()  # (num_envs, 3)

        # Arrow position: box + offset in commanded direction
        arrow_x = box_pos_world[:, 0] + torch.cos(self.cmd_direction) * self.arrow_offset
        arrow_y = box_pos_world[:, 1] + torch.sin(self.cmd_direction) * self.arrow_offset
        arrow_z = box_pos_world[:, 2]  # Same height as box

        # Update target NPC (index 1) position in root_states_npc
        npc_states[:, 1, 0] = arrow_x
        npc_states[:, 1, 1] = arrow_y
        npc_states[:, 1, 2] = arrow_z
        self.root_states_npc[:] = npc_states.reshape(-1, 13)

        # Write ONLY target NPC (index 1) state back to all_root_states
        env_ids = torch.arange(self.num_envs, device=self.device)
        target_npc_ids = self.env.env_npc_indices[env_ids, 1]  # target NPC flat indices
        target_states = npc_states[:, 1, :]  # (num_envs, 13)
        self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 1, :] = target_states

        # Update all_root_states at the target actor positions
        target_actor_ids = self.env.npc_indices[env_ids, 1]  # actor IDs in simulation
        self.env.all_root_states[target_actor_ids.long()] = target_states

        # Push to simulation
        target_actor_ids_int32 = target_actor_ids.to(torch.int32)
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.all_root_states),
            gymtorch.unwrap_tensor(target_actor_ids_int32),
            len(target_actor_ids_int32),
        )

    def _build_obs(self, base_pos, base_rpy):
        """Build agent-centric observations.

        Args:
            base_pos: (num_envs*num_agents, 3) — agent positions in env frame
            base_rpy: (num_envs*num_agents, 3) — agent RPY in env frame

        Returns:
            obs: (num_envs, num_agents, obs_dim) tensor
        """
        # Get box state in env frame
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins  # (num_envs, 3)
        box_quat = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_quat), dim=1)  # (num_envs, 3)

        # Expand to per-agent
        box_pos_exp = box_pos.repeat_interleave(self.num_agents, dim=0)  # (N*A, 3)
        box_rpy_exp = box_rpy.repeat_interleave(self.num_agents, dim=0)  # (N*A, 3)

        # Rotate box position to agent's local frame
        neg_yaw = -base_rpy[:, 2]
        cos_y = torch.cos(neg_yaw)
        sin_y = torch.sin(neg_yaw)

        dx = box_pos_exp[:, 0] - base_pos[:, 0]
        dy = box_pos_exp[:, 1] - base_pos[:, 1]

        rotated_box_x = dx * cos_y - dy * sin_y
        rotated_box_y = dx * sin_y + dy * cos_y
        rotated_box_yaw = normalize_angle(box_rpy_exp[:, 2] - base_rpy[:, 2])

        # Reshape to (num_envs, num_agents, 3)
        rotated_box = torch.stack([rotated_box_x, rotated_box_y, rotated_box_yaw], dim=1)
        rotated_box = rotated_box.reshape(self.num_envs, self.num_agents, 3)

        # Velocity command in agent's local frame
        # cmd_direction is in world frame, rotate to each agent's local frame
        base_rpy_reshaped = base_rpy.reshape(self.num_envs, self.num_agents, 3)
        agent_yaw = base_rpy_reshaped[:, :, 2]  # (num_envs, num_agents)
        cmd_dir_exp = self.cmd_direction.unsqueeze(1).repeat(1, self.num_agents)  # (N, A)
        local_dir = cmd_dir_exp - agent_yaw  # direction in agent's local frame

        cmd_speed_exp = self.cmd_speed.unsqueeze(1).repeat(1, self.num_agents)  # (N, A)

        vel_cmd = torch.stack([
            torch.cos(local_dir),   # cos(theta_local)
            torch.sin(local_dir),   # sin(theta_local)
            cmd_speed_exp,          # speed
        ], dim=2)  # (num_envs, num_agents, 3)

        # Other agents relative to this agent (egocentric)
        base_pos_r = base_pos.reshape(self.num_envs, self.num_agents, 3)
        base_rpy_r = base_rpy.reshape(self.num_envs, self.num_agents, 3)
        base_info = torch.cat([base_pos_r, base_rpy_r], dim=2)  # (N, A, 6)

        all_other_info = []
        for i in range(1, self.num_agents):
            other = deepcopy(torch.roll(base_info, i, dims=1))

            o_dx = other[:, :, 0] - base_pos_r[:, :, 0]
            o_dy = other[:, :, 1] - base_pos_r[:, :, 1]
            neg_yaw_r = -base_rpy_r[:, :, 2]
            cos_yr = torch.cos(neg_yaw_r)
            sin_yr = torch.sin(neg_yaw_r)

            rot_ox = o_dx * cos_yr - o_dy * sin_yr
            rot_oy = o_dx * sin_yr + o_dy * cos_yr
            rot_oyaw = normalize_angle(other[:, :, 5] - base_rpy_r[:, :, 2])

            other_info = torch.stack([rot_ox, rot_oy, rot_oyaw], dim=2)  # (N, A, 3)
            all_other_info.append(other_info)

        if all_other_info:
            all_other_info = torch.cat(all_other_info, dim=2)  # (N, A, 3*(A-1))
        else:
            all_other_info = torch.zeros(
                self.num_envs, self.num_agents, 0, device=self.device
            )

        # Concatenate: [vel_cmd(3), box(3), others(3*(A-1))]
        obs = torch.cat([vel_cmd, rotated_box, all_other_info], dim=2)

        return obs

    def reset(self, next_target_pos=None):
        """Reset all environments and sample new velocity commands."""
        obs_buf = self.env.reset()

        # Sample velocity commands for all environments
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_velocity_commands(all_ids)

        # Update arrow marker to reflect new commands
        self._update_arrow_marker()

        # Build observations
        base_pos = deepcopy(obs_buf.base_pos)
        base_rpy = deepcopy(obs_buf.base_rpy)
        obs = self._build_obs(base_pos, base_rpy)

        return obs

    def step(self, action, next_target_pos=None):
        """Step the environment.

        Args:
            action: (num_envs, num_agents, 3) or (num_envs*num_agents, 3) actions

        Returns:
            obs: (num_envs, num_agents, obs_dim)
            reward: (num_envs, num_agents)
            termination: (num_envs,) bool
            info: dict
        """
        action = torch.clip(action, -1.0, 1.0)
        obs_buf, _, termination, info = self.env.step(
            (action * self.action_scale).reshape(-1, self.action_space.shape[0])
        )

        # Detect resets and sample new velocity commands
        reset_ids = self.env.reset_ids
        if reset_ids is not None and len(reset_ids) > 0:
            self._sample_velocity_commands(reset_ids)

        # Update arrow marker position each step
        self._update_arrow_marker()

        # Extract states
        base_pos = deepcopy(obs_buf.base_pos)
        base_rpy = deepcopy(obs_buf.base_rpy)
        base_vel = deepcopy(obs_buf.lin_vel)

        # Build observation
        obs = self._build_obs(base_pos, base_rpy)

        # --- NaN/Inf detection ---
        obs_nan_mask = (
            torch.isnan(obs).any(dim=2).any(dim=1)
            | torch.isinf(obs).any(dim=2).any(dim=1)
        )

        # Physics NaN detection
        npc_pos = self.root_states_npc[:, :3].reshape(self.num_envs, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins
        base_pos_r = base_pos.reshape(self.num_envs, self.num_agents, -1)
        base_vel_r = base_vel.reshape(self.num_envs, self.num_agents, -1)

        nan_in_box = torch.isnan(box_pos).any(dim=1) | torch.isinf(box_pos).any(dim=1)
        nan_in_base_pos = (
            torch.isnan(base_pos_r).any(dim=2).any(dim=1)
            | torch.isinf(base_pos_r).any(dim=2).any(dim=1)
        )
        nan_in_base_vel = (
            torch.isnan(base_vel_r).any(dim=2).any(dim=1)
            | torch.isinf(base_vel_r).any(dim=2).any(dim=1)
        )
        physics_nan_mask = nan_in_box | nan_in_base_pos | nan_in_base_vel
        self.physics_exception_buf = physics_nan_mask
        self.value_exception_buf = obs_nan_mask | self.physics_exception_buf

        # Clean NaN/Inf
        obs[torch.isnan(obs)] = 0
        obs[torch.isinf(obs)] = 0
        box_pos[torch.isnan(box_pos)] = 0
        box_pos[torch.isinf(box_pos)] = 0
        base_pos_r[torch.isnan(base_pos_r)] = 0
        base_pos_r[torch.isinf(base_pos_r)] = 0
        base_vel_r[torch.isnan(base_vel_r)] = 0
        base_vel_r[torch.isinf(base_vel_r)] = 0

        # --- Reward computation ---
        self.reward_buffer["step_count"] += 1
        reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)

        # Get box velocity and angular velocity from root_states_npc
        npc_full = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)
        box_lin_vel = npc_full[:, 0, 7:10]   # (num_envs, 3) — linear velocity
        box_ang_vel = npc_full[:, 0, 10:13]  # (num_envs, 3) — angular velocity

        # Clean NaN
        box_lin_vel[torch.isnan(box_lin_vel)] = 0
        box_lin_vel[torch.isinf(box_lin_vel)] = 0
        box_ang_vel[torch.isnan(box_ang_vel)] = 0
        box_ang_vel[torch.isinf(box_ang_vel)] = 0

        # --- 1. Velocity tracking reward (PRIMARY) ---
        if self.velocity_tracking_scale != 0:
            # Desired velocity vector in world frame
            desired_vx = self.cmd_speed * torch.cos(self.cmd_direction)  # (N,)
            desired_vy = self.cmd_speed * torch.sin(self.cmd_direction)  # (N,)
            desired_vel = torch.stack([desired_vx, desired_vy], dim=1)   # (N, 2)

            # Actual box velocity (XY plane)
            actual_vel = box_lin_vel[:, :2]  # (N, 2)

            # Cosine similarity for direction matching
            desired_norm = torch.norm(desired_vel, dim=1, keepdim=True).clamp(min=1e-6)
            actual_norm = torch.norm(actual_vel, dim=1, keepdim=True).clamp(min=1e-6)
            cos_sim = torch.sum(desired_vel * actual_vel, dim=1) / (
                desired_norm.squeeze() * actual_norm.squeeze()
            )  # (N,) in [-1, 1]

            # Speed error
            actual_speed = actual_norm.squeeze()  # (N,)
            speed_error = torch.abs(actual_speed - self.cmd_speed)  # (N,)

            # Combined: direction match * speed match
            vel_track_reward = (
                self.velocity_tracking_scale
                * cos_sim
                * torch.exp(-speed_error)
            )  # (N,)

            # Team reward
            reward[:, :] += vel_track_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["velocity_tracking_reward"] += (
                torch.sum(vel_track_reward).cpu().item()
            )

            # Track metrics
            direction_error = torch.acos(cos_sim.clamp(-1, 1))  # radians
            self.reward_buffer["avg_direction_error"] += (
                torch.mean(direction_error).cpu().item()
            )
            self.reward_buffer["avg_speed_error"] += (
                torch.mean(speed_error).cpu().item()
            )

        # --- 2. Angular velocity penalty (COOPERATION KEY) ---
        if self.angular_velocity_penalty_scale != 0:
            box_ang_vel_z = torch.abs(box_ang_vel[:, 2])  # (N,)
            ang_vel_penalty = self.angular_velocity_penalty_scale * box_ang_vel_z  # negative scale

            reward[:, :] += ang_vel_penalty.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["angular_velocity_penalty"] += (
                torch.sum(ang_vel_penalty).cpu().item()
            )
            self.reward_buffer["avg_box_angular_vel"] += (
                torch.mean(box_ang_vel_z).cpu().item()
            )

        # --- 3. Approach reward (approach to box) ---
        if self.approach_reward_scale != 0:
            total_approach = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_agents):
                dist = torch.norm(box_pos - base_pos_r[:, i, :], dim=1)
                approach_r = (-(dist + 0.5) ** 2) * self.approach_reward_scale
                total_approach += approach_r
            # Average across agents
            total_approach = total_approach / self.num_agents

            reward[:, :] += total_approach.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["approach_to_box_reward"] += (
                torch.sum(total_approach).cpu().item()
            )

        # --- 4. Collision punishment ---
        if self.collision_punishment_scale != 0:
            agent_distance = torch.norm(
                base_pos_r[:, 0, :] - base_pos_r[:, 1, :], dim=1
            )
            collision_punishment = (
                (1 / (0.02 + agent_distance / 3)) * self.collision_punishment_scale
            )
            reward[:, :] += collision_punishment.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["collision_punishment"] += (
                torch.sum(collision_punishment).cpu().item()
            )

        # --- 5. Push reward (box is moving) ---
        if self.push_reward_scale != 0:
            box_speed = torch.norm(box_lin_vel[:, :2], dim=1)
            box_moving = box_speed > 0.1
            push_reward = torch.zeros(self.num_envs, device=self.device)
            push_reward[box_moving] = self.push_reward_scale

            reward[:, :] += push_reward.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["push_reward"] += torch.sum(push_reward).cpu().item()

        # --- 6. Exception punishment ---
        if self.exception_punishment_scale != 0:
            reward[self.exception_buf, :] += self.exception_punishment_scale
            reward[self.value_exception_buf, :] += self.exception_punishment_scale
            self.reward_buffer["exception_punishment"] += (
                self.exception_punishment_scale
                * (self.exception_buf.sum().item() + self.value_exception_buf.sum().item())
            )

        # --- Team reward: sum and broadcast ---
        team_reward = reward.sum(dim=1, keepdim=True)  # (N, 1)
        team_reward[torch.isnan(team_reward)] = 0
        team_reward[torch.isinf(team_reward)] = 0
        reward = team_reward.repeat(1, self.num_agents)  # (N, A)

        return obs, reward, termination, info
