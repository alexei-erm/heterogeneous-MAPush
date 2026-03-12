"""Velocity-MAPush wrapper.

Task: Push box in a commanded direction for the full episode.
Cooperation mechanism: Angular velocity penalty makes single-agent torque costly,
while two agents from complementary positions cancel torques.

Observation space (15 dims for 2 agents, agent-centric):
    cmd(2):       [cos_theta_local, sin_theta_local]    — direction only, no speed
    box_pos(3):   [box_dx, box_dy, box_dyaw]           — box position relative to agent
    box_vel(3):   [box_vx_local, box_vy_local, box_wz]  — box velocity in agent frame + angular vel
    self_vel(2):  [self_vx, self_vy]                     — agent's own velocity in agent frame
    other_pos(3): [other_dx, other_dy, other_dyaw]       — other agent relative to this agent
    other_vel(2): [other_vx_local, other_vy_local]       — other agent velocity in agent frame

Reward terms (all team rewards):
    1. velocity_tracking_reward   — cosine_similarity(box_vel, desired_dir) [direction-only]
    2. angular_velocity_penalty   — -|box_angular_vel_z| (COOPERATION KEY)
    3. velocity_ocb_reward        — positioning: be on push side of box (continuous, averaged)
    4. approach_reward            — -(distance_to_box + 0.5)^2 per agent, averaged
    5. collision_punishment       — 1/(0.02 + agent_dist/3)
    6. push_reward                — reward when box is moving
    7. exception_punishment       — NaN/physics failure penalty
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

        # Legacy mode: include cmd_speed in obs (16 dims) for old checkpoints
        self.legacy_vel_obs = getattr(self.cfg.rewards, "legacy_vel_obs", False)

        # Observation space: 2(cmd) + 3(box pos) + 3(box vel+wz) + 2(self vel)
        #   + (num_agents-1) * [3(other pos) + 2(other vel)]
        # = 2 + 3 + 3 + 2 + (A-1)*5 = 10 + 5*(A-1) = 15 for 2 agents
        # Legacy: 3(cmd with speed) → 11 + 5*(A-1) = 16 for 2 agents
        cmd_dims = 3 if self.legacy_vel_obs else 2
        obs_dim = (cmd_dims + 8) + 5 * (self.num_agents - 1)
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
        self.velocity_ocb_scale = self.cfg.rewards.scales.velocity_ocb_scale
        self.approach_reward_scale = self.cfg.rewards.scales.approach_reward_scale
        self.collision_punishment_scale = self.cfg.rewards.scales.collision_punishment_scale
        self.push_reward_scale = self.cfg.rewards.scales.push_reward_scale
        self.exception_punishment_scale = self.cfg.rewards.scales.exception_punishment_scale

        # Dual-push balance config
        self.dual_push_alpha = getattr(self.cfg.rewards, 'dual_push_balance_alpha', 1.0)

        # Compute per-agent total mass from Isaac Gym rigid body properties
        self.agent_masses = self._compute_agent_masses()

        # Pre-compute per-agent body offsets and non-foot body indices for contact
        # force push detection. Handles heterogeneous agents with different body counts.
        # robot_num_bodies: [n_bodies_agent0, n_bodies_agent1, ...]
        robot_num_bodies = getattr(self.env, 'robot_num_bodies', None)
        if robot_num_bodies is None:
            # Homogeneous: all agents have same body count
            robot_num_bodies = [self.env.num_bodies] * self.num_agents

        # Cumulative body offsets in the contact_forces tensor
        self.agent_body_offsets = []
        offset = 0
        for nb in robot_num_bodies:
            self.agent_body_offsets.append(offset)
            offset += nb

        # Per-agent non-foot body indices (local, relative to agent's own bodies)
        # For heterogeneous agents, each robot type has different body count and feet.
        self.per_agent_non_foot_indices = []
        hetero_types = getattr(self.env, 'hetero_agent_types', None)
        for agent_idx in range(self.num_agents):
            nb = robot_num_bodies[agent_idx]
            # Query body names for this agent from Isaac Gym
            agent_handle = self.env.actor_handles[0][agent_idx]
            body_names = self.env.gym.get_actor_rigid_body_names(
                self.env.envs[0], agent_handle
            )
            # Get foot_name: from per-agent robot config if hetero, else from task config
            if hetero_types is not None:
                from mqe.envs.robot_registry import get_robot_config
                robot_cfg = get_robot_config(hetero_types[agent_idx])
                foot_name = robot_cfg.asset.foot_name
            else:
                foot_name = self.cfg.asset.foot_name
            feet_local = set()
            for bi, bname in enumerate(body_names):
                if foot_name in bname:
                    feet_local.add(bi)
            non_foot = torch.tensor(
                [i for i in range(nb) if i not in feet_local],
                dtype=torch.long, device=self.device
            )
            self.per_agent_non_foot_indices.append(non_foot)
            agent_label = hetero_types[agent_idx] if hetero_types else "agent"
            print(f"  Agent {agent_idx} ({agent_label}): {nb} bodies, "
                  f"feet={sorted(feet_local)}, non-foot count={len(non_foot)}")

        # Box velocity threshold: gate activates when box is moving (mass-independent)
        self.box_vel_threshold = 0.01  # m/s

        # Reward buffer for tensorboard logging
        self.reward_buffer = {
            "velocity_tracking_reward": 0,
            "angular_velocity_penalty": 0,
            "velocity_ocb_reward": 0,
            "approach_to_box_reward": 0,
            "collision_punishment": 0,
            "push_reward": 0,
            "exception_punishment": 0,
            # Metrics (not rewards, but tracked for monitoring)
            "avg_direction_error": 0,
            "avg_speed_error": 0,
            "avg_box_angular_vel": 0,
            "avg_dual_push_balance": 0,
            "avg_dual_push_gate": 0,
            "step_count": 0,
        }
        # Per-agent distance metrics (separate from reward_buffer so they don't
        # get mixed into average_step_reward calculations)
        for i in range(self.num_agents):
            self.reward_buffer[f"avg_dist_to_box_agent{i}"] = 0
            self.reward_buffer[f"avg_push_contribution_agent{i}"] = 0

        print(f"[Go1PushVelWrapper] Velocity-MAPush task initialized")
        print(f"  Speed range: {self.speed_range}")
        print(f"  Direction range: {self.direction_range}")
        print(f"  Arrow offset: {self.arrow_offset}")
        print(f"  Obs dim: {obs_dim}")
        print(f"  Reward scales: vel_track={self.velocity_tracking_scale}, "
              f"ang_vel_pen={self.angular_velocity_penalty_scale}, "
              f"vel_ocb={self.velocity_ocb_scale}, "
              f"approach={self.approach_reward_scale}, "
              f"collision={self.collision_punishment_scale}, "
              f"push={self.push_reward_scale}, "
              f"exception={self.exception_punishment_scale}")
        print(f"  Dual-push balance: alpha={self.dual_push_alpha}, "
              f"agent_masses={[m.item() for m in self.agent_masses]}, "
              f"body_offsets={self.agent_body_offsets}, "
              f"box_vel_threshold={self.box_vel_threshold} m/s")

    def _compute_agent_masses(self):
        """Compute total mass for each agent type by summing all rigid body masses.

        Returns:
            list of tensors: [mass_agent0, mass_agent1, ...] on self.device
        """
        masses = []
        env_handle = self.env.envs[0]
        for agent_idx in range(self.num_agents):
            agent_handle = self.env.actor_handles[0][agent_idx]
            body_props = self.env.gym.get_actor_rigid_body_properties(env_handle, agent_handle)
            total_mass = sum(p.mass for p in body_props)
            masses.append(torch.tensor(total_mass, device=self.device))
        return masses

    def _compute_dual_push_balance(self):
        """Compute dual-push balance using contact forces from Isaac Gym.

        Measures each agent's push contribution by reading the horizontal contact
        forces on the agent's non-foot bodies (base, hips, thighs, calves).
        When an agent pushes the box, the box exerts a reaction force on the
        agent's body — this is what we measure.

        We exclude feet (always in contact with ground) and the Z component
        (vertical ground normal). What remains is horizontal contact on the
        body — almost exclusively from the box when the agent is near it.

        Mass-weighted so heavier robots' force contributions are normalized.

        Supports heterogeneous agents with different body counts and feet indices
        via per-agent body offsets (self.agent_body_offsets) and per-agent
        non-foot index lists (self.per_agent_non_foot_indices).

        Returns:
            balance: (N,) tensor in [0, 1]. 1.0 = perfectly balanced, 0.0 = one agent idle
            contributions: (N, A) tensor — per-agent horizontal contact force (Newtons)
        """
        eps = 1e-6
        N = self.num_envs

        # contact_forces shape: (num_envs, total_bodies_in_env, 3)
        contact_forces = self.env.contact_forces

        contributions = torch.zeros(N, self.num_agents, device=self.device)

        for i in range(self.num_agents):
            # Global body indices for agent i's non-foot bodies
            body_indices = self.agent_body_offsets[i] + self.per_agent_non_foot_indices[i]

            # Contact forces on agent i's non-foot bodies: (N, num_non_foot_i, 3)
            agent_contact = contact_forces[:, body_indices, :]

            # Horizontal force magnitude per body: (N, num_non_foot_i)
            horiz_force = torch.norm(agent_contact[:, :, :2], dim=2)

            # Sum across all non-foot bodies: (N,)
            push_force_i = horiz_force.sum(dim=1)

            # Normalize by mass: force-per-kg so heavier robot's raw force advantage is divided out
            contributions[:, i] = push_force_i / self.agent_masses[i]

        # Balance ratio
        max_c = contributions.max(dim=1).values  # (N,)
        min_c = contributions.min(dim=1).values  # (N,)

        # Trigger gating when box is moving (velocity-based, mass-independent)
        npc_full = self.root_states_npc.reshape(N, self.num_npcs, -1)
        box_vel_xy = npc_full[:, 0, 7:9]
        box_speed = torch.norm(box_vel_xy, dim=1)
        box_moving = box_speed > self.box_vel_threshold

        balance = torch.ones(N, device=self.device)
        balance[box_moving] = min_c[box_moving] / (max_c[box_moving] + eps)

        return balance, contributions

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
        in world frame.

        IMPORTANT: In Isaac Gym GPU pipeline, set_actor_root_state_tensor_indexed
        is buffered — a second call before simulate() REPLACES the first. Since
        _reset_root_states() may have just queued state for ALL actors, we must
        push ALL actors here too (not just the target), preserving the pending
        reset state for box and agents while adding our target update.
        """
        env_ids = torch.arange(self.num_envs, device=self.device)

        # Read box position from all_root_states (live view into physics engine)
        box_actor_ids = self.env.npc_indices[env_ids, 0].long()  # box = NPC index 0
        box_pos_world = self.env.all_root_states[box_actor_ids, :3].clone()

        # Arrow distance scales with commanded speed (faster = farther from box)
        speed_scaled_offset = self.arrow_offset * (self.cmd_speed / self.speed_range[1])

        # Arrow position: box + speed-scaled offset in commanded direction
        arrow_x = box_pos_world[:, 0] + torch.cos(self.cmd_direction) * speed_scaled_offset
        arrow_y = box_pos_world[:, 1] + torch.sin(self.cmd_direction) * speed_scaled_offset
        arrow_z = box_pos_world[:, 2]  # Same height as box

        # Quaternion to rotate arrow to point in cmd_direction (rotation around Z)
        # Isaac Gym quaternion format: [x, y, z, w]
        half_angle = self.cmd_direction * 0.5
        qx = torch.zeros_like(self.cmd_direction)
        qy = torch.zeros_like(self.cmd_direction)
        qz = torch.sin(half_angle)
        qw = torch.cos(half_angle)

        # Update target state in all_root_states
        target_actor_ids = self.env.npc_indices[env_ids, 1].long()  # target = NPC index 1
        self.env.all_root_states[target_actor_ids, 0] = arrow_x
        self.env.all_root_states[target_actor_ids, 1] = arrow_y
        self.env.all_root_states[target_actor_ids, 2] = arrow_z
        self.env.all_root_states[target_actor_ids, 3] = qx
        self.env.all_root_states[target_actor_ids, 4] = qy
        self.env.all_root_states[target_actor_ids, 5] = qz
        self.env.all_root_states[target_actor_ids, 6] = qw
        self.env.all_root_states[target_actor_ids, 7:13] = 0  # Zero velocities

        # Push ALL actors to physics (not just target) to avoid overriding
        # any pending reset state from _reset_root_states()
        all_actor_ids = self.env.actor_indices[env_ids].view(-1)
        self.env.gym.set_actor_root_state_tensor_indexed(
            self.env.sim,
            gymtorch.unwrap_tensor(self.env.all_root_states),
            gymtorch.unwrap_tensor(all_actor_ids),
            len(all_actor_ids),
        )

        # Also sync root_states_npc copy (used by observation/reward code)
        npc_states = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)
        npc_states[:, 1, 0] = arrow_x
        npc_states[:, 1, 1] = arrow_y
        npc_states[:, 1, 2] = arrow_z
        npc_states[:, 1, 3] = qx
        npc_states[:, 1, 4] = qy
        npc_states[:, 1, 5] = qz
        npc_states[:, 1, 6] = qw
        npc_states[:, 1, 7:13] = 0
        self.root_states_npc[:] = npc_states.reshape(-1, 13)

    def _build_obs(self, base_pos, base_rpy, base_vel, box_lin_vel, box_ang_vel_z):
        """Build agent-centric observations (15 dims for 2 agents).

        Args:
            base_pos: (num_envs*num_agents, 3) — agent positions in env frame
            base_rpy: (num_envs*num_agents, 3) — agent RPY in env frame
            base_vel: (num_envs*num_agents, 3) — agent linear velocities in env frame
            box_lin_vel: (num_envs, 3) — box linear velocity in env frame
            box_ang_vel_z: (num_envs,) — box angular velocity around z-axis

        Returns:
            obs: (num_envs, num_agents, obs_dim) tensor

        Obs layout (15 dims for 2 agents):
            cmd(2):       [cos θ_local, sin θ_local]            — direction only
            box_pos(3):   [box_dx, box_dy, box_dyaw]           — agent frame
            box_vel(3):   [box_vx_local, box_vy_local, box_ωz] — agent frame + angular vel
            self_vel(2):  [self_vx, self_vy]                    — agent frame
            other_pos(3): [other_dx, other_dy, other_dyaw]      — agent frame
            other_vel(2): [other_vx_local, other_vy_local]      — agent frame
        """
        N, A = self.num_envs, self.num_agents

        # --- Reshape agent data ---
        base_pos_r = base_pos.reshape(N, A, 3)
        base_rpy_r = base_rpy.reshape(N, A, 3)
        base_vel_r = base_vel.reshape(N, A, 3)
        agent_yaw = base_rpy_r[:, :, 2]  # (N, A)
        neg_yaw = -agent_yaw
        cos_y = torch.cos(neg_yaw)  # (N, A)
        sin_y = torch.sin(neg_yaw)  # (N, A)

        # --- Get box state in env frame ---
        npc_pos = self.root_states_npc[:, :3].reshape(N, self.num_npcs, -1)
        box_pos = npc_pos[:, 0, :] - self.env.env_origins  # (N, 3)
        box_quat = self.root_states_npc.reshape(N, self.num_npcs, -1)[:, 0, 3:7]
        box_rpy = torch.stack(get_euler_xyz(box_quat), dim=1)  # (N, 3)

        # --- 1. Velocity command direction in agent's local frame (2 dims) ---
        cmd_dir_exp = self.cmd_direction.unsqueeze(1).expand(-1, A)  # (N, A)
        local_dir = cmd_dir_exp - agent_yaw

        if self.legacy_vel_obs:
            # Legacy mode: include cmd_speed for old 16-dim checkpoints
            cmd_speed_exp = self.cmd_speed.unsqueeze(1).expand(-1, A)  # (N, A)
            vel_cmd = torch.stack([
                torch.cos(local_dir),
                torch.sin(local_dir),
                cmd_speed_exp,
            ], dim=2)  # (N, A, 3)
        else:
            vel_cmd = torch.stack([
                torch.cos(local_dir),
                torch.sin(local_dir),
            ], dim=2)  # (N, A, 2)

        # --- 2. Box position relative to agent in agent's frame (3 dims) ---
        box_pos_xy_exp = box_pos[:, :2].unsqueeze(1).expand(-1, A, -1)  # (N, A, 2)
        dx = box_pos_xy_exp[:, :, 0] - base_pos_r[:, :, 0]  # (N, A)
        dy = box_pos_xy_exp[:, :, 1] - base_pos_r[:, :, 1]  # (N, A)

        rotated_box_x = dx * cos_y - dy * sin_y
        rotated_box_y = dx * sin_y + dy * cos_y
        box_yaw_exp = box_rpy[:, 2].unsqueeze(1).expand(-1, A)  # (N, A)
        rotated_box_yaw = normalize_angle(box_yaw_exp - agent_yaw)

        box_pos_obs = torch.stack([rotated_box_x, rotated_box_y, rotated_box_yaw], dim=2)  # (N, A, 3)

        # --- 3. Box velocity in agent's frame + angular velocity (3 dims) ---
        box_vx_exp = box_lin_vel[:, 0].unsqueeze(1).expand(-1, A)  # (N, A)
        box_vy_exp = box_lin_vel[:, 1].unsqueeze(1).expand(-1, A)  # (N, A)

        rotated_box_vx = box_vx_exp * cos_y - box_vy_exp * sin_y  # (N, A)
        rotated_box_vy = box_vx_exp * sin_y + box_vy_exp * cos_y  # (N, A)
        box_wz_exp = box_ang_vel_z.unsqueeze(1).expand(-1, A)  # (N, A)

        box_vel_obs = torch.stack([rotated_box_vx, rotated_box_vy, box_wz_exp], dim=2)  # (N, A, 3)

        # --- 4. Self velocity in agent's frame (2 dims) ---
        self_vx_world = base_vel_r[:, :, 0]  # (N, A)
        self_vy_world = base_vel_r[:, :, 1]  # (N, A)

        self_vx_local = self_vx_world * cos_y - self_vy_world * sin_y  # (N, A)
        self_vy_local = self_vx_world * sin_y + self_vy_world * cos_y  # (N, A)

        self_vel_obs = torch.stack([self_vx_local, self_vy_local], dim=2)  # (N, A, 2)

        # --- 5. Other agents: position (3 dims) + velocity (2 dims) each ---
        all_other_info = []
        for i in range(1, A):
            other_idx = torch.roll(torch.arange(A, device=self.device), -i)

            # Other agent positions (rolled)
            other_pos = base_pos_r[:, other_idx, :]  # (N, A, 3)
            other_rpy = base_rpy_r[:, other_idx, :]  # (N, A, 3)
            other_vel = base_vel_r[:, other_idx, :]  # (N, A, 3)

            # Position: rotate to ego frame
            o_dx = other_pos[:, :, 0] - base_pos_r[:, :, 0]  # (N, A)
            o_dy = other_pos[:, :, 1] - base_pos_r[:, :, 1]  # (N, A)

            rot_ox = o_dx * cos_y - o_dy * sin_y
            rot_oy = o_dx * sin_y + o_dy * cos_y
            rot_oyaw = normalize_angle(other_rpy[:, :, 2] - agent_yaw)

            # Velocity: rotate to ego frame
            o_vx = other_vel[:, :, 0]  # (N, A)
            o_vy = other_vel[:, :, 1]  # (N, A)
            rot_ovx = o_vx * cos_y - o_vy * sin_y
            rot_ovy = o_vx * sin_y + o_vy * cos_y

            other_info = torch.stack([rot_ox, rot_oy, rot_oyaw, rot_ovx, rot_ovy], dim=2)  # (N, A, 5)
            all_other_info.append(other_info)

        if all_other_info:
            all_other_info = torch.cat(all_other_info, dim=2)  # (N, A, 5*(A-1))
        else:
            all_other_info = torch.zeros(N, A, 0, device=self.device)

        # --- Concatenate: [cmd(3), box_pos(3), box_vel(3), self_vel(2), others(5*(A-1))] ---
        obs = torch.cat([vel_cmd, box_pos_obs, box_vel_obs, self_vel_obs, all_other_info], dim=2)

        return obs

    def reset(self, next_target_pos=None):
        """Reset all environments and sample new velocity commands."""
        obs_buf = self.env.reset()

        # Sample velocity commands for all environments
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_velocity_commands(all_ids)

        # Update arrow marker — this now pushes ALL actors (not just target)
        # to avoid overriding pending reset state in Isaac Gym GPU pipeline
        self._update_arrow_marker()

        # Build observations — velocities are zero on first step after reset
        base_pos = deepcopy(obs_buf.base_pos)
        base_rpy = deepcopy(obs_buf.base_rpy)
        base_vel = torch.zeros_like(obs_buf.base_pos)  # (N*A, 3) zeros
        box_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        box_ang_vel_z = torch.zeros(self.num_envs, device=self.device)

        obs = self._build_obs(base_pos, base_rpy, base_vel, box_lin_vel, box_ang_vel_z)

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

        # Get box velocity from NPC root states (before NaN cleaning)
        npc_full = self.root_states_npc.reshape(self.num_envs, self.num_npcs, -1)
        box_lin_vel_raw = npc_full[:, 0, 7:10].clone()   # (num_envs, 3)
        box_ang_vel_raw = npc_full[:, 0, 10:13].clone()  # (num_envs, 3)
        box_lin_vel_raw[torch.isnan(box_lin_vel_raw)] = 0
        box_lin_vel_raw[torch.isinf(box_lin_vel_raw)] = 0
        box_ang_vel_raw[torch.isnan(box_ang_vel_raw)] = 0
        box_ang_vel_raw[torch.isinf(box_ang_vel_raw)] = 0

        # Build observation
        obs = self._build_obs(base_pos, base_rpy, base_vel, box_lin_vel_raw, box_ang_vel_raw[:, 2])

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

        # --- Per-agent distance to box (monitoring metric) ---
        for i in range(self.num_agents):
            dist_i = torch.norm(box_pos[:, :2] - base_pos_r[:, i, :2], dim=1)  # (N,)
            self.reward_buffer[f"avg_dist_to_box_agent{i}"] += torch.mean(dist_i).cpu().item()

        # --- Reward computation ---
        self.reward_buffer["step_count"] += 1
        reward = torch.zeros(self.num_envs, self.num_agents, device=self.device)

        # Reuse box velocity already extracted above (cleaned of NaN/Inf)
        box_lin_vel = box_lin_vel_raw   # (num_envs, 3) — linear velocity
        box_ang_vel = box_ang_vel_raw   # (num_envs, 3) — angular velocity

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

            # Speed (tracked for monitoring only, not used in reward)
            actual_speed = actual_norm.squeeze()  # (N,)
            speed_error = torch.abs(actual_speed - self.cmd_speed)  # (N,)

            # Direction-only reward: cosine similarity with desired direction.
            # No speed magnitude penalty — agents are rewarded for pushing the box
            # in the right direction regardless of speed. This prevents the weaker
            # agent from giving up when the team can't reach commanded speed.
            vel_track_reward = (
                self.velocity_tracking_scale
                * cos_sim
            )  # (N,)

            # Dual-push balance gating: multiplicatively reduce tracking reward
            # when push effort is unbalanced (one agent freeloading)
            # tracking_reward *= alpha + (1 - alpha) * balance
            # alpha=1.0 → no gating, alpha=0.0 → full gating
            if self.dual_push_alpha < 1.0:
                balance, contribs = self._compute_dual_push_balance()
                gate = self.dual_push_alpha + (1.0 - self.dual_push_alpha) * balance
                vel_track_reward = vel_track_reward * gate
                self.reward_buffer["avg_dual_push_balance"] += torch.mean(balance).cpu().item()
                self.reward_buffer["avg_dual_push_gate"] += torch.mean(gate).cpu().item()
                for i in range(self.num_agents):
                    self.reward_buffer[f"avg_push_contribution_agent{i}"] += torch.mean(contribs[:, i]).cpu().item()

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

        # --- 3. Velocity OCB reward (positioning: be on push side of box) ---
        if self.velocity_ocb_scale != 0:
            # Push direction in world frame (unit vector)
            push_dir = torch.stack([
                torch.cos(self.cmd_direction),
                torch.sin(self.cmd_direction),
            ], dim=1)  # (N, 2)

            # For each agent: dot(agent_pos - box_pos, push_dir)
            # Negative dot = agent is behind box (correct push side) → raw_ocb < 0
            # We negate so positive = correct side
            total_ocb = torch.zeros(self.num_envs, device=self.device)
            for i in range(self.num_agents):
                agent_to_box = base_pos_r[:, i, :2] - box_pos[:, :2]  # (N, 2)
                raw_dot = torch.sum(agent_to_box * push_dir, dim=1)   # (N,)
                # raw_dot < 0 means agent is behind box (good) → negate for positive reward
                ocb_i = -raw_dot * self.velocity_ocb_scale
                total_ocb += ocb_i

            # Average across agents (continuous, teamified)
            total_ocb = total_ocb / self.num_agents

            reward[:, :] += total_ocb.unsqueeze(1).repeat(1, self.num_agents)
            self.reward_buffer["velocity_ocb_reward"] += (
                torch.sum(total_ocb).cpu().item()
            )

        # --- 4. Approach reward (approach to box) ---
        # (renumbered: was #3 before OCB insertion)
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
                (1 / (0.02 + agent_distance / 0.5)) * self.collision_punishment_scale
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
