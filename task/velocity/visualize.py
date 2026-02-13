"""Quick visualization of the Velocity-MAPush environment.

Runs 1 environment with a fixed random-walk policy (no trained model needed).
Displays the viewer and prints per-step reward breakdowns to the terminal.

Usage:
    cd /home/gvlab/new-universal-MAPush
    python task/velocity/visualize.py
"""
import isaacgym  # must import before torch
import torch
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mqe.envs.utils import make_mqe_env, custom_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of steps to run")
    parser.add_argument("--action_mode", type=str, default="forward",
                        choices=["forward", "random", "zero"],
                        help="Fixed policy: forward=both push +x, random=uniform, zero=stand still")
    parser.add_argument("--headless", action="store_true", default=False)
    cli_args = parser.parse_args()

    # Build a minimal args namespace for make_mqe_env
    from isaacgym import gymapi
    args = argparse.Namespace()
    args.task = "go1push_vel"
    args.num_envs = cli_args.num_envs
    args.seed = 42
    args.headless = cli_args.headless
    args.record_video = False
    args.rl_device = "cuda:0"
    args.sim_device = "cuda:0"
    args.device = "cuda"
    args.compute_device_id = 0
    args.sim_device_type = "cuda"
    args.use_gpu_pipeline = True
    args.physics_engine = gymapi.SIM_PHYSX
    args.use_gpu = True
    args.subscenes = 0
    args.num_threads = 0

    env, env_cfg = make_mqe_env(
        args.task, args,
        custom_cfg=custom_cfg(args, agent0="go1", agent1="go1"),
    )

    print(f"\n{'='*70}")
    print(f"VELOCITY-MAPUSH VISUALIZATION")
    print(f"{'='*70}")
    print(f"  Num envs:        {env.num_envs}")
    print(f"  Num agents:      {env.num_agents}")
    print(f"  Obs shape:       {env.observation_space.shape}")
    print(f"  Action shape:    {env.action_space.shape}")
    print(f"  Action mode:     {cli_args.action_mode}")
    print(f"  Steps to run:    {cli_args.steps}")
    print(f"{'='*70}\n")

    obs = env.reset()

    # Debug: print object positions to verify they exist
    base_env = env.env  # underlying legged_robot_field
    npc_states = base_env.all_root_states.view(base_env.num_envs, -1, 13)
    agent_states = npc_states[:, :base_env.num_agents, :]
    npc_only = npc_states[:, base_env.num_agents:, :]
    print(f"  env_origins[0]:  {base_env.env_origins[0].cpu().tolist()}")
    print(f"  Agent 0 pos:     {agent_states[0, 0, :3].cpu().tolist()}")
    print(f"  Agent 1 pos:     {agent_states[0, 1, :3].cpu().tolist()}")
    print(f"  Box (NPC 0) pos: {npc_only[0, 0, :3].cpu().tolist()}")
    print(f"  Target (NPC 1):  {npc_only[0, 1, :3].cpu().tolist()}")
    print(f"  num_actors/env:  {base_env.num_agents + base_env.num_npcs}")
    print(f"  npc_indices[0]:  {base_env.npc_indices[0].cpu().tolist()}")

    # Reposition camera to look at the box from a good angle
    box_pos = npc_only[0, 0, :3].cpu()
    cam_target = gymapi.Vec3(float(box_pos[0]), float(box_pos[1]), 0.3)
    cam_pos = gymapi.Vec3(float(box_pos[0]) - 5, float(box_pos[1]) - 5, 8.0)
    if base_env.viewer:
        base_env.gym.viewer_camera_look_at(base_env.viewer, None, cam_pos, cam_target)
    print(f"  Camera repositioned: pos=[{cam_pos.x:.1f}, {cam_pos.y:.1f}, {cam_pos.z:.1f}] "
          f"lookat=[{cam_target.x:.1f}, {cam_target.y:.1f}, {cam_target.z:.1f}]")
    print()
    print(f"  cmd_direction:   {env.cmd_direction[0].item():.3f} rad "
          f"({torch.rad2deg(env.cmd_direction[0]).item():.1f} deg)")
    print(f"  cmd_speed:       {env.cmd_speed[0].item():.3f} m/s")
    print()

    N = env.num_envs
    A = env.num_agents

    # Column header
    print(f"{'step':>5}  {'vel_track':>10}  {'ang_pen':>10}  {'vel_ocb':>10}  "
          f"{'approach':>10}  {'collision':>10}  {'push':>10}  {'exception':>10}  "
          f"{'TOTAL':>10}  {'dir_err':>8}  {'spd_err':>8}  {'box_wz':>8}")
    print("-" * 140)

    prev_rb = {k: v for k, v in env.reward_buffer.items()}

    for step in range(1, cli_args.steps + 1):
        # Fixed policy
        if cli_args.action_mode == "forward":
            action = torch.zeros(N, A, 3, device="cuda")
            action[:, :, 0] = 1.0   # full forward velocity
        elif cli_args.action_mode == "random":
            action = torch.rand(N, A, 3, device="cuda") * 2 - 1  # uniform [-1, 1]
        else:  # zero
            action = torch.zeros(N, A, 3, device="cuda")

        obs, reward, done, info = env.step(action)

        # Extract per-step reward deltas from the cumulative buffer
        rb = env.reward_buffer
        sc = rb["step_count"]
        if sc == 0:
            continue

        d_vel_track = rb["velocity_tracking_reward"] - prev_rb["velocity_tracking_reward"]
        d_ang_pen = rb["angular_velocity_penalty"] - prev_rb["angular_velocity_penalty"]
        d_vel_ocb = rb["velocity_ocb_reward"] - prev_rb["velocity_ocb_reward"]
        d_approach = rb["approach_to_box_reward"] - prev_rb["approach_to_box_reward"]
        d_collision = rb["collision_punishment"] - prev_rb["collision_punishment"]
        d_push = rb["push_reward"] - prev_rb["push_reward"]
        d_exception = rb["exception_punishment"] - prev_rb["exception_punishment"]
        d_total = reward.sum().item()

        d_dir_err = rb["avg_direction_error"] - prev_rb["avg_direction_error"]
        d_spd_err = rb["avg_speed_error"] - prev_rb["avg_speed_error"]
        d_box_wz = rb["avg_box_angular_vel"] - prev_rb["avg_box_angular_vel"]

        # Print every 10 steps (or first 20)
        if step <= 20 or step % 10 == 0:
            print(f"{step:5d}  {d_vel_track:10.5f}  {d_ang_pen:10.5f}  {d_vel_ocb:10.5f}  "
                  f"{d_approach:10.5f}  {d_collision:10.5f}  {d_push:10.5f}  {d_exception:10.5f}  "
                  f"{d_total:10.5f}  {d_dir_err:8.4f}  {d_spd_err:8.4f}  {d_box_wz:8.4f}")

        prev_rb = {k: v for k, v in rb.items()}

        # Print termination diagnostics and new command on reset
        if done.any():
            base_env = env.env  # underlying legged_robot_field
            reasons = []
            if hasattr(base_env, 'r_term_buff') and base_env.r_term_buff.any():
                reasons.append("ROLL")
            if hasattr(base_env, 'p_term_buff') and base_env.p_term_buff.any():
                reasons.append("PITCH")
            if hasattr(base_env, 'z_wave_term_buff') and base_env.z_wave_term_buff.any():
                reasons.append("Z_WAVE")
            if hasattr(base_env, 'collision_term_buff') and base_env.collision_term_buff.any():
                reasons.append("COLLISION")
            if hasattr(base_env, 'far_away_term_buff') and base_env.far_away_term_buff.any():
                reasons.append("FAR_AWAY")
            if hasattr(base_env, 'out_of_area_term_buff') and base_env.out_of_area_term_buff.any():
                reasons.append("OUT_OF_AREA")
            if base_env.time_out_buf.any():
                reasons.append("TIMEOUT")
            if env.value_exception_buf.any():
                reasons.append("VALUE_EXCEPTION(NaN)")
            reason_str = ", ".join(reasons) if reasons else "UNKNOWN"
            print(f"\n  [RESET reason={reason_str}] New cmd_direction: {env.cmd_direction[0].item():.3f} rad "
                  f"({torch.rad2deg(env.cmd_direction[0]).item():.1f} deg), "
                  f"cmd_speed: {env.cmd_speed[0].item():.3f} m/s\n")

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY after {cli_args.steps} steps")
    print(f"{'='*70}")
    rb = env.reward_buffer
    sc = rb["step_count"] if rb["step_count"] > 0 else 1
    print(f"  Avg velocity tracking reward:  {rb['velocity_tracking_reward'] / (sc * N):10.6f}")
    print(f"  Avg angular velocity penalty:  {rb['angular_velocity_penalty'] / (sc * N):10.6f}")
    print(f"  Avg velocity OCB reward:       {rb['velocity_ocb_reward'] / (sc * N):10.6f}")
    print(f"  Avg approach reward:           {rb['approach_to_box_reward'] / (sc * N):10.6f}")
    print(f"  Avg collision punishment:      {rb['collision_punishment'] / (sc * N):10.6f}")
    print(f"  Avg push reward:               {rb['push_reward'] / (sc * N):10.6f}")
    print(f"  Avg exception punishment:      {rb['exception_punishment'] / (sc * N):10.6f}")
    print(f"  ---")
    print(f"  Avg direction error:           {rb['avg_direction_error'] / sc:8.4f} rad "
          f"({rb['avg_direction_error'] / sc * 57.2958:6.2f} deg)")
    print(f"  Avg speed error:               {rb['avg_speed_error'] / sc:8.4f} m/s")
    print(f"  Avg box angular velocity:      {rb['avg_box_angular_vel'] / sc:8.4f} rad/s")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
