# Latest Changes Summary

**Period:** 2026-02-13 to 2026-04-03
**Branches:** `velocity-MAPush`, `high-level-happo-task_implementation`

This document summarizes all significant code changes made across 13 development sessions. It replaces the individual session files (`session_2026-02-13.md` through `session_2026-04-03.md`).

---

## A. Contact-Force Gating (Anti-Freeloading)

**Problem:** In heterogeneous teams (e.g., Go1 12kg + Anymal C 50kg), the weaker agent freeloads because the stronger agent can solve the task solo. Team rewards give the freeloader credit for doing nothing.

**Solution:** Gate pushing-related rewards by a contact-force balance ratio measured directly from Isaac Gym's `net_contact_force_tensor`.

**Mechanism:**
```
For each agent i:
    raw_force_i = sum(|horizontal XY contact force|) on non-foot bodies
    contribution_i = raw_force_i / agent_mass_i        (N/kg, mass-normalized)

balance = min(contributions) / max(contributions)       [0, 1]
gate = alpha + (1 - alpha) * balance                    [alpha, 1.0]

gated_rewards *= gate
```

Activates only when `box_speed > 0.01 m/s` (box-velocity trigger).

**Gated rewards:**
- Mid task (`go1_push_mid`): `push_reward`, `distance_to_target_reward`
- Velocity task (`go1_push_vel`): `velocity_tracking_reward`

**Flags:** `--contact_force_gating True --contact_force_gating_alpha 0.3`

**Evolution:**
1. Initial: `force * mass_fraction` — WRONG, amplified heavy robot's contribution
2. Fix (2026-03-07): `force / agent_mass` — correct, equal effort gives balance ≈ 1.0
3. Threshold change (2026-03-12): force threshold (0.1 N/kg) replaced with box-velocity trigger (0.01 m/s)
4. Flag fix (2026-03-18): velocity wrapper was ignoring CLI flags, using hardcoded `dual_push_balance_alpha` — unified to match mid wrapper's `contact_force_gating` / `contact_force_gating_alpha` pattern

**TensorBoard metrics:** `avg_dual_push_balance`, `avg_dual_push_gate`, `avg_push_contribution_agent{i}`

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | `_compute_agent_masses()`, `_compute_dual_push_balance()`, body offset precomputation, gating logic, TB metrics |
| `mqe/envs/wrappers/go1_push_vel_wrapper.py` | Same as mid wrapper, plus unified flag naming |
| `HARL/harl_mapush/train.py` | `--contact_force_gating`, `--contact_force_gating_alpha` CLI flags |
| `HARL/harl/envs/mapush/mapush_env.py` | Extracts and passes flags to `custom_cfg()` |
| `mqe/envs/utils.py` | `custom_cfg()` accepts and sets gating params on config |

---

## B. Upper-Level HAPPO Planner Support

**Problem:** The upper-level task (`go1push_upper`) only supported OpenRL PPOModule format for frozen mid-level policies. Needed HAPPO actor support for the hierarchical pipeline: upper (HARL) → mid (frozen HAPPO) → locomotion.

**Architecture:**
```
Upper level (HARL HAPPO, 1 agent)
  → 2D sub-goal [x, y]
    Mid level (frozen HAPPO actors, 2 per-robot)
      → 3-DOF velocity [vx, vy, vyaw] per robot
        Locomotion (Walk-These-Ways / legged_gym)
          → joint torques
```

**Key details:**
- HARL sees `n_agents=1` (single planner). The upper wrapper internally handles 2 robots via frozen mid-level actors.
- 26-dim observation (global: both robots + box + obstacles + trajectory waypoint)
- 2-dim action (sub-goal x, y coordinates)
- Mid-level actors loaded as `StochasticPolicy` from `harl.models.policy_models.stochastic_policy` with matching architecture (hidden_sizes=[256,256], relu, orthogonal init)
- NaN sanitization added at 4 points to handle physics corruption cascades (degenerate quaternions, stale root states)

**Flags:** `--mid_level_checkpoint <path> --mid_level_format happo`

**Bug fix:** `command_observation_space` was 9 dims, actual tensor is 8 dims (target_xy(2) + box_xy(2) + box_yaw(1) + other_agent_xy(2) + other_yaw(1)).

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_upper_wrapper.py` | HAPPO mid-level loading (`_prepare_happo_command_policy()`), inference (`_happo_mid_level_act()`), format dispatch, lazy OpenRL import, obs dim fix |
| `mqe/envs/configs/go1_push_upper_config.py` | Added `mid_level_format = "openrl"` default |
| `mqe/envs/utils.py` | `custom_cfg()` passes `mid_level_checkpoint` and `mid_level_format` |
| `HARL/harl_mapush/train.py` | `go1push_upper` in `--task` choices, `--mid_level_checkpoint`, `--mid_level_format` |
| `HARL/harl/envs/mapush/mapush_env.py` | Upper task detection, space overrides (obs=26d, action=2d), `_step_upper()`, `_reset_upper()`, NaN sanitization |
| `HARL/harl_mapush/test.py` | Upper task calculator/viewer/batch modes, single-agent checkpoint detection, NaN sanitization |

---

## C. Direction-Only Velocity Tracking Reward

**Problem:** The velocity tracking reward `cos_sim * exp(-speed_error)` uniformly suppressed reward for heterogeneous teams that physically couldn't reach the commanded speed. The weaker agent had no gradient to learn from and gave up entirely.

**Fix:** Removed `exp(-speed_error)` from the reward. Now `vel_track_reward = scale * cos_sim` (direction only). Speed error is still computed and logged to TensorBoard for monitoring but does not affect reward.

**Impact:** Any contribution to pushing the box in the correct direction is now rewarded regardless of speed. The weaker agent always has gradient to improve.

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_vel_wrapper.py` | Reward formula change, speed error kept as monitoring metric |
| `mqe/envs/configs/go1_push_vel_config.py` | `velocity_tracking_scale` adjustments through iterations |

---

## D. Isaac Gym Buffered State Fix

**Problem:** In the velocity task, the box disappeared after resets. Root cause: `set_actor_root_state_tensor_indexed` is a **buffered** command in Isaac Gym's GPU pipeline. A second call before `simulate()` **replaces** (not merges with) the first. The wrapper's `_update_arrow_marker()` was overwriting the pending reset states.

**Fix:** Changed `_update_arrow_marker()` to include ALL actors in its `set_actor_root_state_tensor_indexed` call, preserving pending reset states from `_reset_root_states()`.

**Related fix:** `root_states_npc` is a **COPY** of `all_root_states` (non-contiguous slice + reshape creates a copy). Arrow marker was reading stale box positions. Fixed to read from `all_root_states` directly.

**Rule:** Any wrapper that modifies actor states after `_reset_root_states()` must include ALL actors in its indexed state tensor call, not just the ones it changed.

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_vel_wrapper.py` | All-actor push in `_update_arrow_marker()`, read from `all_root_states` |

---

## E. Box Mass Randomization

**Problem:** Default box mass (4kg) is trivially easy for a single robot. Need heavier boxes to force collaboration, and mass randomization for robustness.

**Features:**
- `--box_mass N` — fixed mass override (replaces URDF default)
- `--box_mass_range MIN MAX` — uniform random mass per env at creation time (takes priority over `--box_mass`)
- Inertia tensor recalculated correctly: `I = (1/12) * m * (a² + b²)` using configurable `npc_box_dimensions`

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/npc/go1_object.py` | Configurable `npc_box_dimensions` for inertia, `npc_mass_range` uniform sampling per env |
| `HARL/harl_mapush/train.py` | `--box_mass`, `--box_mass_range` CLI flags |
| `HARL/harl_mapush/test.py` | Same flags for testing |

---

## F. Legacy Velocity Obs Flag

**Problem:** After removing `cmd_speed` from velocity task observations (16→15 dims actor, 18→15 dims critic), old checkpoints trained with 16-dim obs can't be loaded.

**Fix:** `--legacy_vel_obs` flag restores the old observation structure for testing old checkpoints.

| Mode | Actor dims | Critic dims | Velocity command |
|------|-----------|-------------|-----------------|
| Default (new) | 15 | 15 | `[cos, sin]` (direction only) |
| Legacy | 16 | 18 | `[cos, sin, speed]` + extra critic dims |

**Files modified:**
| File | Changes |
|------|---------|
| `mqe/envs/wrappers/go1_push_vel_wrapper.py` | Conditional obs construction in `_build_obs()` |
| `HARL/harl/envs/mapush/mapush_env.py` | `global_state_dim` adjustment, conditional `_construct_vel_global_state()` |
| `mqe/envs/utils.py` | `custom_cfg()` accepts `legacy_vel_obs` kwarg |
| `HARL/harl_mapush/train.py` | `--legacy_vel_obs` flag |
| `HARL/harl_mapush/test.py` | `--legacy_vel_obs` flag in calculator and viewer modes |

---

## G. MAPPO Checkpoint Save Frequency Fix

**Problem:** `openrl_ws/train.py` had `CheckpointCallback(save_freq=10000000)`, but `save_freq` counts **`n_calls`** (one per `env.step()` call), NOT timesteps. With `num_envs=500`, max `n_calls` = 400,000 for 200M training — never reaches 10M threshold. No checkpoints were saved.

**Fix:** Changed `save_freq` to `20000` (= 10M timesteps / 500 envs). Produces checkpoints at 10M, 20M, ..., 200M timesteps.

**Files modified:**
| File | Changes |
|------|---------|
| `openrl_ws/train.py` | `save_freq` 10000000 → 20000 |

---

## H. OpenRL Wrapper Bug Fix

**Problem:** `mqe_openrl_wrapper.batch_rewards()` used `self.num_envs` which doesn't exist on the wrapper class.

**Fix:** Changed to `self.env.num_envs`.

**Files modified:**
| File | Changes |
|------|---------|
| `openrl_ws/utils.py` | `self.num_envs` → `self.env.num_envs` |

---

## Files Created

| File | Purpose |
|------|---------|
| `resources/objects/arrow.urdf` | Green arrow marker for velocity task direction visualization |
| `resources/objects/cuboid/VelBox.urdf` | 1.8m × 1.8m × 0.75m blue box for velocity task |
| `plot_success.py` | Success rate plotting from HAPPO/MAPPO test result files |
| `plot_tb.py` | TensorBoard metrics batch plotting with `--plot_agent_together` flag |
| `tables/baseline_mappo_rewards.tex` | LaTeX reward summary table for report |
| `report_sections/` | Report LaTeX/Markdown drafts (section 3 reward design, critic obs analysis) |
| `test_all_checkpoints_velocity.sh` | Standalone MAPPO checkpoint batch testing script |
