# Velocity-MAPush: Environment Design

The RL-relevant details — task definition, observation spaces, reward design, critic architecture, and why each design choice was made.

---

## Task Definition

**Goal:** Two Go1 quadruped robots cooperatively push a box in a **commanded direction** at a **commanded speed** for the full episode duration.

Each episode samples:
- **Direction** `theta` ~ Uniform(0, 2*pi) — which way to push
- **Speed** `v` ~ Uniform(0.3, 1.0) m/s — how fast to push

There is no target position and no binary success/failure. The episode runs for the full `max_episode_length` (only physics exceptions cause early termination). The velocity command stays fixed within an episode but changes between episodes.

---

## Why This Task Exists (The Free-Rider Problem)

The original MAPush task ("push box to goal position") suffers from a **free-rider problem**: one strong agent can solo-push the box while the other learns to do nothing. A single agent applying force from one side induces both translation (toward goal) and rotation (torque). If the goal-reaching reward is large enough, the rotation penalty doesn't matter — one agent can brute-force the task.

Velocity-MAPush fixes this **structurally**:
- A single agent pushing from one side creates a large **net torque** on the box (the push force doesn't pass through the center of mass)
- Two agents pushing from complementary positions **cancel each other's torques**, producing clean linear motion
- The angular velocity penalty makes single-agent torque expensive
- The velocity tracking reward requires sustained, directional motion — not just "get close to a point"

This makes cooperation **mechanically necessary**, not just helpful.

---

## Observation Spaces

### Actor Observation: 16 dims (Agent-Centric / Egocentric)

Everything is expressed relative to the observing agent's position and heading. Each agent sees its own version of the world.

```
Dim   Component         Description
────  ────────────────  ──────────────────────────────────────────
0     cos(theta_local)  Commanded direction, rotated to agent's frame
1     sin(theta_local)  (sin component)
2     cmd_speed         Commanded speed (scalar, same for all agents)
────  ────────────────  ──────────────────────────────────────────
3     box_dx            Box X position relative to agent (agent frame)
4     box_dy            Box Y position relative to agent (agent frame)
5     box_dyaw          Box yaw relative to agent yaw (normalized [0, 2pi))
────  ────────────────  ──────────────────────────────────────────
6     box_vx_local      Box X velocity in agent's frame
7     box_vy_local      Box Y velocity in agent's frame
8     box_wz            Box angular velocity around Z-axis (not rotated — same in all frames)
────  ────────────────  ──────────────────────────────────────────
9     self_vx           Agent's own X velocity in its frame
10    self_vy           Agent's own Y velocity in its frame
────  ────────────────  ──────────────────────────────────────────
11    other_dx          Other agent's X position relative to this agent (agent frame)
12    other_dy          Other agent's Y position relative to this agent (agent frame)
13    other_dyaw        Other agent's yaw relative to this agent's yaw
────  ────────────────  ──────────────────────────────────────────
14    other_vx_local    Other agent's X velocity in this agent's frame
15    other_vy_local    Other agent's Y velocity in this agent's frame
```

**Formula:** `obs_dim = 11 + 5*(num_agents - 1) = 16` for 2 agents.

**Rotation to agent frame:**
All world-frame vectors `(wx, wy)` are rotated by `-agent_yaw`:
```
local_x = wx * cos(-yaw) - wy * sin(-yaw)
local_y = wx * sin(-yaw) + wy * cos(-yaw)
```

**Why agent-centric?**
- Each agent's policy only needs to know "where is the box relative to me, which way should I push relative to my heading." World coordinates leak information about absolute position that's irrelevant to the task.
- Egocentric observations generalize better — the same policy works regardless of where the agent spawns.

**Why include velocities?**
- **Box velocity (6-8):** The agent needs to know if the box is already moving in the right direction (to maintain it) or the wrong direction (to correct it). Without this, the agent must infer velocity from position changes across timesteps — slower to learn.
- **Box angular velocity (8):** The agent can **actively correct** box rotation, not just get penalized for it. If `box_wz` is large, the agent knows to adjust its push direction to counteract the torque.
- **Self velocity (9-10):** The agent's own velocity tells it about its current momentum. Important for smooth control — the agent can anticipate overshooting.
- **Other agent velocity (14-15):** Enables coordination. If the other agent is pushing hard from one side, this agent can compensate.

**Why NOT include:**
- **Box absolute position:** Irrelevant — the task is about velocity, not reaching a location.
- **Episode time remaining:** The velocity command is constant per episode and there's no target to reach, so time pressure doesn't change the optimal policy.
- **Previous actions:** Would add complexity without clear benefit for this task.

---

### Critic Global State: 18 dims (Box-Centered Frame)

The centralized critic (used in HAPPO's training) sees a **global state** that is expressed in the **box's reference frame** — all positions and velocities are rotated by `-box_yaw` and translated so the box is at the origin.

```
Dim   Component         Description
────  ────────────────  ──────────────────────────────────────────
0     cos(theta_box)    Commanded direction relative to box heading
1     sin(theta_box)    (sin component)
2     cmd_speed         Commanded speed (scalar)
────  ────────────────  ──────────────────────────────────────────
3     box_vx_box        Box X velocity in box's own frame
4     box_vy_box        Box Y velocity in box's own frame
5     box_wz            Box angular velocity (Z-axis)
────  ────────────────  ──────────────────────────────────────────
6     a0_dx             Agent 0 X position relative to box (box frame)
7     a0_dy             Agent 0 Y position relative to box (box frame)
8     a0_dyaw           Agent 0 yaw relative to box yaw
9     a0_vx             Agent 0 X velocity in box frame
10    a0_vy             Agent 0 Y velocity in box frame
────  ────────────────  ──────────────────────────────────────────
11    a1_dx             Agent 1 X position relative to box (box frame)
12    a1_dy             Agent 1 Y position relative to box (box frame)
13    a1_dyaw           Agent 1 yaw relative to box yaw
14    a1_vx             Agent 1 X velocity in box frame
15    a1_vy             Agent 1 Y velocity in box frame
────  ────────────────  ──────────────────────────────────────────
16    err_vx            Velocity error X: (actual - desired) in box frame
17    err_vy            Velocity error Y: (actual - desired) in box frame
```

**Formula:** `global_state_dim = 3 + 3 + 5*num_agents + 2 = 18` for 2 agents.

**Why box-centered?**
- **Translation invariance:** The same physical configuration (agents flanking the box, pushing north at 0.5 m/s) produces the same critic state regardless of where the box is on the map.
- **Rotation invariance:** The box heading is factored out. "Push forward in box frame" always looks the same to the critic, regardless of the box's absolute orientation.
- **Task-centric:** The velocity task is fundamentally about the relationship between agents, box, and commanded direction. The box-centered frame captures exactly this.

**Why the velocity error vector (dims 16-17)?**
The 2D error `(actual_v - desired_v)` in box frame gives the critic a direct, pre-computed signal about how far off the current motion is from the target. This is more informative than having the critic learn to subtract velocity vectors from the command internally. The 2D vector preserves both magnitude and direction of the error.

**Why agent velocities (dims 9-10, 14-15)?**
The critic needs to assess the **dynamics** of the situation, not just the static arrangement. Two agents at the same positions but with very different velocities represent very different value states. Agent velocities let the critic evaluate whether the current motion will be sustained or is about to change.

**Contrast with the actor observation:**
The actor sees the world from its **own** frame (egocentric — "where is the box relative to me?"). The critic sees the world from the **box's** frame (allocentric — "where are the agents relative to the box and the task goal?"). This separation is deliberate:
- The actor needs to make decisions about its own movement → egocentric is natural
- The critic needs to evaluate the global situation → box-centered captures the task structure

---

## Reward Design

All rewards are **team rewards**: computed per-environment, then broadcast identically to all agents. This ensures aligned incentives.

### 1. Velocity Tracking Reward (PRIMARY)

```
reward = 0.01 * cosine_similarity(actual_vel, desired_vel) * exp(-|speed_error|)
```

Where:
- `desired_vel = [cmd_speed * cos(cmd_dir), cmd_speed * sin(cmd_dir)]` — target velocity vector
- `actual_vel = box_lin_vel[:2]` — actual box velocity (XY plane)
- `cosine_similarity = dot(desired, actual) / (|desired| * |actual|)` — direction match, in [-1, 1]
- `speed_error = |actual_speed - cmd_speed|` — speed match

**Why this formula?**
- **Cosine similarity** rewards direction alignment without caring about speed (range [-1, 1])
- **exp(-speed_error)** rewards speed match independently (range (0, 1])
- **Product** means both must be good to get high reward. Pushing fast in the wrong direction scores poorly (cos_sim < 0). Pushing in the right direction but too slowly also scores poorly (exp(-error) < 1).

**Metrics tracked:** `avg_direction_error` (radians), `avg_speed_error` (m/s)

### 2. Angular Velocity Penalty (COOPERATION KEY)

```
penalty = -0.005 * |box_angular_vel_z|
```

**This is the structural cooperation mechanism.** A single robot pushing from one side creates net torque on the box → high `|angular_vel_z|` → large penalty. Two robots pushing from complementary positions cancel torques → low `|angular_vel_z|` → no penalty.

**Why this works:**
- Physics: Force applied off-center creates torque. Torque = r x F. A single agent can't push through the center of mass.
- Two agents on opposite sides: their torques cancel if they push with similar magnitude.
- The penalty is proportional to angular velocity, not angular position. This means the agent must actively prevent rotation, not just avoid large accumulated angles.

**Metric tracked:** `avg_box_angular_vel` (rad/s)

### 3. Approach Reward

```
reward = mean_over_agents( -(distance_to_box + 0.5)^2 * 0.00075 )
```

Encourages agents to stay close to the box. The `+0.5` offset ensures there's always some penalty even when close, preventing agents from sitting on top of the box.

### 4. Collision Punishment

```
punishment = (1 / (0.02 + agent_distance / 3)) * -0.0025
```

Penalizes agents for getting too close to each other. The `1/(0.02 + d/3)` formulation creates a sharp penalty at very close distances that falls off quickly.

### 5. Push Reward

```
reward = 0.0015  if  |box_velocity_xy| > 0.1  else  0
```

Binary reward for making the box move at all. This is a curriculum-like shaping reward that helps early training — before the agents learn directional control, they first learn that pushing the box is good.

### 6. Exception Punishment

```
penalty = -5  per  NaN/Inf detection event
```

Penalizes physics instabilities (NaN/Inf in positions or velocities). This is a safety mechanism, not a learning signal.

---

## Reward Flow

```
Per environment, per step:
    ┌─────────────────────────────┐
    │ Compute 6 reward terms      │
    │ (all per-env scalars)       │
    └──────────┬──────────────────┘
               │ broadcast to all agents
               ▼
    reward: (num_envs, num_agents)
               │ sum across agents
               ▼
    team_reward: (num_envs, 1)
               │ broadcast back to all agents
               ▼
    final_reward: (num_envs, num_agents)  ← identical for all agents in same env
```

---

## Episode Structure

```
Episode start:
    Sample cmd_direction ~ Uniform(0, 2pi)
    Sample cmd_speed ~ Uniform(0.3, 1.0)
    Place arrow marker at box_pos + [cos(theta), sin(theta)] * 2.0

Each step:
    1. Agents output actions [vx, vy, vyaw] ∈ [-1, 1], scaled by 0.5
    2. Physics steps
    3. Check for auto-resets (timeout or physics exception)
       → If env resets: sample new velocity command for that env
    4. Update arrow marker position (follows box)
    5. Build 16-dim agent-centric observations
    6. Compute 6 reward terms → team reward
    7. Return (obs, reward, done, info)

Episode end:
    Timeout at max_episode_length (no success/failure)
    Physics exception (rare)
    → New velocity command sampled for next episode
```

---

## Velocity Command Visualization

The target NPC (normally used as the goal marker in the mid task) is repurposed as a **directional arrow**. Each step it's positioned at:

```
arrow_pos = box_pos + [cos(cmd_direction), sin(cmd_direction)] * 2.0 meters
```

This moves with the box and always points in the commanded direction. When an environment resets and gets a new command, the arrow immediately jumps to the new direction.

---

## Key Design Comparisons

### Velocity-MAPush vs. Original MAPush (go1push_mid)

| Aspect | go1push_mid | go1push_vel |
|--------|-------------|-------------|
| **Goal** | Push box to target position | Push box in commanded direction at commanded speed |
| **Success** | Box within threshold of target | None (no binary success) |
| **Episode end** | Success or timeout | Timeout only |
| **Primary reward** | Distance to target | Velocity tracking (direction + speed) |
| **Cooperation signal** | Optional (one agent can solo) | Structural (angular vel penalty) |
| **Command** | Static goal position | Random direction + speed per episode |
| **Actor obs dims** | 8 (position-based) | 16 (position + velocity) |
| **Critic dims** | 9-16 (varies by mode) | 18 (box-centered) |
| **Velocities in obs** | No | Yes (box, self, other agent) |
| **Target NPC** | Goal marker | Direction arrow |

### Why Velocities Matter Here But Not in Mid-Task

In the mid-task, the goal is a **static position** — the optimal strategy depends primarily on where things are, not how fast they're moving. The policy can infer motion from position changes across timesteps.

In the velocity task, the goal **is velocity itself**. The agent must:
1. Know the current box velocity to decide whether to push harder/softer/differently
2. Know its own velocity for smooth control
3. Know the other agent's velocity to coordinate (avoid conflicting pushes)
4. Know the box angular velocity to actively counteract rotation

Without velocity observations, the agent would need to learn to estimate derivatives from position sequences — possible but much slower and noisier.

---

## Training Configuration

### MAPPO (via OpenRL)
```
Environments:    500 parallel
Total steps:     200,000,000 (200M)
Network:         2-layer MLP, hidden_size=128
Algorithm:       PPO
Box mass:        8 kg (lighter for faster learning)
```

### HAPPO (via HARL)
```
Rollout threads: 500
Total steps:     200,000,000 (200M)
Algorithm:       HAPPO
Critic:          Box-centered global state (18 dims)
```

---

## Expected Learning Dynamics

**Early training (0-50M steps):**
- Agents learn to approach the box (approach reward)
- Agents learn that pushing the box is good (push reward)
- Random pushing — direction and speed matching are poor

**Mid training (50-150M steps):**
- Agents learn directional pushing (velocity tracking reward dominates)
- Angular velocity penalty drives agents to opposite sides of the box
- Cooperation emerges: agents discover that flanking the box reduces rotation penalty

**Late training (150-200M steps):**
- Fine-tuning of speed control
- Smooth coordinated pushing at commanded velocity
- Low angular velocity, high direction accuracy

**What to monitor on tensorboard:**
- `avg_direction_error` — should decrease from ~pi/2 (random) toward 0
- `avg_speed_error` — should decrease toward 0
- `avg_box_angular_vel` — should decrease as cooperation improves
- `velocity_tracking_reward` — should increase (positive → box moving in right direction)
- `angular_velocity_penalty` — should approach 0 (less rotation → less penalty)
