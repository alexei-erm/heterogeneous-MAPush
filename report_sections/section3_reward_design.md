# Section 3 — HAPPO for Homogeneous Teams: Reward Design

*Period: Dec 14–16, 2025 | Source files: `claude_summaries/individual_rewards_testing/iter1–iter12`*

---

## 3.1 The Freeloading Problem Under Shared Rewards

When HAPPO was first trained with 2×Go1 robots using the original MAPush reward structure (7 reward components, all shared or per-agent but not credit-assigned), a consistent failure mode emerged: **one agent converged to solo-pushing while the other hovered nearby doing nothing**. Both agents received the same team reward, so the freeloading agent was reinforced for its inaction as long as its partner succeeded.

The natural hypothesis: **individualize rewards** so that only agents actively contributing receive credit. This launched a series of 12 iterations testing increasingly sophisticated reward formulations — all of which failed, ultimately revealing that reward individualization is fundamentally incompatible with HAPPO's architecture.

> **[PLOT: TensorBoard screenshot of baseline HAPPO training showing freeloading]**
> Path: Early HAPPO training run with original rewards. Show `train/episode_rewards` where both agents receive ~same reward, plus viewer screenshot showing one agent pushing and one hovering.

---

## 3.2 Reward Individualization Attempts (Iterations 1–5)

### Iteration 1: Contact-Based Reward Weighting

**Idea:** Weight the three main shared rewards (reach_target, distance_to_target, push_reward) by each agent's proximity to the box. Agents far from the box receive attenuated rewards.

**Formula:**
```
contact_weight_i = clamp(1.0 - (agent_box_dist_i - 0.8) / 0.8, 0, 1)
```
- Agent at box surface (dist ≤ 0.8m): weight = 1.0
- Agent at 1.6m: weight = 0.0

**Result:** Training destabilized immediately. When agents drifted far from the box (common early in training), they received near-zero reward, losing all learning signal. The approach_reward (which incentivizes moving toward the box) conflicted with the individualized shared rewards — agents were simultaneously told "go to the box" and "you get nothing because you're not at the box."

**Lesson:** Agents need a base reward signal to learn approach behavior. Zeroing rewards for distant agents destroys the curriculum.

### Iteration 5: All-Positive Individual + Goal Push Bonus

*Iterations 2–4 involved incremental parameter tweaks (idle penalties, direction-aware approach, collision scale adjustments) that produced similar failures. Iteration 5 represented a philosophical shift worth detailing.*

**Idea:** Separate rewards by sign — all positive rewards become individual (weighted by proximity), all negative rewards remain shared. Added a new `goal_push_bonus` (scale 0.003) rewarding box movement toward the goal.

**Result:** Agent 1's reward collapsed from −7 to −18.66 over training. The agent discovered a degenerate local minimum: **escape entirely**. By moving far from the box, it avoided collision penalties (shared) while the proximity weighting zeroed out the positive rewards it was "missing" anyway. The optimizer found it cheaper to flee than to learn pushing.

> **[PLOT: TensorBoard `train/episode_rewards` for iter5]**
> Path: `individual_rewards_testing/` run. Show the reward collapse to −18 and the divergence between Agent 0 and Agent 1 rewards. Per-agent logging was added in this iteration.

**Lesson:** Individualization creates degenerate local minima. When positive rewards are attenuated for non-contributing agents, the gradient landscape develops attractors where agents optimize for *avoiding penalties* rather than *completing the task*.

### Pattern Across Iterations 1–5

Every individualization attempt produced one of two failure modes:
1. **Training destabilization** — agents lost learning signal and couldn't recover
2. **Degenerate solutions** — agents optimized something other than pushing (hovering, escaping, staying still)

The common thread: **modifying per-agent reward magnitudes to encode credit assignment fights HAPPO's own credit assignment mechanism** (the sequential importance-weighted update), creating conflicting optimization pressures.

---

## 3.3 Discovery: EP Mode Breaks with Individual Rewards (Iteration 6)

### The Positive-Reinforcement Redesign

Iteration 6 took a fundamentally different approach: replace all punishment-based rewards with positive incentives, reasoning that agents shouldn't need to avoid bad behavior but rather be attracted to good behavior.

**New reward components:**
| Reward | Scale | Description |
|--------|-------|-------------|
| engagement_bonus | 0.02 | Agent within 1.5m of box |
| cooperation_bonus | 0.01 | Both agents within 1.5m |
| same_side_bonus | 0.02 | Both agents on the push side of box |
| blocking_penalty | −0.05 | Agent between box and goal |
| goal_push_bonus | 0.15 | Box velocity toward goal |
| directional_progress | 0.15 | Box-to-goal distance decreased (shared) |

### The Critical Bug

Training with these rewards revealed something unexpected: **both agents received nearly identical rewards** (within 0.1% of each other) despite the rewards being individually computed. Investigation traced the problem to HARL's EP (Environment Provided) critic mode.

In `on_policy_base_runner.py` (line 453):
```python
self.critic_buffer.insert(
    share_obs[:, 0],   # ← Only Agent 0's observation
    ...
    rewards[:, 0],      # ← Only Agent 0's reward
)
```

In EP mode, **the centralized critic is trained only on Agent 0's reward**. Agent 1's individually computed reward is silently discarded. The advantage estimates for Agent 1 are derived from Agent 0's value function, meaning Agent 1 receives **garbage gradient signal** that has nothing to do with its own actions.

**Evidence:**
- Critic value loss exploded: 0.01 → 0.25 (+170% increase)
- Agent rewards were identical despite individual formulas — because the critic broadcast Agent 0's values to both
- No amount of reward engineering could fix this: the learning algorithm wasn't even reading Agent 1's rewards

> **[PLOT: TensorBoard `train/value_loss` for iter6]**
> Path: `individual_rewards_testing/` run. Show the value loss explosion from 0.01 to 0.25. Annotate the EP mode bug as root cause.

**Lesson:** Before designing complex reward functions, verify that the learning algorithm actually processes them correctly. In EP mode, individual rewards are architecturally impossible.

---

## 3.4 FP Critic Mode and First Task Success (Iterations 7–9)

### The Switch to FP Mode (Iteration 7)

The EP mode bug motivated switching to FP (Feature Pruned) critic mode, where each agent maintains its own value function and receives advantages computed from its own returns. This should have solved the reward routing problem.

### Gated Shared Rewards (Iteration 8)

**Idea:** Multiply all shared rewards by `min(engagement_agent0, engagement_agent1)`. If either agent is far from the box, *both* get zero reward — creating "peer pressure."

**Result:** Training was unstable. The reward landscape became non-stationary because the gate value fluctuated as agents moved. Value loss oscillated (0.03 → 0.16 → 0.05) as the critic struggled to predict rewards that depended on both agents' positions.

**Fix discovered:** Average rewards across agents before returning from the wrapper to smooth the signal. This helped stability but diluted individual accountability.

### First Success: 18% (Iteration 9)

**Configuration:** FP mode + original MAPush reward structure + contact-weighted push_reward and reach_target.

**Contact weighting:**
```
contact_weight_i = clamp(1.0 - (dist_i - threshold) / threshold, 0, 1)
```

**Result: 18% success rate at 100M steps** — the first time HAPPO successfully completed the pushing task. Both agents learned to approach the box, though they hovered without actively pushing in many episodes.

> **[PLOT: Success rate curve for iter9]**
> Path: `individual_rewards_testing/` test results. Show success rate progression from 0% to 18% over 100M steps. Compare with MAPPO baseline (~90%) to contextualize the gap.

This was a proof-of-concept that HAPPO *could* learn the task, but the 18% rate (vs MAPPO's ~90%) indicated fundamental issues remained.

---

## 3.5 Collision Penalty and Learning Rate Fixes (Iterations 10–11)

### Iteration 10: The Collision Penalty Problem

Quantitative analysis of the collision penalty revealed why agents avoided each other:

```
Original collision punishment scale: −0.0025
At 0.5m inter-agent distance: penalty = −0.0134 per step
Over one episode (~1000 steps at this distance): −13.4 total
Typical positive episode reward: ~25–30

→ Collision penalty at moderate distance ≈ 50% of total episode reward
```

The penalty function was a smooth curve (not a hard threshold), meaning agents experienced significant negative reward well before actual collision. **Agents rationally learned to stay apart**, which prevented cooperative pushing.

**Fix:** Reduced collision_punishment_scale 5×: −0.0025 → −0.0005. At 0.5m distance, penalty dropped to −0.0027/step — still discouraging collision but no longer dominating the reward landscape.

> **[PLOT: Collision penalty magnitude vs inter-agent distance, before and after]**
> Either a hand-drawn figure or computed: plot `penalty = scale * exp(-dist^2 / sigma^2)` for both scale values, annotating the 0.5m distance point and the typical episode reward magnitude.

### Iteration 11: Learning Rate Mismatch

HAPPO's default learning rate (0.005) was 10× higher than MAPPO's (0.0005). With the higher LR, the critic value loss *diverged* over training (0.10 → 0.19) instead of converging (MAPPO: 0.53 → 0.03). The critic was overshooting on each update.

**Fix:** Matched MAPPO's LR: actor and critic both set to 0.0005.

These two fixes (collision scale + LR) were necessary but not sufficient — they removed obstacles to learning but didn't address the fundamental architectural issue.

---

## 3.6 Architectural Insight: HAPPO Requires Team Rewards (Iteration 12)

### The Root Cause

After 11 iterations of reward engineering, the root cause was identified: **HAPPO's credit assignment comes from its sequential importance-weighted update scheme, not from reward decomposition.**

In HAPPO's update procedure:
1. Agent 0 updates first with the full advantage signal
2. Agent 1 updates second with advantages *reweighted by Agent 0's importance ratio*
3. This reweighting is what assigns credit — Agent 1's update accounts for how Agent 0's policy has already changed

When rewards are individualized:
- Each agent's advantage is computed from different reward streams
- The importance reweighting loses its theoretical meaning
- The sequential update becomes a source of noise rather than credit assignment

### The Correct Configuration

```
Critic mode:       EP (single centralized critic, global state)
Reward structure:  Team rewards — sum per-agent components, broadcast identical
                   reward to all agents
Collision scale:   −0.0005 (5× reduced from default)
Learning rates:    0.0005 (matched to MAPPO)
Actor networks:    Separate per agent (NOT shared — this is the key HAPPO feature)
```

The ONLY difference between this correct HAPPO setup and MAPPO should be:
1. **Separate actor networks** (HAPPO) vs shared parameters (MAPPO)
2. **Sequential importance-weighted updates** (HAPPO) vs simultaneous updates (MAPPO)

Everything else — critic mode, reward structure, state representation — should be identical.

### Team Reward Implementation

```python
# In the environment wrapper:
team_reward = reward.sum(dim=1, keepdim=True)       # Sum per-agent contributions
reward = team_reward.expand(-1, self.num_agents)     # Broadcast to all agents
```

> **[PLOT: Comparison of critic value_loss trajectories]**
> Three curves on one plot:
> 1. EP + individual rewards (iter6): diverging, 0.01 → 0.25
> 2. FP + individual rewards (iter9): oscillating, 0.03 → 0.16
> 3. EP + team rewards (iter12 / expected): converging like MAPPO
> Path: TensorBoard data from respective runs.

---

## 3.7 Section Summary

Twelve iterations of reward engineering revealed that **reward individualization is the wrong approach for HAPPO**. The progression:

| Phase | Iterations | Approach | Best SR | Failure Mode |
|-------|-----------|----------|---------|--------------|
| Individualization | 1–5 | Weight rewards by agent contribution | 0% | Destabilization, degenerate local minima |
| Bug discovery | 6 | Positive reinforcement redesign | 0% | EP mode discards Agent 1's reward |
| FP mode | 7–9 | Per-agent critics + individual rewards | 18% | Hovering, insufficient cooperation |
| Hyperparameter fix | 10–11 | Collision 5×↓, LR 10×↓ | ~18% | Removed obstacles, didn't fix root cause |
| Root cause fix | 12 | EP mode + team rewards | — | Correct architecture identified |

**Key findings:**
1. HAPPO's sequential importance-weighted update IS the credit assignment mechanism — external reward decomposition fights it
2. EP mode with team rewards is mandatory; FP mode with individual rewards violates theoretical assumptions
3. Collision penalty magnitude must be carefully calibrated — at original scale, avoidance was the rational policy
4. HAPPO's default learning rate (0.005) is too high for this environment; matching MAPPO's (0.0005) is necessary

The correct HAPPO reward configuration was carried forward into Section 4 (critic representation search), where the remaining performance gap (18% → 85%) was closed through critic input design rather than reward engineering.
