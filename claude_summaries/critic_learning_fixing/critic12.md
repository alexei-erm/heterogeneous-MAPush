# CRITIC12: Three-Tier Cooperation Bonuses

> **Date:** December 28, 2025
> **Updates:**
> - Dec 28: v1 - 10x Scale Increase (bonuses too weak)
> - Dec 29: v2 - Relaxed Thresholds (Tier 2 & 3 not triggering)
> - Dec 30: v3 - Asymmetric OCB (wrong side penalty 2x stronger)
> - Dec 31: v4 - Symmetric OCB + increased scale (0.004 → 0.01)
> - Dec 31: v5 - **Joint Binary OCB ±0.004 + ALL 3 coop bonuses** ⭐ **BEST: 79.8% SR, OCB +0.009**
> - Dec 31: v5b - Disabled cooperation bonuses, testing joint OCB alone
> - Jan 2: v5c - **Joint Binary OCB ±0.01** (2.5x stronger signal) - 44% SR, too strong penalty
> - Jan 2: v5d - **Asymmetric OCB** (+0.01/-0.004) - strong reward, weak penalty - FAILED
> - Jan 2: v6 - **Re-enabled cooperation bonuses** (minus Dual Engagement) - 80% SR but pushing wrong direction
> - Jan 5: v7 - **Directional Push Reward** - rewards box velocity TOWARD goal (scale 0.003)
> - Jan 5: v7b - **Increased directional push** 0.003 → 0.01 (stronger punishment for wrong direction)
> **Goal:** Prevent freeloading + ensure pushing in correct direction
> **Status:** TRAINING v7b (stronger directional push + asymmetric OCB + 2-tier coop bonuses)
> **Flag:** `--cooperation_rewards True`
> **Current Scales:** Original MAPPO values, OCB asymmetric +0.01/-0.004, goal_push_bonus 0.01
> **Cooperation Bonuses:** Synchronized Contact (+0.008) + Bilateral Push (+0.01)
> **OCB:** Joint binary asymmetric: +0.01 (both correct), -0.004 (any wrong)
> **NEW:** Directional push reward: `0.01 * dot(box_velocity, goal_direction)` - continuous, +/- based on direction

---

## Trained Models & Version History

| Run Directory | Version | Steps | SR | Key Config |
|---------------|---------|-------|-----|------------|
| `seed-00001-2025-12-28-17-53-36` | v1 | 90M | 29% | 10x coop bonuses, strict thresholds (1.5m, 0.8m, 0.1), OCB sym 0.004 |
| `seed-00007-2025-12-28-23-01-30` | v2 | 200M | 81% | Relaxed thresholds (2.0m, 1.2m, 0.03), OCB sym 0.004 |
| `seed-00007-2025-12-29-16-14-47` | v2 | 150M | 85% | *(continued below)* |
| `seed-00007-2025-12-30-03-08-06` | v2 | 350M | 85% | *(continued from above, Ctrl+C resume)* |
| `seed-00007-2025-12-30-17-41-37` | v3 | 200M | 70% | Asymmetric OCB (+0.004/-0.008), thresholds unchanged |
| `seed-00007-2025-12-31-15-21-53` | v4 | 25M | 14% | Symmetric per-agent OCB 0.01, **OCB went negative!** |
| `seed-00007-2025-12-31-17-20-22` | **v5** | **250M** | **79.8%** | **Joint binary OCB ±0.004 + ALL 3 cooperation bonuses - OCB WENT POSITIVE +0.009!** |
| `seed-00007-2026-01-01-10-54-10` | v5b | 200M | 82.7% | Joint binary OCB ±0.004, cooperation bonuses DISABLED |
| `seed-00007-2026-01-02-01-26-14` | v5c | 195M | 44.5% | Joint binary OCB ±0.01 - **too strong penalty, worse than v5b** |
| `seed-00007-2026-01-02-*` | v5d | - | FAIL | Asymmetric OCB +0.01/-0.004 - still bad without coop bonuses |
| `seed-00007-2026-01-02-17-07-58` | v6 | 200M | 80.3% | Asymmetric OCB + 2-tier coop bonuses - **pushing wrong direction** |
| `seed-00007-2026-01-05-02-09-31` | v7 | 150M | 79.2% | v6 + Directional Push 0.003 - **still pushing wrong direction often** |
| `seed-00007-2026-01-05-*` | **v7b** | - | - | **Increased directional push 0.003 → 0.01 (3.3x stronger penalty)** |

### Version Details

---

## ⭐ **V5 BEST RUN - DETAILED BREAKDOWN** (seed-00007-2025-12-31-17-20-22)

**Training:** 250.5M steps | **Success Rate:** 79.8% | **Date:** Dec 31, 2025 17:20

### Flags & Config (from config.json)
```
--cooperation_rewards True
--use_concat_agent_observations_critic True
--individualized_rewards False
--shared_gated_rewards False
--seed 7
```

### Complete Reward Breakdown (from TensorBoard logs)

| Reward Component | Start | End | Change |
|-----------------|-------|-----|--------|
| **average_step_reward** | -0.0146 | **+0.0226** | +255.6% |
| **success_rate** | 0.000 | **0.798** | - |
| **ocb_reward** | +0.00002 | **+0.00916** | +55,417% ✅ POSITIVE! |
| dual_engagement_bonus | 0.00166 | 0.00267 | +60.8% |
| synchronized_contact_bonus | 0.00061 | 0.00377 | +514.4% |
| bilateral_push_bonus | 0.00018 | 0.00120 | +571.2% |
| distance_to_target_reward | -0.00724 | -0.00500 | +30.9% |
| approach_to_box_reward | -0.00788 | -0.00473 | +39.9% |
| push_reward | 0.00001 | 0.00092 | +12,133% |
| reach_target_reward | 0.000 | 0.01822 | - |
| collision_punishment | -0.00148 | -0.00238 | -61.0% |
| exception_punishment | -0.00044 | -0.00118 | -168.2% |
| goal_push_bonus | 0.000 | 0.000 | (not implemented) |

### Key Success Factors

1. **OCB Positioning Success:** OCB went POSITIVE (+0.00916), indicating agents learned correct positioning (69% of time both on correct side)
2. **All 3 Cooperation Bonuses Active:**
   - Dual Engagement: +0.00267/step (both within 2.0m)
   - Synchronized Contact: +0.00377/step (both within 1.2m)
   - Bilateral Push: +0.00120/step (both pushing toward goal)
   - **Total cooperation bonus: +0.00764/step**
3. **Symmetric OCB:** ±0.004 (not asymmetric)
4. **Joint Binary OCB:** Both must be correct for positive reward

### Why This Worked

- **Dual Engagement** encouraged agents to stay near box (exploration)
- This exploration → discovered correct positioning → OCB turned positive
- Cooperation bonuses provided dense positive signal
- Symmetric OCB didn't discourage exploration with overly strong penalty

---

- **v1**: 10x cooperation bonuses (0.003, 0.008, 0.01/step), strict thresholds (1.5m, 0.8m, 0.1 vel), symmetric OCB 0.004
- **v2**: Relaxed thresholds → dual_engagement: 2.0m, contact: 1.2m, push_vel: 0.03 (Tier 2&3 now trigger)
- **v3**: Asymmetric OCB → wrong side penalty 2x stronger (+0.004/-0.008)
- **v4**: Reverted to symmetric OCB, increased scale 0.004 → 0.01 (2.5x stronger positioning signal)
- **v5**: **Joint binary OCB** → ±0.004 symmetric, BOTH must be correct for positive. **ALL 3 cooperation bonuses active** (dual engagement + sync contact + bilateral push). This is THE BEST RUN - OCB went positive!
- **v5b**: Disabled cooperation bonuses, testing joint OCB fix alone with original MAPPO reward scales
- **v5c**: **Increased OCB to ±0.01** (2.5x stronger signal), joint binary, coop bonuses disabled. **FAILED: 44.5% SR** - strong penalty discouraged exploration, agents stayed far from box
- **v5d**: **Asymmetric OCB** → +0.01 (both correct), -0.004 (any wrong). Coop bonuses still disabled. **FAILED** - OCB alone not enough
- **v6**: **Re-enabled cooperation bonuses** (2-tier: Sync Contact + Bilateral Push). Dual Engagement removed. Asymmetric OCB. **80% SR but pushing wrong direction!**
- **v7**: **Added Directional Push Reward** (`goal_push_bonus`). Rewards box velocity projected onto goal direction (continuous: positive toward goal, negative away). Scale 0.003. Still pushed wrong direction too often.
- **v7b**: **Increased directional push** 0.003 → 0.01 (3.3x). Stronger punishment for pushing away from goal. Command: `./run_training.sh --algo happo --env mapush --exp_name critic12 --use_concat_agent_observations_critic True --cooperation_rewards True --seed 7`

---

## Overview

CRITIC12 adds **three-tier cooperation bonuses** to the reward structure to address the freeloading problem observed in CRITIC10 (60% success with one agent working, one hovering).

Unlike gating or penalties, CRITIC12 uses **pure additive positive rewards** that encourage both agents to:
1. Be near the box together (Dual Engagement)
2. Make physical contact together (Synchronized Contact)
3. Push in useful directions together (Bilateral Push)

All bonuses are **shared** (identical for both agents) to work with HAPPO's centralized critic.

---

## ⚠️ 10x SCALE INCREASE (Dec 28, 2025)

### Why the Increase?

After initial testing with original magnitudes, we discovered bonuses were **too weak** to provide meaningful learning signal:

**Initial Run (87.5M steps) - Original Scales:**
- `dual_engagement_bonus`: 0.00017/step (only 57% of designed 0.0003)
- `synchronized_contact_bonus`: 0.000001/step (**800x smaller** than designed 0.0008)
- `bilateral_push_bonus`: 0.0/step (never triggered)
- **Success rate: 28.8%** (worse than CRITIC10's 57.5%)
- **Total cooperation signal: ~0.02/episode** (only 4% of task reward)

**Problem:** Agents weren't cooperating enough to get meaningful bonus signal, creating a chicken-and-egg problem:
- Too little cooperation → too little bonus → no incentive to cooperate more

**Solution:** Increase magnitudes 10x to provide stronger signal during exploration phase.

### New Magnitudes (10x)

| Bonus | Original | New (10x) | Episode Total (Expected) |
|-------|----------|-----------|--------------------------|
| **Dual Engagement** | 0.0003/step | **0.003/step** | +0.6/episode |
| **Synchronized Contact** | 0.0008/step | **0.008/step** | +0.4-0.8/episode |
| **Bilateral Push** | 0.001/step | **0.01/step** | +0.3-0.5/episode |
| **Total Cooperation** | 0.0021/step | **0.021/step** | **+1.3-1.9/episode** |

**Comparison to task reward:**
- Task success: +0.5/episode (reach_target)
- Cooperation bonuses: +1.3-1.9/episode (**2.6-3.8x task reward**)
- This large ratio is intentional to overcome the freeloading equilibrium

**Expected outcome:** Strong cooperation signal should drive both agents to engage, even if it temporarily reduces task success rate during exploration.

---

## ⚠️ v2: RELAXED THRESHOLDS (Dec 29, 2025)

### Why Relax Thresholds?

After testing with 10x magnitudes (seed-00007), we achieved **81-85% success rate** but observed in viewer mode:
- **Cooperation barely exists** - agents hover near box but don't push together
- **Lazy pushing** - one agent does most work
- **Much worse than MAPPO baseline** in actual cooperation quality

**TensorBoard evidence:**
- `dual_engagement_bonus`: 0.00187/step ✅ (working - both near box)
- `synchronized_contact_bonus`: 0.000019/step ❌ (800x lower than expected!)
- `bilateral_push_bonus`: 0.000003/step ❌ (3000x lower than expected!)

**Problem:** Agents learned to hover near box (Tier 1) but thresholds for Tier 2 & 3 were too strict to ever trigger.

### Threshold Changes

| Threshold | Original | v2 (Relaxed) | Change |
|-----------|----------|--------------|--------|
| `dual_engagement_threshold` | 1.5m | **2.0m** | +33% (easier) |
| `contact_threshold_sync` | 0.8m | **1.2m** | +50% (easier) |
| `push_force_threshold` | 0.1 | **0.03** | -70% (much easier) |

### Rationale

1. **Contact threshold 0.8m → 1.2m**
   - Original 0.8m was too strict (barely within arm's reach of box)
   - 1.2m allows agents to be in "pushing range" without perfect positioning
   - Should trigger `synchronized_contact_bonus` more often

2. **Push velocity threshold 0.1 → 0.03**
   - Original 0.1 required significant velocity toward goal
   - Hard to achieve simultaneously by both agents
   - 0.03 allows "any movement toward goal" to count
   - Should trigger `bilateral_push_bonus` more often

3. **Dual engagement threshold 1.5m → 2.0m**
   - Slightly relaxed to provide more gradient
   - Easier baseline to build upon

### Expected Outcome

With relaxed thresholds:
- `synchronized_contact_bonus` should trigger ~50-100x more often
- `bilateral_push_bonus` should actually trigger (was nearly 0)
- Agents should learn TRUE cooperation, not just "hovering near box"
- Actual pushing together should emerge

---

## ⚠️ v3: ASYMMETRIC OCB (Dec 30, 2025)

### Why Asymmetric OCB?

After testing v2 (150M→200M steps, seed-00007), we achieved **81-87% success rate** but still observed:
- **One agent sometimes reluctant to help** - freeloading still occurs
- **Push box wrong direction then "remember"** - agents push away from goal, then switch sides
- **OCB reward cancels out** - one correct + one wrong agent = 0 total OCB signal

**TensorBoard evidence:**
- `ocb_reward`: 0.000322 → 0.001242 (+285% but still weak ~0.001)
- Success rate plateaued at 81-87%
- Cooperation bonuses declining (agents learning they can succeed without full cooperation)

**Root Cause:**
The OCB reward is symmetric: correct side = +0.004, wrong side = -0.004.
When one agent is correct and one is wrong, they cancel to 0 total signal.
This means the team learns nothing about positioning when agents are on opposite sides.

### Asymmetric OCB Solution

Make wrong-side penalty **2x stronger** than correct-side reward:

| Position | Old (Symmetric) | New (Asymmetric) |
|----------|-----------------|------------------|
| Correct side | +0.004 | +0.004 (unchanged) |
| Wrong side | -0.004 | **-0.008** (2x penalty) |
| Both correct | +0.008 | +0.008 |
| Both wrong | -0.008 | **-0.016** |
| **One each** | **0** (cancel!) | **-0.004** (net negative!) |

### Why This Works

1. **Breaks the cancellation problem**
   - Old: Agent A correct (+0.004) + Agent B wrong (-0.004) = 0 (no signal)
   - New: Agent A correct (+0.004) + Agent B wrong (-0.008) = -0.004 (negative signal!)

2. **Creates pressure on wrong-side agent**
   - Even when partner is correct, being on wrong side hurts the team
   - Agent on wrong side must move to correct side to maximize team reward

3. **Preserves positive signal for full cooperation**
   - Both correct still gives +0.008 (same as before)
   - Goal: both agents on correct pushing side

### Implementation

**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py` (lines 487-495)

```python
# CRITIC12 v3: Asymmetric OCB - wrong side penalty is 2x stronger
# Positive OCB = correct side (behind box relative to goal)
# Negative OCB = wrong side (between box and goal)
raw_ocb = torch.sum(target_direction * normal_vector, dim=1)
ocb_reward = torch.where(
    raw_ocb >= 0,
    raw_ocb * self.ocb_reward_scale,           # Correct side: +0.004 scale
    raw_ocb * self.ocb_reward_scale * 2.0      # Wrong side: -0.008 scale (2x penalty)
)
```

### Expected Outcome

With asymmetric OCB:
- OCB reward should be net negative more often (pushing agents to correct side)
- Both agents should converge to pushing from correct side faster
- "Push wrong way then remember" behavior should decrease
- May see brief success rate dip as agents relearn positioning

---

## ⚠️ v4: SYMMETRIC OCB + INCREASED SCALE (Dec 31, 2025)

### Why Revert to Symmetric + Increase Scale?

After testing v3 (200M steps, seed-00007-2025-12-30-17-41-37), we achieved **~70% success rate** but:
- **OCB reward still negative** (-0.00085 to -0.001 per step)
- **Agents don't care about positioning** - success reward dominates
- **Asymmetric penalty made things worse** - net OCB more negative

**TensorBoard Analysis @ 200M steps:**

| Reward Component | Per-step | Per-episode (200 steps) | % of Budget |
|------------------|----------|------------------------|-------------|
| reach_target_reward | +0.01384 | **+7.00** (at 70% SR) | **59%** |
| distance_to_target | -0.00546 | -1.09 | 9% |
| synchronized_contact | +0.00538 | +1.08 | 9% |
| approach_to_box | -0.00376 | -0.75 | 6% |
| dual_engagement | +0.00290 | +0.58 | 5% |
| collision_punishment | -0.00236 | -0.47 | 4% |
| bilateral_push | +0.00175 | +0.35 | 3% |
| push_reward | +0.00095 | +0.19 | 2% |
| **ocb_reward** | **-0.00085** | **-0.17** | **1.4%** |

**Root Cause:**
1. `reach_target_reward_scale = 10` dominates (59% of budget)
2. OCB is only 1.4% of budget - too weak to matter
3. Asymmetric 2x penalty made net OCB more negative

### v4 Solution

1. **Revert OCB to symmetric** (remove 2x wrong-side penalty)
2. **Increase OCB scale** from 0.004 → 0.01 (2.5x increase)

**Expected impact:**
- OCB becomes ~3.5% of budget (was 1.4%)
- Symmetric means balanced signal for correct/wrong positioning
- Agents should care more about being on correct side

### Implementation

**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py` (lines 487-492)

```python
# CRITIC12 v4: Symmetric OCB (reverted from v3 asymmetric)
# Positive OCB = correct side (behind box relative to goal)
# Negative OCB = wrong side (between box and goal)
# Scale increased from 0.004 to 0.01 to be more significant in reward budget
raw_ocb = torch.sum(target_direction * normal_vector, dim=1)
ocb_reward = raw_ocb * self.ocb_reward_scale
```

**File:** `task/cuboid/config.py`

```python
ocb_reward_scale = 0.01  # CRITIC12 v4: increased from 0.004 (2.5x)
```

### Future Considerations

If OCB still doesn't drive positioning behavior, consider:
1. **Reduce `reach_target_reward_scale`** from 10 → 2-5 (make per-step shaping matter more)
2. **Further increase OCB scale** to 0.02-0.03
3. **Add explicit "both on correct side" bonus** in cooperation rewards

---

## ⚠️ v5: JOINT BINARY OCB (Dec 31, 2025)

### Why v4 Failed

After testing v4 (25M steps, seed-00007-2025-12-31-15-21-53):
- **OCB reward went NEGATIVE** (-0.00087 at 25M steps)
- Meanwhile, baseline MAPPO at 25M steps: **+0.0025** (positive!)

**Comparison with baseline MAPPO:**

| Metric | CRITIC12 v4 (HAPPO) | Baseline MAPPO |
|--------|---------------------|----------------|
| OCB @ 500K | +0.00009 | +0.00022 |
| OCB @ 25M | **-0.00087** | **+0.0025** |
| Trend | Negative, declining | Positive, growing |

### Root Cause: Team Reward Aggregation

The critical difference is how rewards flow:

**Baseline MAPPO (individual rewards):**
```python
reward[:, 0] = +0.004  # Agent0 on correct side → learns to stay
reward[:, 1] = -0.004  # Agent1 on wrong side → learns to move!
```

**Our HAPPO (team reward aggregation):**
```python
reward[:, 0] = +0.01   # Agent0 on correct side
reward[:, 1] = -0.01   # Agent1 on wrong side
team_reward = reward.sum(dim=1)  # = 0 !!! Cancels out!
# Neither agent gets a signal about OCB positioning
```

When agents are on opposite sides, their OCB rewards cancel to zero in team reward. No learning signal!

### v5 Solution: Joint Binary OCB

Only reward when BOTH agents are on correct side:

```python
both_correct = (raw_ocb_0 > 0) & (raw_ocb_1 > 0)

joint_ocb_reward = where(
    both_correct,
    +0.004,  # Both on correct pushing side
    -0.004   # Any agent on wrong side
)

# Same reward for both agents (team reward)
reward[:, :] += joint_ocb_reward
```

### Why These Magnitudes?

Using original MAPPO scale (0.004), symmetric:

| Situation | Old (cancelled) | New (joint) |
|-----------|-----------------|-------------|
| Both correct | +0.004 + 0.004 = +0.008 | **+0.004** |
| One each | +0.004 - 0.004 = **0** | **-0.004** |
| Both wrong | -0.004 - 0.004 = -0.008 | **-0.004** |

- **Symmetric** (±0.004) matches original MAPPO magnitude
- **No cancellation**: even when one is wrong, team gets -0.004 signal
- **Clear binary signal**: easier for critic to learn than continuous

### Expected Episode Impact (200 steps)

| Behavior | Per-episode |
|----------|-------------|
| Both always correct (ideal) | **+0.8** |
| Both always wrong (worst) | -0.8 |
| 50% correct / 50% wrong | 0.0 |

Matches original MAPPO OCB magnitude.

### Implementation

**File:** `mqe/envs/wrappers/go1_push_mid_wrapper.py`

```python
# Compute raw OCB for each agent
raw_ocb_list = []
for i in range(self.num_agents):
    # ... compute normal_vector for agent i ...
    raw_ocb = torch.sum(target_direction * normal_vector, dim=1)
    raw_ocb_list.append(raw_ocb)

# Joint OCB: both must be correct for positive reward
both_correct = (raw_ocb_list[0] > 0) & (raw_ocb_list[1] > 0)

joint_ocb_reward = torch.where(
    both_correct,
    torch.full_like(raw_ocb_list[0], self.ocb_reward_scale),   # Both correct: +0.004
    torch.full_like(raw_ocb_list[0], -self.ocb_reward_scale)   # Otherwise: -0.004
)

# Add as TEAM reward (same for both agents)
reward[:, :] += joint_ocb_reward.unsqueeze(1)
```

---

## Motivation: The Freeloading Problem

### CRITIC10 Observed Behavior

In the 200M step CRITIC10 run:
- **Success rate:** 60% (good!)
- **Problem:** Only one agent pushing, other hovering/blocking
- **Evidence:**
  - Agent 0 policy loss: -0.0003 (weak signal)
  - Agent 1 policy loss: -0.002 (strong signal, 6.8x larger)
  - Agent 0 gradients: +178% (not converged, searching)
  - Agent 1 gradients: -75% (converged to solo strategy)

### Root Cause

1. **Solo pushing "good enough"**: One agent can achieve 60% success alone
2. **No cooperation incentive**: All cooperation bonuses were disabled (set to 0)
3. **Credit assignment confusion**: Hovering agent gets positive returns when working agent succeeds
4. **Stable equilibrium**: Agent 1 converges to solo pushing, Agent 0 settles in shallow local minimum

### Why Not Gating or Penalties?

User requirement: **"prefer not gating. rather have new source of rewards rather than punish through neg reward or reducing other existing rewards"**

CRITIC12 follows this by providing **only positive additive bonuses** for cooperation.

---

## Three-Tier Cooperation Bonus System

### Tier 1: Dual Engagement Bonus 🟢 FOUNDATIONAL

**Reward:** `+0.003/step` (10x scale)
**Condition:** Both agents within 2.0m of box (v2: was 1.5m)

```python
both_near = (dist_agent0 < 2.0) AND (dist_agent1 < 2.0)
reward += 0.003 if both_near else 0
```

**Why this works:**
- Simplest cooperation requirement (just be near)
- Counters hovering far from box
- Provides consistent background signal for engagement
- Expected: ~200 steps/episode → **+0.6 per episode**

---

### Tier 2: Synchronized Contact Bonus 🟡 STRONGER

**Reward:** `+0.008/step` (10x scale)
**Condition:** Both agents within 1.2m of box (v2: was 0.8m, +50% easier)

```python
both_in_contact = (dist_agent0 < 1.2) AND (dist_agent1 < 1.2)
reward += 0.008 if both_in_contact else 0
```

**Why this works:**
- Requires physical proximity (pushing range)
- Freeloader hovering at 2.0m won't get this bonus
- 10x magnitude of existing `push_reward` for stronger signal
- v2: Relaxed threshold should trigger much more often
- Expected: ~50-100 steps/episode → **+0.4 to +0.8 per episode**

---

### Tier 3: Bilateral Push Bonus 🔴 STRONGEST

**Reward:** `+0.01/step` (10x scale)
**Condition:** Both agents in contact AND pushing toward goal (v2: relaxed thresholds)

```python
# Must be in contact (v2: 1.2m, was 0.8m)
both_in_contact = (dist_agent0 < 1.2) AND (dist_agent1 < 1.2)

# Both moving toward goal (v2: 0.03, was 0.1 - 70% easier)
box_to_goal = normalize(target_pos - box_pos)
agent0_vel_toward_goal = dot(agent0_velocity, box_to_goal)
agent1_vel_toward_goal = dot(agent1_velocity, box_to_goal)

both_pushing = (agent0_vel_toward_goal > 0.03) AND
               (agent1_vel_toward_goal > 0.03) AND
               both_in_contact

reward += 0.01 if both_pushing else 0
```

**Why this works:**
- Highest quality cooperation signal (both doing useful work)
- Directly measures effective collaboration
- Largest per-step bonus for strongest signal
- v2: Relaxed velocity threshold (0.03) should trigger much more often
- Expected: ~30-50 steps/episode → **+0.3 to +0.5 per episode**

---

## Bonus Magnitudes: Scale Justification (10x Updated)

### Comparison to Existing Rewards

| Reward Component | Magnitude/Step | Episode Total | Type |
|------------------|----------------|---------------|------|
| **Task Rewards (Baseline)** |
| `reach_target` | +0.5 (single event) | +0.5 | Task success |
| `push_reward` | ~0.0008/step | ~0.8 | Box moving |
| `distance_to_target` | varies | varies | Progress |
| `approach_to_box` | negative penalty | ~-0.006 | Stay close |
| **CRITIC12 Cooperation Bonuses (10x Scale)** |
| Dual Engagement | +0.003/step | +0.6 | Cooperation |
| Synchronized Contact | +0.008/step | +0.4-0.8 | Cooperation |
| Bilateral Push | +0.01/step | +0.3-0.5 | Cooperation |
| **Total Cooperation** | **+0.021/step (max)** | **+1.3-1.9** | **260-380% of task** |

### Design Principles (Updated for 10x)

1. **Total cooperation bonus >> task reward** (intentional)
   - Task success: +0.5
   - Full cooperation: +1.3-1.9
   - **2.6-3.8x larger** to overcome freeloading equilibrium
   - Strong signal needed during exploration phase

2. **Per-step magnitudes 10x baseline rewards**
   - Synchronized Contact: 0.008 = **10x** `push_reward`
   - Bilateral Push: 0.01 = premium for quality cooperation
   - Dual Engagement: 0.003 = strong baseline engagement signal

3. **Bonuses stack but don't overlap**
   - All three bonuses are independent
   - Can earn 0, 1, 2, or all 3 simultaneously
   - Maximum: 0.003 + 0.008 + 0.01 = 0.021/step

4. **Why so large?**
   - Original scales (38% of task) were too weak
   - Chicken-and-egg: need cooperation to get bonus, need bonus to learn cooperation
   - 10x scale breaks the deadlock by making cooperation highly rewarding
   - Once learned, can potentially reduce scales in future iterations

---

## Implementation Details

### Files Modified

#### 1. `HARL/harl_mapush/train.py`

**Added flag (lines 52-53):**
```python
parser.add_argument("--cooperation_rewards", type=lambda x: (str(x).lower() == 'true'), default=False,
                   help="CRITIC12: Enable three-tier cooperation bonuses. DEFAULT: False")
```

**Pass to environment (lines 97, 109):**
```python
use_cooperation = args.get("cooperation_rewards", False)
env_args = {
    ...
    "cooperation_rewards": use_cooperation,  # CRITIC12
}
```

#### 2. `mqe/envs/wrappers/go1_push_mid_wrapper.py`

**Flag reading (lines 100-104):**
```python
# CRITIC12: Three-tier cooperation bonuses
self.cooperation_rewards = getattr(self.cfg.rewards, "cooperation_rewards", False)
self.dual_engagement_threshold = 1.5  # meters
self.contact_threshold_sync = 0.8  # meters
self.push_force_threshold = 0.1  # velocity threshold
```

**Reward buffer (lines 87-90):**
```python
"dual_engagement_bonus": 0,
"synchronized_contact_bonus": 0,
"bilateral_push_bonus": 0,
```

**Reward computation (lines 494-535):**
```python
if self.cooperation_rewards:
    # Calculate distances
    dist_agent0 = torch.norm(box_pos[:, :2] - base_pos[:, 0, :2], dim=1)
    dist_agent1 = torch.norm(box_pos[:, :2] - base_pos[:, 1, :2], dim=1)

    # Tier 1: Dual Engagement (10x scale)
    both_near = (dist_agent0 < 1.5) & (dist_agent1 < 1.5)
    dual_engagement_bonus[both_near] = 0.003  # 10x: was 0.0003
    reward[:, :] += dual_engagement_bonus.unsqueeze(1).repeat(1, self.num_agents)

    # Tier 2: Synchronized Contact (10x scale)
    both_in_contact = (dist_agent0 < 0.8) & (dist_agent1 < 0.8)
    synchronized_contact_bonus[both_in_contact] = 0.008  # 10x: was 0.0008
    reward[:, :] += synchronized_contact_bonus.unsqueeze(1).repeat(1, self.num_agents)

    # Tier 3: Bilateral Push (10x scale)
    box_to_goal_norm = normalize(target_pos - box_pos)
    agent0_vel_toward_goal = dot(base_vel[:, 0, :2], box_to_goal_norm)
    agent1_vel_toward_goal = dot(base_vel[:, 1, :2], box_to_goal_norm)
    both_pushing = (agent0_vel > 0.1) & (agent1_vel > 0.1) & both_in_contact
    bilateral_push_bonus[both_pushing] = 0.01  # 10x: was 0.001
    reward[:, :] += bilateral_push_bonus.unsqueeze(1).repeat(1, self.num_agents)
```

---

## Usage

### Training with CRITIC12

```bash
# Basic usage with cooperation bonuses
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic12_cooperation \
    --cooperation_rewards True \
    --seed 1

# Combine with specific critic architecture (recommended: CRITIC10)
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic12_plus_critic10 \
    --cooperation_rewards True \
    --use_concat_agent_observations_critic True \
    --seed 1

# Combine with CRITIC7 (absolute coordinates)
./run_training.sh \
    --algo happo \
    --env mapush \
    --exp_name critic12_plus_critic7 \
    --cooperation_rewards True \
    --seed 1
```

### Verify Configuration

```bash
# Check that cooperation bonuses are enabled
cat results/mapush/go1push_mid/happo/critic12_cooperation/seed-*/config.json | grep "cooperation_rewards"

# Should show:
# "cooperation_rewards": true
```

### Monitor Training

Watch for cooperation bonus values in TensorBoard:
- `dual_engagement_bonus` - should increase over time
- `synchronized_contact_bonus` - should increase as agents learn contact
- `bilateral_push_bonus` - should increase as agents learn coordinated pushing

**Expected progression:**
1. **Early (0-20M steps):** Low bonuses, agents learning basics
2. **Mid (20-80M steps):** Dual engagement bonus increases (learning to approach together)
3. **Late (80-150M steps):** Contact and push bonuses increase (learning coordination)

---

## Expected Behavior

### Hypothesis

The three-tier bonuses will:

1. **Break freeloading equilibrium**
   - Solo pushing: only gets task rewards (~0.5/episode)
   - Cooperative pushing: gets task + cooperation (~0.7/episode)
   - 40% more reward for cooperation drives both agents to participate

2. **Provide clear learning signal**
   - Early: Learn Tier 1 (dual engagement) - easier
   - Mid: Learn Tier 2 (synchronized contact) - medium
   - Late: Learn Tier 3 (bilateral push) - harder
   - Curriculum-like structure

3. **Improve final success rate**
   - CRITIC10 baseline: 60% (solo pushing)
   - CRITIC12 target: 80-90% (true cooperation)
   - 30-50% improvement from coordination

### Expected Metrics

**Agent 0 (Previously Freeloader):**
- Policy loss magnitude: -0.0003 → -0.002 (increase 6x, matching Agent 1)
- Gradient trend: +178% → converge to near-zero
- Should start contributing to task

**Agent 1 (Worker):**
- Policy loss: Remains negative (already working)
- Gradient: Remains low (already converged)
- May improve strategy to coordinate with Agent 0

**Cooperation Bonuses (TensorBoard):**
- `dual_engagement_bonus`: 0 → 0.06/episode
- `synchronized_contact_bonus`: 0 → 0.04-0.08/episode
- `bilateral_push_bonus`: 0 → 0.03-0.05/episode

**Success Rate:**
- Baseline (CRITIC10): 60%
- Target (CRITIC12): 80-90%

---

## Key Properties

### ✅ Positive Reward Only

All bonuses are additive and positive. No penalties or gating.
- User requirement: **"prefer not gating"**
- No reduction of existing rewards
- Only new sources of reward

### ✅ Shared for Both Agents

All bonuses are identical for Agent 0 and Agent 1.
- Compatible with HAPPO centralized critic
- Both agents see same cooperation signal
- Encourages symmetric contribution

### ✅ Balanced Magnitudes

Total cooperation bonus ≈ 38% of task reward.
- Large enough to matter
- Small enough not to dominate task objective
- Scales match existing reward components

### ✅ Progressive Difficulty

Three tiers provide curriculum-like structure:
- Tier 1: Easy (just be near)
- Tier 2: Medium (make contact)
- Tier 3: Hard (push together)

### ✅ Task-Aligned

All bonuses require proximity to box and/or movement toward goal.
- Not orthogonal to task
- Reinforces useful behavior
- No "gaming" possible

---

## Comparison to Other Approaches

### vs. Gating (Option 5 from brainstorm)

**Gating:** Reduce rewards when not cooperating
```python
reward = base_reward * min(engagement_agent0, engagement_agent1)
```
- ❌ Punishes through reduction
- ❌ Negative framing

**CRITIC12:** Add rewards when cooperating
```python
reward = base_reward + cooperation_bonuses
```
- ✅ Positive framing
- ✅ New reward source

### vs. Individualized Rewards

**Individualized:** Each agent gets different reward
- ❌ Breaks HAPPO centralized critic
- ❌ Advantage calculation undefined

**CRITIC12:** Both agents get same reward
- ✅ Compatible with HAPPO
- ✅ Centralized critic works correctly

### vs. CRITIC11 (Explicit Inter-Robot Distance)

**CRITIC11:** Add inter-robot distance to critic input
- Implicit: Critic must learn to value cooperation
- Indirect signal

**CRITIC12:** Explicit cooperation rewards
- Direct: Reward cooperation explicitly
- Clear signal to both critic and actors

---

## Potential Issues & Mitigations

### Issue 1: Bonuses Too Small

**Symptom:** Freeloading persists, success rate stays at 60%

**Diagnosis:** Cooperation signal too weak to overcome solo equilibrium

**Mitigation:** Increase bonus magnitudes
```python
dual_engagement_bonus = 0.0005  # was 0.0003
synchronized_contact_bonus = 0.0012  # was 0.0008
bilateral_push_bonus = 0.0015  # was 0.001
```

### Issue 2: Bonuses Too Large

**Symptom:** Agents cluster near box, ignore task objective

**Diagnosis:** Cooperation rewards dominate task rewards

**Mitigation:** Reduce bonus magnitudes or add distance-to-goal requirement

### Issue 3: Bilateral Push Never Triggers

**Symptom:** Tier 3 bonus always 0 in logs

**Diagnosis:** Velocity threshold too high or agents never in position

**Mitigation:** Lower `push_force_threshold` from 0.1 to 0.05

### Issue 4: Still Freeloading After 200M Steps

**Symptom:** One agent still hovering despite bonuses

**Diagnosis:** Local minimum too deep, need stronger intervention

**Next steps:**
1. Combine with curriculum learning (vary box difficulty)
2. Add initialization bias (spawn both agents on push side)
3. Try MAPPO (shared parameters) instead of HAPPO

---

## Experiment Design

### Test 1: CRITIC12 vs CRITIC10 Baseline

**Goal:** Does cooperation bonus reduce freeloading?

```bash
# Baseline (no cooperation bonuses)
./run_training.sh --exp_name critic10_baseline \
    --use_concat_agent_observations_critic True --seed 1 &

# CRITIC12 (with cooperation bonuses)
./run_training.sh --exp_name critic12_cooperation \
    --use_concat_agent_observations_critic True \
    --cooperation_rewards True --seed 1 &
```

**Metrics to compare:**
- Agent 0 vs Agent 1 policy loss magnitude
- Agent 0 vs Agent 1 gradient norms
- Success rate progression
- Cooperation bonus values (should be > 0 for CRITIC12)

**Expected result:**
- CRITIC12: Both agents working (symmetric losses)
- CRITIC12: Higher success rate (80-90% vs 60%)
- CRITIC12: Positive cooperation bonuses in logs

---

### Test 2: CRITIC12 with Different Critic Architectures

**Goal:** Does cooperation work with all critic types?

```bash
# CRITIC12 + CRITIC7 (absolute coords)
./run_training.sh --exp_name critic12_absolute \
    --cooperation_rewards True --seed 1 &

# CRITIC12 + CRITIC9 (box-centered)
./run_training.sh --exp_name critic12_boxcentered \
    --cooperation_rewards True \
    --use_box_centered_critic True --seed 1 &

# CRITIC12 + CRITIC10 (concatenated obs)
./run_training.sh --exp_name critic12_concat \
    --cooperation_rewards True \
    --use_concat_agent_observations_critic True --seed 1 &
```

**Expected result:**
- All should show cooperation (bonuses > 0)
- Success rates may vary by critic architecture
- CRITIC10 or CRITIC7 likely best

---

### Test 3: Ablation Study

**Goal:** Which bonus tier matters most?

Modify code to enable/disable individual tiers, then run:
```bash
./run_training.sh --exp_name critic12_tier1only ...  # Only dual engagement
./run_training.sh --exp_name critic12_tier2only ...  # Only contact
./run_training.sh --exp_name critic12_tier3only ...  # Only bilateral push
./run_training.sh --exp_name critic12_all3 ...       # All three (full CRITIC12)
```

**Expected result:**
- Tier 1 alone: Some improvement, not full cooperation
- Tier 2 alone: Better, but still incomplete
- Tier 3 alone: Best single tier (direct push signal)
- All three: Best overall (curriculum effect)

---

## Summary

**CRITIC12 = Three-Tier Positive Cooperation Bonuses + Asymmetric OCB (v3)**

**Key Innovations:**
1. Pure additive positive rewards for cooperation (no penalties or gating)
2. Asymmetric OCB: wrong-side penalty 2x stronger than correct-side reward

**Three Tiers (10x magnitudes):**
1. **Dual Engagement** (+0.003/step): Both near box
2. **Synchronized Contact** (+0.008/step): Both touching box
3. **Bilateral Push** (+0.01/step): Both pushing toward goal

**Asymmetric OCB (v3):**
- Correct side (behind box): +0.004 per agent
- Wrong side (between box & goal): -0.008 per agent (2x penalty)
- One correct + one wrong = -0.004 net (breaks cancellation problem)

**Version History:**
- v1: Original magnitudes (too weak)
- v2: 10x scale + relaxed thresholds (Tier 2&3 now triggering)
- v3: Asymmetric OCB (wrong side penalty 2x, breaks cancellation)

**Expected Impact (v3):**
- Reduce "push wrong way then remember" behavior
- Break the symmetric OCB cancellation problem
- Both agents converge to correct pushing side faster
- Target: 90%+ success with proper cooperative pushing

**Total cooperation bonus per episode (10x scale):**
- Minimal: +0.6 (just engagement)
- Medium: +1.4 (engagement + contact)
- Full: +1.9 (all three tiers)
- **3.8x task reward** (+0.5) - intentionally large

**Compatibility:**
- ✅ Works with all critic architectures (CRITIC7/9/10/11)
- ✅ Compatible with HAPPO centralized critic
- ✅ Shared rewards (same for both agents)
- ✅ Positive cooperation bonuses only

**Next Steps:**
1. Test v3 asymmetric OCB (resume from 200M checkpoint)
2. Monitor OCB reward (should be more negative when agents on opposite sides)
3. Watch for reduced "wrong way pushing" behavior
4. Target: both agents consistently on correct pushing side
