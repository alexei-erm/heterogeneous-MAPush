# CRITIC17: Positive Approach-to-Box Reward

> **Date:** January 11, 2026
> **Status:** Testing
> **Based on:** CRITIC15 (original MAPush rewards teamified)
> **Goal:** Test if positive proximity reward improves agent engagement compared to negative penalty

---

## Motivation

In original MAPush and CRITIC15, the `approach_to_box_reward` uses a **negative quadratic penalty**:

```python
distance_reward = (-(distance + 0.5)**2) * 0.00075
```

**Problems with negative penalty:**
- Always negative (punishment-based learning)
- Quadratic growth when far from box (harsh penalty at distance)
- Agents learn to "avoid being far" rather than "seek being close"

**Hypothesis:** A **positive inverse distance reward** might encourage better engagement:
- Positive reinforcement for proximity
- Smoother gradient (hyperbolic decay vs quadratic)
- Agents learn to "seek closeness" rather than "avoid distance"

---

## Implementation

### Flag Added: `--positive_approachtobox_reward`

When enabled, switches the approach_to_box calculation from negative penalty to positive reward.

### Files Modified

1. **`mqe/envs/wrappers/go1_push_mid_wrapper.py`** (lines 141-144, 464-490)
   - Added flag: `self.positive_approachtobox_reward`
   - Modified reward calculation with conditional:

```python
if self.positive_approachtobox_reward:
    # CRITIC17: Positive reward - inverse distance (high when close, low when far)
    distance_reward = (1.0 / (distance + 0.5)) * self.approach_reward_scale
else:
    # Original: Negative quadratic penalty (always negative)
    distance_reward = (-(distance + 0.5)**2) * self.approach_reward_scale
```

2. **`HARL/harl_mapush/train.py`** (lines 110, 127)
   - Extract flag from command line args
   - Pass to environment config

3. **`mqe/envs/utils.py`** (lines 46, 79-81)
   - Add flag to `custom_cfg` function signature
   - Set flag in reward config when enabled

---

## Reward Formula Comparison

### Original (Negative Quadratic Penalty)

**Formula:** `reward = -(distance + 0.5)² × 0.00075`

| Distance | Reward | Notes |
|----------|--------|-------|
| 0.0m | -0.000188 | Best case (still negative!) |
| 1.0m | -0.001688 | |
| 2.0m | -0.004688 | |
| 3.0m | -0.009188 | Penalty grows rapidly |

**Characteristics:**
- Always negative (punishment-based)
- Quadratic growth (penalty accelerates with distance)
- At distance=0: still penalized (-0.000188)

---

### CRITIC17 (Positive Inverse Distance)

**Formula:** `reward = (1.0 / (distance + 0.5)) × 0.00075`

| Distance | Reward | Notes |
|----------|--------|-------|
| 0.0m | **+0.00150** | Maximum reward (2.0 × scale) |
| 1.0m | **+0.00050** | Medium reward |
| 2.0m | **+0.00030** | Low but positive |
| 3.0m | **+0.00021** | Very low but still positive |

**Characteristics:**
- Always positive (attraction-based)
- Hyperbolic decay (smooth gradient)
- At distance=0: maximum reward (+0.00150)

---

## Team Reward Structure

Both original and CRITIC17 use the same teamification:

```python
total_approach_reward = torch.zeros(self.num_envs, device=self.device)

# 1. Compute per-agent reward
for i in range(self.num_agents):
    distance = torch.norm(box_pos - base_pos[:, i, :], dim=1)
    distance_reward = [formula depends on flag]
    total_approach_reward += distance_reward

# 2. Average when using mapush_og_rewards_teamified
if self.mapush_og_rewards_teamified:
    total_approach_reward = total_approach_reward / self.num_agents

# 3. Both agents get same team reward
reward[:, :] += total_approach_reward.unsqueeze(1).repeat(1, self.num_agents)
```

**Example with 2 agents:**
- Agent 0: 1.0m from box → +0.00050
- Agent 1: 2.0m from box → +0.00030
- Sum: +0.00080
- **Average: +0.00040** (both agents receive this)

---

## Usage

### Training Command

```bash
python HARL/harl_mapush/train.py \
  --algo happo \
  --exp_name critic17_positive_approach \
  --mapush_og_rewards_teamified True \
  --positive_approachtobox_reward True \
  --seed 1
```

### Flags Required

- `--mapush_og_rewards_teamified True`: Use original MAPush rewards (teamified)
- `--positive_approachtobox_reward True`: Enable CRITIC17 positive reward

---

## Initial Observations (First 12M Steps)

### ⚠️ Approach Reward Decreasing Over Training

**Observation:** `approach_to_box_reward` is strictly decreasing from 0M to 12M steps.

**Why this happens:**

1. **Early training (0-2M):**
   - Random exploration frequently brings agents near box
   - High approach reward from accidental proximity
   - Agents haven't learned task structure yet

2. **Mid training (2M-12M):**
   - Agents learn to push box toward goal
   - Pushing requires repositioning (moving away temporarily)
   - Coordinated pushing means agents not always hovering at box
   - Approach reward decreases as agents optimize for task, not proximity

**This is NOT necessarily bad:**
- Task completion > box proximity
- Agents should prioritize pushing over hovering
- Decreasing approach reward may indicate learning structured behavior

---

## Key Differences from Negative Penalty

### Original Negative Penalty Trend
- **Starts:** Very negative (agents far from box)
- **Ends:** Less negative (agents learn to stay near)
- **Direction:** **Increases** (less negative = improvement)

### CRITIC17 Positive Reward Trend
- **Starts:** High (random exploration near box)
- **Ends:** Lower (structured pushing, less hovering)
- **Direction:** **Decreases** (but doesn't mean worse performance!)

**Interpretation shift required:**
- Negative penalty: Increasing = good
- Positive reward: Decreasing may be fine (if task metrics improve)

---

## What to Monitor

### Primary Metrics (More Important than Approach Reward)
1. **Success rate** - Is task completion improving?
2. **`reach_target_reward`** - Are agents reaching the goal?
3. **`push_reward`** - Are agents actively pushing?
4. **`distance_to_target_reward`** - Is box moving toward goal?

### Secondary Metrics
5. **`approach_to_box_reward`** - Proximity to box (may decrease)
6. **Visual inspection** - Are both agents engaged? (viewer mode)
7. **Episode length** - Are agents efficient?

### Signs of Success
✅ Success rate increasing
✅ Reach target reward increasing
✅ Both agents pushing in viewer mode
✅ Approach reward stable or slowly decreasing (structured behavior)

### Signs of Failure
❌ Success rate plateauing or decreasing
❌ One agent running away (freeloading)
❌ Approach reward dropping to near-zero (agents ignoring box)
❌ Episode reward decreasing

---

## Potential Issues

### Issue 1: Scale Too Weak

At current scale (`0.00075`), positive reward is small:
- At 1m: +0.0005/step
- At 2m: +0.0003/step

Compare to task rewards:
- `reach_target_reward`: +10 (huge!)
- `distance_to_target_reward`: ~0.00325/step (4.3x larger)

**Approach reward might be too weak to influence behavior.**

### Issue 2: No Incentive for Both Agents

Current reward:
- Averages both agents' proximity
- One agent close, one far: still gets medium reward
- No explicit bonus for **both** agents being engaged

### Issue 3: Hovering vs Pushing

Positive reward encourages staying near box, but:
- Task requires **pushing** (dynamic behavior)
- Agents might learn to hover instead of push
- Original penalty avoided this (punishment for being far forces action)

---

## Recommendations

### If Success Rate is Increasing: ✅ Keep Current Settings
- Decreasing approach reward is fine
- Agents learning structured pushing
- Monitor through 50M-100M steps

### If Success Rate Plateaus: Try These

**Option 1: Increase Scale**
```python
approach_reward_scale = 0.003  # 4x increase (was 0.00075)
```
Makes proximity competitive with task rewards.

**Option 2: Add Dual Engagement Bonus**
```python
# Bonus when BOTH agents near box
if both_agents_within_2m:
    reward += 0.002  # Explicit collaboration
```
Prevents one agent running away.

**Option 3: Revert to Negative Penalty**
If positive reward causes hovering instead of pushing:
```bash
# Train without the flag (use original penalty)
python HARL/harl_mapush/train.py --mapush_og_rewards_teamified True
```

**Option 4: Hybrid Approach**
```python
# Positive reward when close, penalty when very far
if distance < 2.0:
    reward = (1.0 / (distance + 0.5)) * scale  # Positive
else:
    reward = -(distance - 2.0)**2 * scale      # Penalty
```

---

## Comparison to Related Work

### CRITIC15 (Baseline)
- Uses original negative penalty
- Agents learn to avoid being far
- Success rate: [TBD from experiments]

### CRITIC17 (This Work)
- Uses positive inverse reward
- Agents learn to seek closeness
- Success rate: [TBD - currently training]

### CRITIC12 (Cooperation Bonuses)
- Added explicit dual engagement bonus
- Forced both agents to stay near box
- Success rate: 79.8% (best run)
- **May be better approach than modifying approach_to_box**

---

## Hypothesis to Test

**H1:** Positive reward increases exploration near box early in training
- **Test:** Compare exploration metrics (entropy, distance variance) at 0-10M steps

**H2:** Positive reward reduces freeloading (both agents stay engaged)
- **Test:** Compare visual inspection of agent behavior vs CRITIC15

**H3:** Positive reward improves final success rate vs negative penalty
- **Test:** Compare success rate at 100M+ steps (CRITIC17 vs CRITIC15)

**H4:** Positive reward may cause hovering instead of pushing
- **Test:** Compare `push_reward` magnitude (CRITIC17 vs CRITIC15)

---

## Files Changed Summary

| File | Lines Changed | Description |
|------|---------------|-------------|
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | 141-144, 464-490 | Added flag, modified reward calculation |
| `HARL/harl_mapush/train.py` | 110, 127 | Flag extraction and env_args passing |
| `mqe/envs/utils.py` | 46, 79-81 | Config flag propagation |

---

## Next Steps

1. **Monitor training through 50M-100M steps**
   - Watch success rate trend
   - Check if approach reward stabilizes
   - Compare to CRITIC15 baseline

2. **Visual inspection at checkpoints**
   - 10M, 30M, 50M steps
   - Verify both agents pushing (not hovering)
   - Check for freeloading behavior

3. **TensorBoard analysis**
   - Compare all reward components to CRITIC15
   - Look for unexpected trends
   - Identify if approach reward becomes irrelevant

4. **Consider scale adjustment**
   - If approach reward drops to <0.0001/step: increase scale
   - If agents hovering instead of pushing: revert to negative
   - If success rate low: try dual engagement bonus instead

---

## References

- **Original MAPush:** Negative quadratic penalty for approach
- **CRITIC15:** Teamified original rewards (baseline for comparison)
- **CRITIC12:** Explicit cooperation bonuses (alternative approach)
- **CRITIC10:** Freeloading analysis (why engagement matters)

---

## Conclusion

CRITIC17 tests whether **positive proximity rewards** work better than **negative distance penalties** for multi-agent box pushing.

**Key insight:** A decreasing approach reward during training may indicate agents learning structured task behavior (pushing with repositioning) rather than random hovering near the box.

**Decision point:** If success rate increases, CRITIC17 works. If success rate plateaus below CRITIC15, revert to negative penalty or add explicit dual engagement bonus.

**Current status:** Under evaluation (first 12M steps show decreasing approach reward - need to monitor task metrics to determine if this is beneficial or problematic).
