# Session Summary - 2026-02-13

## Work Completed

### 1. Implemented `--require_both_contact_for_success` Flag (HAPPO)

**Goal:** Force collaboration by only giving `reach_target_reward` (10 points) if BOTH agents are within `contact_threshold` (0.8m) of the box at success time. Solo pushing = NO reward.

**Files Modified:**
- `HARL/harl_mapush/train.py` - Added command-line argument
- `HARL/harl/envs/mapush/mapush_env.py` - Wired flag through env_args
- `mqe/envs/utils.py` - Added to `custom_cfg()` function
- `mqe/envs/wrappers/go1_push_mid_wrapper.py` - Implemented reward gating logic

**Implementation in wrapper:**
```python
if self.require_both_contact_for_success:
    dist_agent0 = torch.norm(box_pos[:, :2] - base_pos[:, 0, :2], dim=1)
    dist_agent1 = torch.norm(box_pos[:, :2] - base_pos[:, 1, :2], dim=1)
    both_in_contact = (dist_agent0 < self.contact_threshold) & (dist_agent1 < self.contact_threshold)
    valid_success = self.finished_buf & both_in_contact
else:
    valid_success = self.finished_buf
```

**Result:** FAILED - Too strict. Blocked ALL successes because agents never randomly stumble upon perfect coordination. No learning signal = no learning.

---

## Problem Analysis: Why Collaboration Doesn't Emerge

### The Core Issue
It's NOT that agents "refuse to help" - it's that **solo success prevents learning collaboration**:
1. One agent (especially heavy ones like Cassie 42kg, Anymal 50kg) can push the 8kg box alone
2. Solo pusher gets full reward → learns solo behavior
3. Other agent learns to stay out of the way (avoiding collision penalty)
4. No incentive gradient toward collaboration

### Robot Masses (for reference)
- Go1: ~11 kg
- Cassie: ~42 kg
- Anymal C: ~50 kg
- Default box: 4 kg (trivial for any robot)
- Heavy box: 8 kg (still easy for Cassie/Anymal solo)

### Box Mass Calculation for Equivalent Difficulty
For 2x Go1 with 8kg box (ratio ~0.36 of combined mass):
- Anymal + Cassie = 92kg combined → equivalent box ≈ 33kg

---

## Alternative Approaches Discussed (Not Yet Implemented)

### Option 1: Softer Gating (Scaled Rewards)
Scale reach_target_reward by number of agents in contact at success:
- 2 agents in contact: 100% reward (10 points)
- 1 agent in contact: 25% reward (2.5 points)
- 0 agents in contact: 0%

**Pros:** Still penalizes solo pushing but gives SOME learning signal
**Cons:** May still be too weak a signal

### Option 2: Progressive Contact Bonus (Multiplier)
Keep full reach_target_reward, add multiplier:
- Both agents contributed: 2x multiplier (20 points)
- One agent contributed: 1x multiplier (10 points - baseline)

**Pros:** Doesn't block learning, makes collaboration MORE rewarding
**Cons:** Solo pushing still profitable

### Option 3: Collaboration Bonus Throughout Episode
Instead of gating success reward, add continuous rewards when both agents push together. Similar to existing `--collaboration_rewards` flag but stronger.

**Pros:** Provides dense learning signal throughout episode
**Cons:** May encourage agents to just "touch box together" without actually pushing

### Option 4: Minimum Contact Time Requirement
Track total contact time during episode. Require minimum threshold (e.g., both agents must have been in contact for at least 20% of episode steps).

**Pros:** Rewards sustained collaboration, not just end-state
**Cons:** Complex to tune threshold

---

## Recommendation for Next Session

**Try Option 1 (Softer Gating)** first:
- Modify the `require_both_contact_for_success` logic to scale rewards instead of binary gate
- This preserves some learning signal while still making collaboration more attractive

**Alternative:** Combine approaches - e.g., softer gating (Option 1) + collaboration bonus during episode (Option 3)

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `HARL/harl_mapush/train.py` | HAPPO training entry point, all flags |
| `HARL/harl/envs/mapush/mapush_env.py` | HARL environment wrapper |
| `mqe/envs/utils.py` | `custom_cfg()` function, flag → config |
| `mqe/envs/wrappers/go1_push_mid_wrapper.py` | Core reward computation |
| `mqe/envs/configs/go1_push_mid_config.py` | Environment config (box mass, etc.) |
| `claude_summaries/heavy_cuboid_testing_happo.md` | HAPPO experiment documentation |

---

## Current Best HAPPO Config (8kg box, 2x Go1)

```bash
python HARL/harl_mapush/train.py \
  --exp_name heavy8kg_concat_critic_teamified \
  --n_rollout_threads 500 \
  --num_env_steps 150000000 \
  --use_concat_agent_observations_critic True \
  --reward_scale_testing True
```

Key finding from previous sessions: **Distance-to-target reward HURTS HAPPO** - must be disabled (`--reward_scale_testing True` does this).

---

**Last Updated:** 2026-02-13
