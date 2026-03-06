# Individual Rewards Testing — Summary for Report

**Task:** Cuboid-push (mid-level controller, 2x Go1, HAPPO algorithm)
**Problem:** Freeloading — one agent stops pushing while the other does all the work under shared team rewards.
**Period:** Dec 14–16, 2025 (12 iterations)

---

## Experiment Progression

### Phase 1: Reward Individualization (Iters 1–5) — All Failed

| Iter | Approach | Result |
|------|----------|--------|
| 1 | All shared rewards weighted by agent's proximity to box | Too aggressive — agents got near-zero reward when far, destabilizing learning |
| 2 | Only push_reward contact-weighted, others shared | Push reward too sparse/noisy for stable individual credit |
| 3 | Idle penalty when agent far from box and box stationary | Freeloading agent was actively moving away, not idle — penalty never triggered |
| 4 | Direction-aware approach + collision penalty reduced 3x | Showed collision penalty (-0.0025/step) made escaping rational; reduction helped but didn't solve freeloading |
| 5 | All positive rewards individualized + goal_push_bonus | Agent 1 escaped into local minimum (reward tanked to -18.66) |

**Conclusion:** Naively individualizing rewards destabilizes HAPPO training. Agents find degenerate strategies (escape, hover) rather than learning to push.

### Phase 2: Positive Reinforcement Redesign (Iter 6) — Revealed Critical Bug

Replaced punishments with positive incentives (engagement bonus, cooperation bonus, same-side bonus, velocity-based push contribution). Discovered a **fundamental bug**: in EP (Environment Provided) critic mode, HAPPO was only reading Agent 0's rewards for the centralized critic, so Agent 1 received garbage advantage estimates. Critic value loss exploded (+170%).

### Phase 3: FP Critic Mode (Iters 7–9) — Partial Success

Switched to FP (Feature Pruned) critic mode so each agent's critic saw its own rewards. Stripped back to simplified reward structure.

| Iter | Change | Result |
|------|--------|--------|
| 7 | FP critic mode | Intended to fix EP bug; training more stable but later found FP is wrong for HAPPO |
| 8 | Gated shared rewards (both agents must engage or both get zero) | Reward averaging diluted accountability; oscillating value loss |
| 9 | FP + original MAPPO reward structure, only push/reach_target individualized | **18% success rate at 100M steps** — best so far, but agents hovered without actively pushing |

### Phase 4: Hyperparameter Fixes (Iters 10–11)

| Iter | Change | Result |
|------|--------|--------|
| 10 | Collision penalty reduced 5x (−0.0025 → −0.0005) | Old penalty dominated episode reward (−13.4/episode at 0.5m distance); agents rationally avoided each other |
| 11 | Learning rate reduced 10x (0.005 → 0.0005) | HAPPO default LR was 10x higher than MAPPO's, causing critic divergence (value_loss: 0.10 → 0.19) |

### Phase 5: Root Cause Identified (Iter 12) — Architectural Fix

**Discovery:** HAPPO requires a **single centralized critic (EP mode)** with **team rewards**. FP mode (per-agent critics) caused critic divergence (value_loss: 0.014 → 0.27) and was architecturally incompatible with HAPPO's sequential update scheme.

**Correct HAPPO setup:**
- EP critic (single shared critic seeing global state)
- Team rewards (sum per-agent components, broadcast identically to all agents)
- Separate actor networks per agent
- Credit assignment via HAPPO's sequential importance-weighting mechanism — not via reward decomposition

This matches HAPPO theory: the sequential update "factor" (accumulated importance weights across agents) is the credit assignment mechanism. Trying to decompose rewards per-agent fights the algorithm's design.

---

## Key Findings

1. **Don't individualize rewards with HAPPO.** The algorithm's sequential update mechanism handles credit assignment internally. Reward shaping for individual accountability destabilizes training.

2. **Collision penalty magnitude matters.** At −0.0025/step, the collision penalty dominated total episode reward and made escaping a rational strategy. Reducing to −0.0005 removed this perverse incentive.

3. **HAPPO's default learning rate (0.005) is too high** for this environment. Matching MAPPO's 0.0005 stabilized critic training.

4. **EP mode is mandatory for HAPPO.** FP mode gives each agent its own critic, which breaks the centralized value function assumption that HAPPO's theoretical guarantees depend on.

5. **The correct configuration** (EP + team rewards + reduced collision + lower LR) is expected to match MAPPO's ~90% success rate while supporting heterogeneous observation/action spaces.

---

## Recommended HAPPO Configuration (from Iter 12)

```
Critic mode:        EP (Environment Provided — single shared critic)
Reward structure:   Team rewards (shared, broadcast to all agents)
Collision scale:    −0.0005 (not −0.0025)
Actor LR:           0.0005
Critic LR:          0.0005
Actor networks:     Separate per agent (no parameter sharing)
```
