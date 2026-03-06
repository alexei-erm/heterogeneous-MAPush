# Critic Learning Fixing — Summary for Report

**Task:** Cuboid-push (mid-level controller, HAPPO algorithm)
**Problem:** HAPPO's centralized critic failed to converge — value loss diverged instead of decreasing, causing poor policy learning (~0–20% success rate vs MAPPO baseline ~90%).
**Period:** Dec 18, 2025 – Jan 11, 2026 (17 iterations + analysis documents)

---

## The Core Problem

HAPPO uses a single centralized critic (EP mode) shared across agents. Unlike MAPPO (simultaneous updates), HAPPO updates agents sequentially — each agent's policy shift changes the data distribution the critic was trained on, creating a "moving target" that the critic struggles to track. The critic's value loss diverged (~0.05–0.25) instead of converging (~0.004 in MAPPO baseline).

---

## Phase 1: Hyperparameter Tuning (Critic 1–5) — Max ~20% SR

| Iter | Approach | Result |
|------|----------|--------|
| 1 | More critic epochs (5→25), higher value loss coef (1→5) | Failed — more training on a moving target doesn't help |
| 2 | Pre-train critic before each agent update loop | Failed — loss plateaued at 0.25, SR stuck at ~15% |
| 3 | Slow actor updates (only every 3rd iteration) | **Best so far: ~20% SR** — giving critic time to stabilize helped |
| 4 | Higher learning rates on top of critic3 | Failed — faster rise to same 15% plateau |
| 5 | Comprehensive stability overhaul (lower LR, tighter clip, larger critic network) | Failed — too conservative, never learned |

**Takeaway:** Hyperparameter tuning alone cannot fix the critic divergence. The problem is the critic input representation, not training dynamics.

## Phase 2: Critic Input Representation (Critic 6–11) — Max ~60% SR

Systematic search for the right coordinate frame and features for the critic's global state input.

| Iter | Critic Input | Dims | Key Property | Result |
|------|-------------|------|--------------|--------|
| 6 | + 0.5x action scaling (matching OpenRL) | — | Action range fix | **0% SR** — scaling killed exploration; reverted |
| 7 | Absolute positions, no velocities | 11 | Simpler input | Baseline comparison |
| 8 | Concatenated agent local observations | 16 | Translation & rotation invariant | Foundation for best approach |
| 9 | Box-centered relative coordinates | 9 | Translation invariant, compact | Implemented as flag option |
| 10 | **Concatenated local obs (flag system)** | 16 | Invariant, rich | **57–60% SR at 150–200M steps** |
| 11 | Box-centered + explicit inter-robot distance | 9 | Coordination-aware | Implemented, superseded by critic16 |

**Takeaway:** Critic input representation matters enormously. Concatenated local observations (critic10) tripled performance over hyperparameter tuning. However, critic10 revealed a new problem: **freeloading** — one agent learned solo-pushing while the other hovered nearby.

### Freeloading Analysis (Critic 10 deep-dives)

Three analysis documents examined the freeloading:
- Agent 1 did all the pushing; Agent 0 hovered near the box contributing nothing
- Agent 1 gradients vanished (converged to solo strategy); Agent 0 had small, unstable gradients
- Policy loss magnitude asymmetry: Agent 1's loss was 6.8x larger than Agent 0's
- Root cause: team rewards give Agent 0 credit for Agent 1's work; no signal to differentiate contributions
- Comparison with MAPPO baseline showed the real diagnostic is **value loss convergence**, not entropy or policy loss oscillation

## Phase 3: Cooperation Reward Shaping (Critic 12–13) — Max ~80% SR

Added reward signals specifically designed to break the freeloading equilibrium.

| Iter | Approach | Result |
|------|----------|--------|
| 12 | Three-tier cooperation bonuses (dual engagement, synchronized contact, bilateral push) + OCB positioning reward. 9 sub-versions tested. | **Best: 79.8% SR** (v5 at 250M steps). OCB turned positive for first time. But agents pushed in wrong direction frequently. |
| 13 | "Minimal essentials" — stripped to non-overlapping core signals only | Failed — too sparse for curriculum learning. Re-adding push_reward helped but only ~20% collaborative episodes. |

**Takeaway:** Cooperation bonuses improved SR from 60% to 80%, but agents still didn't learn true bilateral pushing. The reward structure had drifted far from the proven MAPPO baseline.

## Phase 4: Teamification & Return to Basics (Critic 14–17) — Max ~85% SR

Returned to the original 7 MAPush rewards but properly converted for HAPPO's centralized critic.

| Iter | Approach | Result |
|------|----------|--------|
| 14 | Fixed reward-critic mismatch — converted per-agent rewards (approach, collision) to explicit team rewards | Approach reward converged faster (25M steps) |
| 15 | **Original MAPush rewards, teamified** (`--mapush_og_rewards_teamified`). 4 sub-versions. | **v2: 85% SR** — best overall. But inefficient solo/sequential pushing, not true cooperation. |
| 16 | Goal-centered critic (9 dims, stationary reference frame) | Under evaluation — expected to produce clearer spatial value gradients |
| 17 | Positive approach reward (inverse distance) instead of negative penalty | Under evaluation — early results ambiguous |

**Takeaway:** The original MAPush reward structure, properly teamified, achieved the best results. The remaining gap from MAPPO's ~90% is likely from the solo/sequential pushing strategy rather than a critic convergence issue.

---

## Key Findings

1. **Critic input representation > hyperparameter tuning.** Concatenated local observations (16-dim, translation/rotation invariant) was the single most impactful change, tripling SR from ~20% to ~60%.

2. **Team rewards are mandatory for HAPPO.** Per-agent rewards create variance in advantage estimation since the centralized critic estimates team value V(s). All rewards must be summed/averaged into a team signal.

3. **The original MAPush rewards work best** when properly teamified. Complex cooperation bonuses (critic12) helped but introduced new failure modes. Returning to the proven 7-reward structure (critic15) achieved the highest SR.

4. **Freeloading is the remaining challenge.** Even at 85% SR, agents learn solo/sequential pushing rather than true bilateral cooperation. The team reward structure inherently allows one agent to free-ride on the other's work.

5. **Value loss convergence is the key diagnostic.** Entropy, policy loss oscillation, and gradient magnitudes are unreliable indicators. The critical metric is whether the critic's value loss converges to near-zero.

---

## Flag System Reference

All critic variants were implemented as command-line flags for easy A/B testing:

| Flag | Critic Mode | Dims | ID |
|------|------------|------|-----|
| `--use_concat_agent_observations_critic True` | Concatenated local obs | 16 | CRITIC10 |
| `--use_box_centered_critic True` | Box-centered relative | 9 | CRITIC9 |
| `--use_goal_centered_critic True` | Goal-centered relative | 9 | CRITIC16 |
| `--use_relative_obs_critic True` | Relative + inter-robot dist | 9 | CRITIC11 |
| (default) | Absolute world frame | 11 | CRITIC7 |

Priority: relative_obs > concat_obs > goal_centered > box_centered > absolute

---

## Recommended Configuration (from Critic 15 v2)

```
Critic mode:           EP (single shared critic)
Critic input:          Concatenated local obs (--use_concat_agent_observations_critic True)
Reward structure:      Original MAPush teamified (--mapush_og_rewards_teamified True)
Actor LR:              0.0005
Critic LR:             0.0005
All other HAPPO params: Defaults (reverted from earlier experiments)
```

**Result:** ~85% success rate at 200M steps.

---

## Success Rate Progression

```
Phase 1 (hyperparams only):        ~20% SR
Phase 2 (critic input fix):        ~60% SR
Phase 3 (cooperation rewards):     ~80% SR
Phase 4 (OG rewards teamified):    ~85% SR
MAPPO baseline:                    ~90% SR
```
