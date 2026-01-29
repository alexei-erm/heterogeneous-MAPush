# Go1 Locomotion Policy - Origin, Training & Portability Analysis

**Date:** 2026-01-19
**Question:** Is the Go1 locomotion policy specific to Isaac Gym, or can it be retrained for other robots?

---

## Executive Summary

**Short Answer:** No, the locomotion policy is NOT specific to Isaac Gym. It's trained IN Isaac Gym but is robot-specific and CAN be retrained for other quadrupeds like Anymal.

**Key Finding:** There ARE existing Anymal locomotion policies trained with the same framework!

---

## 1. What is "walk-these-ways"?

### Origin
- **Project:** Walk These Ways
- **Institution:** MIT Improbable AI Lab
- **GitHub:** [Improbable-AI/walk-these-ways](https://github.com/Improbable-AI/walk-these-ways)
- **Paper:** "Walk these Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior"

### Purpose
A sim-to-real RL training and deployment system for quadruped locomotion, specifically for the **Unitree Go1 Edu robot**.

### Training Framework
```
Isaac Gym (simulator)
  ↓
PPO (reinforcement learning)
  ↓
Domain Randomization (terrain, friction, mass, etc.)
  ↓
Multiplicity of Behavior (MoB)
  ↓
Locomotion Policy (.jit files)
```

---

## 2. Is It Isaac Gym Specific?

### No - It's Trained IN Isaac Gym, Not FOR Isaac Gym

**Training Environment:**
- Uses Isaac Gym as the **physics simulator**
- Built on [legged_gym](https://github.com/leggedrobotics/legged_gym) from ETH Zurich
- Requires GPU with ≥10GB VRAM
- Runs 4,000 parallel environments by default

**Resulting Policy:**
- Exported as PyTorch JIT files (`body_latest.jit`, `adaptation_module_latest.jit`)
- **Robot-specific** (Go1 joint angles, DOFs, kinematics)
- **Deployment-agnostic** (can run anywhere that can execute PyTorch)
- Deployed on real Go1 hardware via `unitree_legged_sdk`

### Key Distinction

| Component | Isaac Gym Dependent? |
|-----------|---------------------|
| **Training Process** | ✅ Yes (uses Isaac Gym) |
| **Trained Policy** | ❌ No (PyTorch model) |
| **Robot Hardware** | ❌ No (works on real Go1) |
| **Simulation for Training** | ✅ Yes (requires Isaac Gym) |

---

## 3. Can It Be Retrained for Other Robots?

### YES - With Proper Configuration

The underlying framework ([legged_gym](https://github.com/leggedrobotics/legged_gym)) is **robot-agnostic**.

**What Needs Changing:**
1. **URDF/Robot Model** - Anymal URDF instead of Go1
2. **Configuration** - Update in `legged_robot_config.py`:
   - Mass distribution
   - Leg lengths
   - Joint limits
   - Motor torque limits
   - Control frequencies
3. **Initial Joint Positions** - Anymal's default stance
4. **Training Hyperparameters** - May need tuning

**What Stays the Same:**
- Isaac Gym environment
- PPO algorithm
- Domain randomization techniques
- Training pipeline

---

## 4. CRITICAL DISCOVERY: Anymal Policies Already Exist!

### ETH Zurich's Legged Gym

**Repository:** [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
**Maintainer:** Nikita Rudin, ETH Zurich Robotic Systems Lab
**Paper:** https://arxiv.org/abs/2109.11978

### What It Provides

**The ORIGINAL quadruped locomotion framework in Isaac Gym!**

- ✅ **Anymal C locomotion** already implemented and trained
- ✅ Same technology as walk-these-ways
- ✅ Includes actuator network for sim-to-real
- ✅ Domain randomization (friction, mass, terrain, noise)
- ✅ Training takes <30 min on A100 GPU

### Key Features

```
Supported Robots:
  - ANYmal C (primary)
  - ANYmal D
  - Unitree A1
  - Unitree Go1 (added later)
  - Custom robots (configurable)

Capabilities:
  - Terrain: flat, rough, stairs, slopes
  - Commands: [vx, vy, vyaw] (same as our system!)
  - Training time: <30 min on A100
  - Sim-to-real: tested on real ANYmal hardware
```

### Migration Note

With NVIDIA shifting from Isaac Gym → Isaac Sim → Isaac Lab, the legged_gym repo will receive **limited future updates**. However, the trained policies and framework remain valid.

---

## 5. Our Current Setup

### What We Have (Go1)

```
MAPush Environment
  ↓
Go1 Robot (12 DOF)
  ↓
Mid-level RL Policy (outputs [vx, vy, vyaw])
  ↓
walk-these-ways Locomotion Policy (converts to joint positions)
  ↓
Actuator Network (converts to torques)
  ↓
Isaac Gym Physics
```

**Files:**
- `./mqe/utils/locomotion_checkpoints/walk_these_ways/body_latest.jit`
- `./mqe/utils/locomotion_checkpoints/walk_these_ways/adaptation_module_latest.jit`

### What We Need (Anymal)

```
MAPush Environment
  ↓
Anymal Robot (12 DOF)
  ↓
Mid-level RL Policy (outputs [vx, vy, vyaw])
  ↓
??? Anymal Locomotion Policy ???  <-- MISSING
  ↓
??? Anymal Actuator Network ???    <-- MISSING
  ↓
Isaac Gym Physics
```

---

## 6. Options for Anymal Integration

### Option A: Use Existing Anymal Policy from legged_gym ⭐

**Pros:**
- ✅ Already exists and trained
- ✅ Same framework (Isaac Gym + PPO)
- ✅ Tested on real Anymal hardware
- ✅ Command interface: [vx, vy, vyaw] (matches our system!)
- ✅ Can download and use immediately

**Cons:**
- ⚠️ May need adaptation to MAPush's specific terrain/task
- ⚠️ Might have different observation space than walk-these-ways
- ⚠️ Need to verify compatibility with our codebase

**Effort:** Medium (1-2 weeks integration + testing)

### Option B: Retrain Anymal Policy from Scratch

**Pros:**
- ✅ Fully customized for MAPush environment
- ✅ Guaranteed compatibility
- ✅ Can optimize for our specific task

**Cons:**
- ❌ Requires GPU with ≥10GB VRAM
- ❌ Training time: <30 min to hours (depending on complexity)
- ❌ Need to set up training pipeline
- ❌ Risk of suboptimal performance

**Effort:** High (2-4 weeks setup + training + validation)

### Option C: Use Direct Position Control (No Locomotion Policy)

**Pros:**
- ✅ Simple implementation
- ✅ No locomotion policy needed
- ✅ Can start immediately

**Cons:**
- ❌ Mid-level RL outputs 12D joint targets directly
- ❌ Different architecture than Go1 (no hierarchical control)
- ❌ May not learn coordination as well
- ❌ Less sim-to-real transferability

**Effort:** Low (1 week)

---

## 7. Detailed Comparison: Go1 vs Anymal Policies

| Aspect | Go1 (walk-these-ways) | Anymal (legged_gym) |
|--------|----------------------|---------------------|
| **Training Framework** | Isaac Gym + PPO | Isaac Gym + PPO |
| **Institution** | MIT Improbable AI | ETH Zurich RSL |
| **Robot DOF** | 12 (3 per leg) | 12 (3 per leg) |
| **Command Space** | [vx, vy, vyaw] | [vx, vy, vyaw] |
| **Actuator Network** | ✅ Yes | ✅ Yes |
| **Domain Randomization** | ✅ Yes | ✅ Yes |
| **Sim-to-Real** | ✅ Tested on Go1 | ✅ Tested on ANYmal |
| **Training Time** | Hours | <30 min (A100) |
| **Availability** | In our codebase | Need to download |

---

## 8. Integration Effort Estimate

### If Using Existing Anymal Policy (Option A)

**Week 1:**
- Day 1-2: Download legged_gym, extract Anymal policy
- Day 3-4: Integrate into MAPush (modify `mqe/envs/anymal/anymal.py`)
- Day 5: Test in isolation (simple environment)

**Week 2:**
- Day 1-2: Test in MAPush environment
- Day 3-4: Debug observation/action interface
- Day 5: Validate locomotion quality

**Week 3 (Optional):**
- Fine-tune if needed
- Compare performance with Go1

**Total:** 2-3 weeks

---

## 9. Recommendation

### Best Path Forward: Option A (Use Existing Anymal Policy)

**Rationale:**
1. ✅ **Proven technology** - Already tested on real Anymal
2. ✅ **Time efficient** - No retraining needed
3. ✅ **Same architecture** - Both use hierarchical control
4. ✅ **Compatible command space** - [vx, vy, vyaw] matches our system
5. ✅ **Lower risk** - Established baseline performance

**Steps:**
1. Clone [leggedrobotics/legged_gym](https://github.com/leggedrobotics/legged_gym)
2. Extract Anymal policy checkpoints
3. Create `mqe/envs/anymal/` directory structure (mirroring Go1)
4. Modify policy loading code to use Anymal checkpoints
5. Test and validate

**Expected Outcome:**
- Heterogeneous agents (Go1 + Anymal) both with hierarchical control
- True morphological diversity (two different quadrupeds)
- Both proven to work in Isaac Gym

---

## 10. Alternative: If Locomotion Policy Integration Fails

### Fallback to Direct Control

If integrating Anymal's locomotion policy proves too difficult:

```python
# mqe/envs/anymal/anymal_config.py
control_type = 'P'  # Position control (not hierarchical 'C')
```

This allows:
- RL policy directly outputs 12D joint targets
- No locomotion policy layer needed
- **Different from Go1** (which uses hierarchical control)
- Still viable for heterogeneous training, just different architectures

---

## Sources

**Walk These Ways:**
- [GitHub Repository](https://github.com/Improbable-AI/walk-these-ways)

**Legged Gym (ETH Zurich):**
- [GitHub Repository](https://github.com/leggedrobotics/legged_gym)
- [Paper: Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2109.11978)

**Related Projects:**
- [MQE: Multi-agent Quadruped Environment](https://github.com/ziyanx02/multiagent-quadruped-environment)
- [Awesome Legged Locomotion Learning](https://github.com/gaiyi7788/awesome-legged-locomotion-learning)

---

## Conclusion

**To directly answer your question:**

1. **Is the Go1 policy Isaac Gym specific?**
   - No. It's **trained** in Isaac Gym but is a **robot-specific PyTorch model** that can run anywhere.

2. **Can it be retrained for other robots?**
   - Yes! The framework is robot-agnostic. You just need to configure for the target robot.

3. **Does Anymal policy already exist?**
   - **YES!** ETH Zurich's legged_gym includes trained Anymal policies.

4. **Can we use it?**
   - **YES!** Same command interface [vx, vy, vyaw], same hierarchical architecture.

**Bottom Line:** We CAN integrate Anymal with a locomotion policy. The policy exists, it's proven, and it uses the same control interface as Go1. This makes **Option B (Go1 + Anymal heterogeneous quadrupeds)** significantly more viable than previously thought!

---

**Date:** 2026-01-19
**Status:** Analysis complete - Anymal integration IS feasible with existing policies
