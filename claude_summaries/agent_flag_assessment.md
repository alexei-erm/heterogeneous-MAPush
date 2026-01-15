# Agent Flag Assessment: --agent0/--agent1 Approach

**Date:** 2026-01-15
**Task:** Assess feasibility of changing from `--hetero_agent` to `--agent0 <robot> --agent1 <robot>`
**Decision:** ✅ **MODIFY CURRENT IMPLEMENTATION** (Recommended)

---

## Executive Summary

**Recommendation:** Modify the current implementation. The change is **straightforward** and requires modifications to only **8 files**, most of which are simple string replacements or argument additions.

**Rationale:**
1. The underlying heterogeneous system is **fully generic** and already supports arbitrary agent orderings
2. Only the **entry points** (training scripts) hardcode agent0=go1
3. All infrastructure from today's work (differential drive, buffer handling, dynamic classes, etc.) remains fully compatible
4. Starting from main branch would require **redoing all 6 major fixes** from today's session
5. Estimated implementation time: **30-45 minutes** vs. **6+ hours** to redo from main

---

## Current Implementation Analysis

### How `--hetero_agent` Works

**Current Flag:**
```bash
python train.py --hetero_agent jackal
```

**What Happens:**
1. Training scripts parse `--hetero_agent jackal`
2. Agent list is hardcoded as `['go1', hetero_agent]` in 3 places
3. `make_hetero_env()` receives `['go1', 'jackal']`
4. System creates heterogeneous environment with agent0=go1, agent1=jackal

### Hardcoding Locations (8 files)

#### 1. **Print Statements Only (5 files)** - Cosmetic changes
These files only print the hardcoded assumption but don't enforce it:

| File | Line | Code |
|------|------|------|
| `HARL/harl_mapush/train.py` | 62 | Help text: "Agent0 will be Go1, Agent1 will be..." |
| `HARL/harl_mapush/test.py` | 224 | `print(f"agent0=go1, agent1={hetero_agent}")` |
| `HARL/harl_mapush/test.py` | 468 | `print(f"agent0=go1, agent1={args.hetero_agent}")` |
| `openrl_ws/train.py` | 26 | `print(f"agent0=go1, agent1={hetero_agent}")` |
| `openrl_ws/test.py` | 110 | `print(f"agent0=go1, agent1={hetero_agent}")` |

**Impact:** Low - just update print statements

#### 2. **Actual Logic (3 files)** - Requires minor code changes

**`mqe/envs/utils.py:110`** (custom_cfg function)
```python
# Current
cfg.hetero.hetero_agent_types = ['go1', hetero_agent]

# New approach
cfg.hetero.hetero_agent_types = [args.agent0, args.agent1]
```

**`HARL/harl/envs/mapush/mapush_env.py:77`** (MAPushEnv initialization)
```python
# Current
agent_types = ['go1', hetero_agent]

# New approach
agent_types = [args.agent0, args.agent1]
```

**`task/cuboid/config.py:16`** (Default config)
```python
# Current
hetero_agent_types = ['go1', 'go1']  # Default: homogeneous

# New approach
hetero_agent_types = ['go1', 'go1']  # Keep as default, overridden by flags
```

**Impact:** Medium - requires passing agent0/agent1 instead of single hetero_agent

### Key Insight: System is Already Generic!

**Critical Realization:** The core heterogeneous system (`HeteroRobot`, `merge_hetero_configs()`, `create_hetero_config()`, etc.) is **completely generic**. It doesn't hardcode agent0=go1 anywhere!

Example from `mqe/envs/base/hetero_robot.py:74-76`:
```python
print(f"\n[HeteroRobot] Initializing with {len(self.hetero_agent_types)} agent types:")
for i, robot_name in enumerate(self.hetero_agent_types):
    print(f"  Agent {i}: {robot_name} ({self.hetero_action_dims[i]} DOF)")
```

It just receives a list `self.hetero_agent_types` and works with **any** robots in **any** order!

---

## Proposed New Approach

### New Flags

```bash
# Homogeneous (default - both go1)
python train.py

# Heterogeneous (go1 + jackal)
python train.py --agent0 go1 --agent1 jackal

# Heterogeneous (jackal + go1) - reversed!
python train.py --agent0 jackal --agent1 go1

# Heterogeneous (jackal + jackal)
python train.py --agent0 jackal --agent1 jackal

# Future: any robot combinations
python train.py --agent0 spot --agent1 anymal
```

### Argument Parser Changes

**Old:**
```python
parser.add_argument("--hetero_agent", type=str, default=None,
    help="Enable heterogeneous agents. Specify second robot type.")
```

**New:**
```python
parser.add_argument("--agent0", type=str, default='go1',
    help="Robot type for agent 0 (default: go1)")
parser.add_argument("--agent1", type=str, default='go1',
    help="Robot type for agent 1 (default: go1)")
```

### Detection of Hetero Mode

**Old:**
```python
is_hetero = (hetero_agent is not None)
```

**New:**
```python
is_hetero = (agent0 != agent1)
# OR explicitly check both are provided and different
```

---

## Required Changes

### Step-by-Step Modification Plan

#### **Step 1: Update Training Scripts** (4 files - 15 min)

**Files:**
- `HARL/harl_mapush/train.py`
- `HARL/harl_mapush/test.py`
- `openrl_ws/train.py`
- `openrl_ws/test.py`

**Changes:**
1. Replace `--hetero_agent` argument with `--agent0` and `--agent1`
2. Update `is_hetero` detection logic
3. Update print statements to show both agents
4. Pass both agents to downstream functions

**Example (train.py):**
```python
# Add arguments
parser.add_argument("--agent0", type=str, default='go1',
    help="Robot type for agent 0 (default: go1)")
parser.add_argument("--agent1", type=str, default='go1',
    help="Robot type for agent 1 (default: go1)")

# Detect hetero mode
is_hetero = (args.agent0 != args.agent1)
if is_hetero:
    print(f"Agent types: HETEROGENEOUS (agent0={args.agent0}, agent1={args.agent1})")
else:
    print(f"Agent types: HOMOGENEOUS ({args.agent0})")

# Pass to env creation
env_args = {
    "agent0": args.agent0,
    "agent1": args.agent1,
    # ... other args
}
```

#### **Step 2: Update Environment Creation** (2 files - 10 min)

**Files:**
- `mqe/envs/utils.py`
- `HARL/harl/envs/mapush/mapush_env.py`

**Changes:**
1. Accept `agent0` and `agent1` parameters instead of `hetero_agent`
2. Create agent_types list from both parameters
3. Update print statements

**Example (utils.py custom_cfg):**
```python
def custom_cfg(args, individualized_rewards=False, ..., agent0='go1', agent1='go1'):
    def fn(cfg:LeggedRobotFieldCfg):
        # ... existing code ...

        # Detect hetero mode
        is_hetero = (agent0 != agent1)

        if is_hetero and hasattr(cfg, 'hetero'):
            cfg.hetero.use_hetero = True
            cfg.hetero.hetero_agent_types = [agent0, agent1]
            print(f"[custom_cfg] Enabled hetero mode: agent0={agent0}, agent1={agent1}")

        # ... rest of function ...
```

**Example (mapush_env.py):**
```python
# Get agent types
agent0 = env_args.get("agent0", "go1")
agent1 = env_args.get("agent1", "go1")
is_hetero = (agent0 != agent1)

if is_hetero:
    from mqe.envs.utils import make_hetero_env
    agent_types = [agent0, agent1]
    print(f"[MAPushEnv] Creating heterogeneous environment: {agent_types}")

    self.env, self.env_cfg = make_hetero_env(
        args.task,
        agent_types,  # Now can be any combination!
        args,
        custom_cfg=custom_cfg(args, ..., agent0=agent0, agent1=agent1)
    )
```

#### **Step 3: Update Default Config** (1 file - 2 min)

**File:** `task/cuboid/config.py`

**Changes:**
- Update comments to reflect new flag approach
- Default remains `['go1', 'go1']` (homogeneous)

```python
class hetero:
    use_hetero = False
    hetero_agent_types = ['go1', 'go1']  # Default: homogeneous Go1
    # Use --agent0 <robot> --agent1 <robot> flags to specify heterogeneous agents
    # Example: --agent0 go1 --agent1 jackal
```

#### **Step 4: Update Documentation** (2-3 files - 10 min)

**Files:**
- `claude_summaries/jackal_integration.md`
- `claude_summaries/hetero_implementation_progress.md`
- `claude_summaries/heterogeneous_agent_implementation.md` (if exists)

**Changes:**
- Update usage examples to show new flags
- Update "How to Use" sections
- Add examples of different agent combinations

---

## Comparison: Modify vs. Start From Main

### Option A: Modify Current Implementation ✅ (RECOMMENDED)

**Pros:**
- ✅ Quick implementation (30-45 minutes)
- ✅ Keeps all today's critical fixes:
  - Differential drive controller
  - Unified action space (3 DOF for both robots)
  - Per-agent torque limits
  - Dynamic HeteroTask class creation
  - Config inheritance preservation
  - Terrain bug fix (LeggedRobotField inheritance)
- ✅ System is already generic - no architectural changes needed
- ✅ Only 8 files to modify (mostly print statements)
- ✅ Low risk - core logic unchanged

**Cons:**
- ⚠️ Buffer initialization issue still needs fixing (already planned)
- ⚠️ Need to update documentation

**Estimated Time:** 1 hour total
- 30-45 min for implementation
- 15-20 min for documentation updates

---

### Option B: Start From Main Branch ❌ (NOT RECOMMENDED)

**Pros:**
- ✅ Clean slate
- ✅ No buffer initialization issue (but will arise once hetero is implemented)

**Cons:**
- ❌ Must redo ALL 6 major fixes from today:
  1. **Unified Action Space** - Remove per-agent dimensions, add differential drive (2-3 hours)
  2. **Terrain Configuration** - Fix inheritance (LeggedRobotField) (30 min)
  3. **Per-Agent Torque Limits** - Implement flexible system (1 hour)
  4. **Dynamic HeteroTask Class** - Solve NPC creation issue (1 hour)
  5. **Config Preservation** - Fix merge_hetero_configs (30 min)
  6. **Differential Drive Controller** - Implement for Jackal (1 hour)
- ❌ Buffer initialization will still need fixing (30-60 min)
- ❌ Need to re-test everything
- ❌ Risk of introducing new bugs

**Estimated Time:** 6-8 hours total
- 6-7 hours to re-implement today's fixes
- 30-45 min for new flag approach
- 30-60 min for buffer fix
- 1 hour for testing and debugging

---

## Risk Assessment

### Modify Current Implementation (Option A)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Breaking existing functionality | Low | Medium | Test with homogeneous mode first |
| Backward compatibility issues | Low | Low | Old checkpoints won't work anyway |
| Missing edge cases | Low | Low | Validation in robot_registry |
| Buffer initialization still broken | High | High | Already planned fix (separate from this change) |

**Overall Risk:** ⚠️ **LOW** - Changes are isolated and well-understood

### Start From Main (Option B)

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Re-introducing today's bugs | High | High | Careful re-implementation |
| Missing subtle fixes | Medium | Medium | Thorough code review |
| Time overrun | High | Medium | Already 90% complete with Option A |
| New bugs from re-implementation | Medium | High | Extensive testing |

**Overall Risk:** 🚨 **HIGH** - Significant re-work with high error potential

---

## Implementation Checklist

If proceeding with **Option A** (recommended):

### Phase 1: Core Modifications (30-45 min)

- [ ] Update `HARL/harl_mapush/train.py`
  - [ ] Add `--agent0` and `--agent1` arguments
  - [ ] Remove `--hetero_agent` argument
  - [ ] Update hetero detection logic
  - [ ] Update print statements
  - [ ] Pass both agents to env_args

- [ ] Update `HARL/harl_mapush/test.py`
  - [ ] Same changes as train.py
  - [ ] Update `test_viewer_mode()` signature

- [ ] Update `openrl_ws/train.py`
  - [ ] Same changes as HARL train.py

- [ ] Update `openrl_ws/test.py`
  - [ ] Same changes as HARL test.py

- [ ] Update `mqe/envs/utils.py`
  - [ ] Modify `custom_cfg()` signature
  - [ ] Update agent_types construction

- [ ] Update `HARL/harl/envs/mapush/mapush_env.py`
  - [ ] Accept agent0/agent1 from env_args
  - [ ] Update agent_types construction
  - [ ] Update print statements

- [ ] Update `task/cuboid/config.py`
  - [ ] Update comments

### Phase 2: Testing (15 min)

- [ ] Test homogeneous mode (agent0=go1, agent1=go1)
  ```bash
  python train.py --agent0 go1 --agent1 go1 --num_env_steps 1000
  ```

- [ ] Test heterogeneous mode (agent0=go1, agent1=jackal)
  ```bash
  python train.py --agent0 go1 --agent1 jackal --num_env_steps 1000
  ```

- [ ] Test reversed hetero (agent0=jackal, agent1=go1)
  ```bash
  python train.py --agent0 jackal --agent1 go1 --num_env_steps 1000
  ```

- [ ] Test invalid robot name (should error gracefully)
  ```bash
  python train.py --agent0 invalid_robot --agent1 go1
  ```

### Phase 3: Documentation (15-20 min)

- [ ] Update `claude_summaries/jackal_integration.md`
  - [ ] Replace all `--hetero_agent` examples
  - [ ] Add new flag examples

- [ ] Update `claude_summaries/hetero_implementation_progress.md`
  - [ ] Document flag change rationale
  - [ ] Update usage examples

- [ ] Create migration guide (optional)
  - [ ] Old vs. new flag comparison
  - [ ] Examples of equivalent commands

---

## Example Usage After Implementation

### Basic Examples

```bash
# Homogeneous training (default - both go1)
python HARL/harl_mapush/train.py \
  --exp_name homogeneous_baseline

# Heterogeneous training (go1 + jackal)
python HARL/harl_mapush/train.py \
  --exp_name go1_jackal_hetero \
  --agent0 go1 \
  --agent1 jackal

# Heterogeneous training (jackal + go1) - reversed order
python HARL/harl_mapush/train.py \
  --exp_name jackal_go1_hetero \
  --agent0 jackal \
  --agent1 go1

# Both Jackal (homogeneous but different from default)
python HARL/harl_mapush/train.py \
  --exp_name jackal_homogeneous \
  --agent0 jackal \
  --agent1 jackal
```

### Testing

```bash
# Test heterogeneous checkpoint (calculator mode)
python HARL/harl_mapush/test.py \
  --checkpoint ./results/.../checkpoints/10M \
  --mode calculator \
  --num_episodes 100 \
  --num_envs 300 \
  --agent0 go1 \
  --agent1 jackal

# Test heterogeneous checkpoint (viewer mode)
python HARL/harl_mapush/test.py \
  --checkpoint ./results/.../checkpoints/10M \
  --mode viewer \
  --num_episodes 5 \
  --agent0 go1 \
  --agent1 jackal
```

---

## Future Extensibility

### Adding New Robots

With the new flag approach, adding new robots is trivial:

**Step 1:** Register robot in `mqe/envs/robot_registry.py`
```python
'spot': {
    'class_path': 'mqe.envs.spot.spot.Spot',
    'config_path': 'mqe.envs.spot.spot_config.SpotCfg',
    'default_control': 'C',  # Hierarchical
    'num_actions': 3,  # [vx, vy, vyaw]
    'description': 'Boston Dynamics Spot quadruped'
}
```

**Step 2:** Train with any combination
```bash
# Go1 + Spot
python train.py --agent0 go1 --agent1 spot

# Jackal + Spot
python train.py --agent0 jackal --agent1 spot

# Spot + Spot
python train.py --agent0 spot --agent1 spot
```

**No training script changes needed!** ✨

### Multi-Agent (>2 agents)

Future expansion to 3+ agents is straightforward:
```bash
python train.py --agent0 go1 --agent1 jackal --agent2 spot
```

Just extend the argument parsing and pass `agent_types = [agent0, agent1, agent2]` to the system.

---

## Conclusion

### Recommendation: **MODIFY CURRENT IMPLEMENTATION** ✅

**Rationale:**
1. **Minimal effort** - 30-45 minutes of coding vs. 6+ hours to redo
2. **Low risk** - Only entry points change, core system untouched
3. **Preserves progress** - Keeps all 6 critical fixes from today's session
4. **Generic architecture** - System already supports arbitrary agent combinations
5. **Clean design** - New flags are more intuitive and flexible

### Next Steps

**Immediate (Today's Session):**
1. Implement new flag approach (30-45 min)
2. Test basic functionality (15 min)
3. Update documentation (15 min)
4. **Total: ~1 hour**

**Next Session:**
1. Fix buffer initialization issue (30-60 min)
2. Run comprehensive integration tests
3. Full training run to validate
4. Performance comparison vs. homogeneous baseline

---

## Appendix: File Dependency Map

```
Training Scripts (Entry Points)
├── HARL/harl_mapush/train.py
│   └─> HARL/harl/envs/mapush/mapush_env.py
│       └─> mqe/envs/utils.py (make_hetero_env + custom_cfg)
│           └─> mqe/utils/hetero_config.py (create_hetero_config)
│               └─> mqe/envs/base/hetero_robot.py (HeteroRobot)
│                   └─> [Generic - no hardcoding]
│
├── HARL/harl_mapush/test.py
│   └─> [Same chain as train.py]
│
├── openrl_ws/train.py
│   └─> mqe/envs/utils.py (custom_cfg)
│
└── openrl_ws/test.py
    └─> mqe/envs/utils.py (custom_cfg)

Config Files
└── task/cuboid/config.py
    └─> [Default config - overridden by flags]
```

**Hardcoding only exists in the top layer (entry points)!**
**All underlying layers (hetero_robot.py, hetero_config.py, etc.) are fully generic!**

---

**Status:** Ready for implementation
**Confidence:** High (95%)
**Time to implement:** 1 hour
**Risk level:** Low

🚀 **Let's proceed with modifying the current implementation!**
