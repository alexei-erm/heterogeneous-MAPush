# START HERE - Next Session Quick Reference

**Last Updated**: 2026-01-19
**Current Status**: Anymal C policy trained ✅, ready for MAPush integration ⏳

---

## What's Done ✅

1. **Anymal C locomotion policy trained** in Isaac Gym (500 iterations, 8.5 min)
2. **Policy exported to JIT format** at `/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/`
3. **Policy verified working** with play.py visualization
4. **Complete documentation** in `ANYMAL_C_TRAINING_SETUP.md` and `SESSION_SUMMARY_ANYMAL_C_INTEGRATION.md`

---

## What's Next ⏳

**Goal**: Integrate Anymal C into MAPush for heterogeneous training with Go1

### Immediate Next Steps (in order)

1. **Create Anymal C directory**
   ```bash
   cd /home/gvlab/new-universal-MAPush
   mkdir -p mqe/envs/anymal_c
   ```

2. **Create config file** by adapting Go1 config:
   ```bash
   # Copy and modify
   cp mqe/envs/go1/go1_config.py mqe/envs/anymal_c/anymal_c_config.py
   ```

   **Key changes needed**:
   - URDF path: `resources/robots/anymal_c/urdf/anymal_c.urdf`
   - Policy path: `resources/robots/anymal_c/policy_500.jit/`
   - Joint names (check URDF)
   - Default joint angles
   - Foot name (check URDF)

3. **Create robot class** by adapting Go1:
   ```bash
   cp mqe/envs/go1/go1.py mqe/envs/anymal_c/anymal_c.py
   ```

   **Critical change**: Policy loading (line ~396)
   ```python
   # Go1 uses body + adaptation_module
   # Anymal C uses single policy file:
   policy = torch.jit.load(self.cfg.control.locomotion_policy_dir + '/policy_1.pt')
   ```

4. **Register in robot_registry.py**

5. **Test standalone** before heterogeneous integration

---

## Key Files to Reference

### Templates (copy and modify)
- `/home/gvlab/new-universal-MAPush/mqe/envs/go1/go1_config.py` → template for anymal_c_config.py
- `/home/gvlab/new-universal-MAPush/mqe/envs/go1/go1.py` → template for anymal_c.py

### Documentation
- `/home/gvlab/new-universal-MAPush/SESSION_SUMMARY_ANYMAL_C_INTEGRATION.md` - Full details
- `/home/gvlab/new-universal-MAPush/ANYMAL_C_TRAINING_SETUP.md` - Training setup

### Resources
- **Policy**: `/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/policy_1.pt`
- **URDF**: `/home/gvlab/new-universal-MAPush/resources/robots/anymal_c/urdf/anymal_c.urdf`
- **Training checkpoints**: `/home/gvlab/legged_gym/logs/flat_anymal_c_rtx2070/Jan19_18-51-31_/`

---

## Critical Differences: Go1 vs Anymal C

| Aspect | Go1 | Anymal C |
|--------|-----|----------|
| Policy format | body.jit + adaptation_module.jit | Single policy_1.pt |
| Observation dim | 48 (+ history) | 48 (+ history) |
| Action dim | 12 | 12 |
| Policy loading | Two models | One model |
| Adaptation module | Yes | No |
| Joint names | FL_hip_joint, etc. | LF_HAA, LF_HFE, etc. (check URDF) |

---

## Quick Commands

### Activate training env (if needed)
```bash
conda activate anymal_training
```

### Activate MAPush env
```bash
conda activate mapush
```

### Verify policy exists
```bash
ls -lh /home/gvlab/new-universal-MAPush/resources/robots/anymal_c/policy_500.jit/
```

### Check URDF for joint names
```bash
grep "joint name" /home/gvlab/new-universal-MAPush/resources/robots/anymal_c/urdf/anymal_c.urdf
```

---

## Expected Issues & Solutions

### Issue: Policy won't load
**Check**: Is the path correct in config?
```python
locomotion_policy_dir = "./resources/robots/anymal_c/policy_500.jit"
```

### Issue: Joint name mismatch
**Fix**: Parse URDF to get exact names, update config's `default_joint_angles`

### Issue: Dimension mismatch
**Check**: Observation preprocessing in `preprocess_action()` fills all 48 dims correctly

### Issue: Robot falls immediately
**Check**:
- Default joint angles produce stable standing pose
- Torque limits are reasonable
- Actuator network is compatible

---

## Testing Checklist

Before moving to heterogeneous setup:
- [ ] Anymal C environment creates without errors
- [ ] Policy loads successfully
- [ ] Robot spawns and doesn't fall
- [ ] Velocity commands work (vx, vy, vyaw)
- [ ] No NaN/Inf in observations or actions
- [ ] Physics stable for 10+ seconds

---

## Available Robots in legged_gym

If Anymal C integration works and you want to add more robots:
- **Unitree A1**: Similar to Go1, should be easy
- **Anymal B**: Older ANYmal version
- **Cassie**: Agility Robotics biped (different morphology)

All can be trained with same process as Anymal C.

---

## Training Command (for reference)
```bash
cd /home/gvlab/legged_gym
conda activate anymal_training
python legged_gym/scripts/train.py --task=anymal_c_flat_rtx2070 --headless --pipeline gpu
```

---

**Read `SESSION_SUMMARY_ANYMAL_C_INTEGRATION.md` for complete details**
