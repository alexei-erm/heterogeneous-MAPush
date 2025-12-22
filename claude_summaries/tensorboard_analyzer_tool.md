# TensorBoard Log Analyzer Tool

**Created:** December 21, 2025
**Location:** `scripts/analyze_tensorboard_logs.py`

## Purpose

A reusable analysis tool for diagnosing training issues from TensorBoard logs. This script extracts key metrics, computes statistics, and automatically identifies potential problems.

## Usage

### Basic Analysis
```bash
python scripts/analyze_tensorboard_logs.py <log_dir>
```

### Verbose Output (includes all metrics)
```bash
python scripts/analyze_tensorboard_logs.py <log_dir> --verbose
```

### Export JSON Report
```bash
python scripts/analyze_tensorboard_logs.py <log_dir> --json output.json
```

### Save Text Report
```bash
python scripts/analyze_tensorboard_logs.py <log_dir> --output report.txt
```

### Example
```bash
python scripts/analyze_tensorboard_logs.py \
    results/mapush/go1push_mid/happo/critic6/seed-00001-2025-12-19-16-53-40/logs \
    --verbose
```

## Features

### 1. Automatic Metric Extraction
- Loads TensorBoard event files
- Also parses `summary.json` for metrics in nested subdirectories (like agent entropy)

### 2. Diagnostic Checks

| Check | What it Detects |
|-------|-----------------|
| **Success Rate** | Zero/low success = training failed |
| **Reward Trend** | Stagnant/decreasing rewards = no learning |
| **Entropy** | Increasing entropy = policy becoming random (critical!) |
| **Value Loss** | Diverging critic |
| **Policy Loss** | Vanishing gradients |
| **Gradient Norms** | Exploding/vanishing gradients |
| **Reward Components** | Near-zero or decreasing reward components |

### 3. Status Levels

- 🔴 **CRITICAL**: Training has fundamental problems, likely failed
- 🟡 **WARNING**: Potential issues that may affect results
- 🟢 **OK**: Metric looks healthy

### 4. Trend Analysis

- **increasing**: Metric is going up over training
- **decreasing**: Metric is going down over training
- **stable**: Metric is roughly constant
- **volatile**: Metric has high variance

## Key Indicators for Failed Training

### Increasing Entropy (CRITICAL)
```
Entropy: 1.31 → 2.17 (+65.8%)
```
**Meaning**: Policy is becoming MORE random over time. In successful training, entropy should DECREASE as the policy converges to good actions. Increasing entropy indicates the policy finds no useful actions and drifts to maximum randomness.

**Common Causes**:
1. Actions too small to generate meaningful rewards
2. Reward signal too weak or sparse
3. Wrong hyperparameters (lr, entropy_coef)

### Zero Success Rate
```
Success Rate: 0.0000 throughout
```
**Meaning**: Task was never completed during training.

**Common Causes**:
1. Task is too hard for the current setup
2. Insufficient exploration
3. Action space mismatch

### Near-Zero Push Reward
```
push_reward: 0.000006 → 0.000016
```
**Meaning**: Box is barely moving. Push reward triggers when box velocity > 0.1 m/s.

**Common Causes**:
1. Actions too small to push effectively
2. Agents not reaching the box
3. Poor coordination between agents

## Output Format

### Console Output
```
================================================================================
TENSORBOARD LOG ANALYSIS REPORT
================================================================================
Log Directory: results/.../logs
Total Steps: 100,000,000
Metrics Found: 25

--------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------
CRITICAL: 2 critical issues found. Training likely failed.

--------------------------------------------------------------------------------
DIAGNOSTICS
--------------------------------------------------------------------------------
🔴 [CRITICAL] Success Rate
   Success rate is near zero (0.0000). Training has not learned the task.

🔴 [CRITICAL] Entropy
   Entropy is INCREASING (65.8%). Policy is becoming MORE random...
```

### JSON Output
```json
{
  "log_dir": "...",
  "total_steps": 100000000,
  "summary": "CRITICAL: 2 critical issues found.",
  "diagnostics": [
    {
      "name": "Success Rate",
      "status": "CRITICAL",
      "message": "Success rate is near zero...",
      "details": {...}
    }
  ],
  "metric_stats": {
    "mapush/success_rate": {
      "first_value": 0.0,
      "last_value": 0.0,
      "trend": "volatile"
    }
  }
}
```

## Comparing Multiple Runs

To compare runs, export JSON from each and compare:

```bash
# Export reports
python scripts/analyze_tensorboard_logs.py run1/logs --json run1_report.json
python scripts/analyze_tensorboard_logs.py run2/logs --json run2_report.json

# Compare in Python
import json
r1 = json.load(open("run1_report.json"))
r2 = json.load(open("run2_report.json"))

# Compare success rates
print(f"Run1 success: {r1['metric_stats']['mapush/success_rate']['last_value']}")
print(f"Run2 success: {r2['metric_stats']['mapush/success_rate']['last_value']}")
```

## Integration with Claude Sessions

When analyzing a training run, use this tool first:

```
1. Run analyzer:
   python scripts/analyze_tensorboard_logs.py <log_dir> --verbose

2. Check SUMMARY section for overall status

3. Review CRITICAL diagnostics first

4. For deeper analysis, check specific metrics:
   - Entropy trend (should decrease)
   - Success rate (should increase)
   - Reward components (identify which ones are zero)
```

## Dependencies

- Python 3.8+
- tensorboard (`pip install tensorboard`)
- numpy

## Extension Points

The analyzer can be extended by:

1. Adding new diagnostic checks in `run_diagnostics()`
2. Adding new metric patterns to the pattern lists
3. Modifying thresholds for WARNING/CRITICAL status
