#!/bin/bash
# MAPPO: Test all checkpoints for velocity task (calculator mode)
# Extracts velocity-specific metrics (direction error, speed error, angular velocity)
# Usage: ./test_all_checkpoints_velocity.sh <run_folder>
# Example: ./test_all_checkpoints_velocity.sh log/MQE/go1push_vel/velocity/run1

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_folder>"
    echo "Example: $0 log/MQE/go1push_vel/velocity/run1"
    exit 1
fi

run_folder=$1

# If user passed the checkpoints dir directly, go up one level
if [[ "$(basename "$run_folder")" == "checkpoints" ]]; then
    run_folder="$(dirname "$run_folder")"
fi

cd /home/gvlab/new-universal-MAPush
export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH

# Parse agent0, agent1, algo from command.txt if it exists
if [ -f "$run_folder/command.txt" ]; then
    agent0=$(grep -oP '(?<=--agent0 )\S+' "$run_folder/command.txt" | head -1)
    agent1=$(grep -oP '(?<=--agent1 )\S+' "$run_folder/command.txt" | head -1)
    algo=$(grep -oP '(?<=--algo )\S+' "$run_folder/command.txt" | head -1)
else
    echo "WARNING: $run_folder/command.txt not found, using defaults"
    agent0="go1"
    agent1="go1"
    algo="ppo"
fi

# Default if not found in command.txt
agent0=${agent0:-go1}
agent1=${agent1:-go1}
algo=${algo:-ppo}

results_file="$run_folder/velocity_metrics.txt"

echo "============================================"
echo "Velocity-MAPush Checkpoint Testing"
echo "Run folder: $run_folder"
echo "  agent0: $agent0"
echo "  agent1: $agent1"
echo "  algo: $algo"
echo "  task: go1push_vel"
echo "Results: $results_file"
echo "============================================"
echo ""

echo "============================================" > "$results_file"
echo "Velocity-MAPush Calculator Results" >> "$results_file"
echo "Run: $run_folder" >> "$results_file"
echo "Date: $(date)" >> "$results_file"
echo "============================================" >> "$results_file"

checkpoint_count=0
for ckpt in $(ls -d $run_folder/checkpoints/rl_model_*_steps 2>/dev/null | sort -V); do
    ckpt_name=$(basename "$ckpt")
    checkpoint_count=$((checkpoint_count + 1))
    echo ""
    echo "[$checkpoint_count] Testing: $ckpt_name"
    echo "" >> "$results_file"
    echo "--- $ckpt_name ---" >> "$results_file"
    python ./openrl_ws/test.py --num_envs 300 \
            --algo "$algo" \
            --task go1push_vel \
            --checkpoint "$ckpt/module.pt" \
            --agent0 $agent0 \
            --agent1 $agent1 \
            --test_mode calculator \
            --headless  >> "$results_file" 2>&1
done

if [ $checkpoint_count -eq 0 ]; then
    echo "ERROR: No checkpoints found in $run_folder/checkpoints/"
    echo "Looking for: rl_model_*_steps directories"
    exit 1
fi

echo ""
echo "============================================"
echo "Done! Tested $checkpoint_count checkpoints."
echo "Results saved to: $results_file"
echo "============================================"
