#!/bin/bash
# MAPPO: Test all checkpoints in a run folder (calculator mode)
# Automatically extracts agent0, agent1, algo, task from command.txt
# Usage: ./test_all_checkpoints.sh <run_folder>
# Example: ./test_all_checkpoints.sh log/MQE/go1push_mid/homogenMAPPObaseline_go1/run1

if [ $# -lt 1 ]; then
    echo "Usage: $0 <run_folder>"
    echo "Example: $0 log/MQE/go1push_mid/homogenMAPPObaseline_go1/run1"
    exit 1
fi

run_folder=$1

cd /home/gvlab/new-universal-MAPush
export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH

# Check if command.txt exists
if [ ! -f "$run_folder/command.txt" ]; then
    echo "ERROR: $run_folder/command.txt not found"
    exit 1
fi

# Parse agent0, agent1, algo, task from command.txt
agent0=$(grep -oP '(?<=--agent0 )\S+' "$run_folder/command.txt" | head -1)
agent1=$(grep -oP '(?<=--agent1 )\S+' "$run_folder/command.txt" | head -1)
algo=$(grep -oP '(?<=--algo )\S+' "$run_folder/command.txt" | head -1)
task=$(grep -oP '(?<=--task )\S+' "$run_folder/command.txt" | head -1)

echo "============================================"
echo "Run folder: $run_folder"
echo "Parsed from command.txt:"
echo "  agent0: $agent0"
echo "  agent1: $agent1"
echo "  algo: $algo"
echo "  task: $task"
echo "Results: $run_folder/success_rate.txt"
echo "============================================"
echo ""

for ckpt in $(ls -d $run_folder/checkpoints/rl_model_*_steps 2>/dev/null | sort -V); do
    echo "Testing: $ckpt"
    python ./openrl_ws/test.py --num_envs 300 \
            --algo "$algo" \
            --task "$task" \
            --checkpoint "$ckpt/module.pt" \
            --agent0 $agent0 \
            --agent1 $agent1 \
            --test_mode calculator \
            --headless  >> $run_folder/success_rate.txt 2>&1
done

echo ""
echo "Done! Results in: $run_folder/success_rate.txt"
