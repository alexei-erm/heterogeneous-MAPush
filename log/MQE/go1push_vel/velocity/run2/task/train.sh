#!/bin/bash
# Velocity-MAPush MAPPO training script
exp_name="velocity"
current_dir=$(pwd)
algo="ppo"
script_path=$(realpath "${BASH_SOURCE[0]}")
script_dir=$(dirname "$script_path")
test_mode=${1:-False}

# Agent configuration
agent0='go1'
agent1='go1'

# Set PYTHONPATH
export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH

# NOTE: No update_config.py call needed — velocity task uses its own config
# registered via Go1PushVelCfg in mqe/envs/configs/go1_push_vel_config.py

if [ "$test_mode" = "False" ]; then
    # Train
    num_envs=500
    num_steps=200000000

    python ./openrl_ws/train.py \
        --num_envs $num_envs \
        --train_timesteps $num_steps \
        --algo $algo \
        --config ./openrl_ws/cfgs/ppo.yaml \
        --seed 1 \
        --exp_name $exp_name \
        --task go1push_vel \
        --use_tensorboard \
        --headless \
        --agent0 $agent0 \
        --agent1 $agent1 \
        --layer_N 2 \
        --hidden_size 128

    echo "Training complete. Running calculator mode on all checkpoints..."

    # Find the run folder (most recent run)
    run_folder=$(ls -dt log/MQE/go1push_vel/$exp_name/run* 2>/dev/null | head -1)
    if [ -z "$run_folder" ]; then
        echo "WARNING: Could not find run folder in log/MQE/go1push_vel/$exp_name/"
        echo "Skipping calculator mode."
    else
        results_file="$run_folder/velocity_metrics.txt"
        echo "============================================" > "$results_file"
        echo "Velocity-MAPush Calculator Results" >> "$results_file"
        echo "Run: $run_folder" >> "$results_file"
        echo "Date: $(date)" >> "$results_file"
        echo "============================================" >> "$results_file"

        for ckpt in $(ls -d $run_folder/checkpoints/rl_model_*_steps 2>/dev/null | sort -V); do
            ckpt_name=$(basename "$ckpt")
            echo ""
            echo "Testing: $ckpt_name"
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

        echo ""
        echo "Done! Results saved to: $results_file"
    fi

else
    # Test/viewer mode
    test_checkpoint="log/MQE/go1push_vel/$exp_name/run1/checkpoints/rl_model_100000000_steps/module.pt"
    python ./openrl_ws/test.py --num_envs 1 \
            --algo "$algo" \
            --task go1push_vel \
            --checkpoint "$test_checkpoint" \
            --test_mode viewer \
            --agent0 $agent0 \
            --agent1 $agent1
fi
