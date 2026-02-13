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

    # Calculate metrics at checkpoints
    steps=()
    for ((i=1; i<=num_steps/10000000; i++)); do
        steps+=("${i}0000000")
    done

    log_dir="$current_dir/log/MQE/go1push_vel/$exp_name"
    last_folder=$(ls -dt $log_dir/*/ 2>/dev/null | head -n 1)

    echo "last_folder: $last_folder"
    for step in "${steps[@]}"; do
        filename="rl_model_${step}_steps/module.pt"
        test_checkpoint="$last_folder/checkpoints/$filename"
        python ./openrl_ws/test.py --num_envs 300 \
                --algo "$algo" \
                --task go1push_vel \
                --checkpoint "$test_checkpoint" \
                --agent0 $agent0 \
                --agent1 $agent1 \
                --test_mode calculator \
                --headless >> $last_folder/metrics.txt 2>&1
    done

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
