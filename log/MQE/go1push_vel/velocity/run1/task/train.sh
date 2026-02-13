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

    # NOTE: Calculator test mode is skipped for velocity task.
    # The velocity task has no success criterion (THRESHOLD=-1.0, finished_buf always False),
    # so success_rate/collision_degree/collaboration_degree are meaningless.
    # Monitor training progress via tensorboard:
    #   rewards/velocity_tracking_reward, rewards/avg_direction_error,
    #   rewards/avg_speed_error, rewards/avg_box_angular_vel
    echo "Training complete. Check tensorboard for velocity metrics."

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
