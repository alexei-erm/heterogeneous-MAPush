#!/bin/bash
# Velocity-MAPush HAPPO training script
exp_name="velocity_happo"

# Agent configuration
agent0='go1'
agent1='go1'

# Set PYTHONPATH
export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH

# Train with HAPPO
python HARL/harl_mapush/train.py \
    --task go1push_vel \
    --exp_name $exp_name \
    --n_rollout_threads 500 \
    --num_env_steps 200000000 \
    --seed 1 \
    --agent0 $agent0 \
    --agent1 $agent1 \
    --use_relative_obs_critic False

# Test at intervals
echo "Training complete. Run test.py with checkpoint for evaluation."
