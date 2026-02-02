exp_name="heterogenMAPPObaseline_go1_anymal"
current_dir=$(pwd)
algo="ppo"
script_path=$(realpath "${BASH_SOURCE[0]}")
# script_path=$(realpath $0)
script_dir=$(dirname "$script_path")
test_mode=$1

# Agent configuration (used in both train and test modes)
agent0='go1'
agent1='anymal_c' # or cassie, anymal_c or go1

# Set PYTHONPATH to include the project root
export PYTHONPATH=/home/gvlab/new-universal-MAPush:$PYTHONPATH

# update config
python ./openrl_ws/update_config.py --filepath $script_dir/config.py

if [ $test_mode = False ]; then
    # train
    num_envs=500
    num_steps=150000000
    checkpoint=/None  # "/results/07-28-13_task1/checkpoints/rl_model_100000000_steps/module.pt"

    python ./openrl_ws/train.py  --num_envs $num_envs --train_timesteps $num_steps\
    --algo $algo \
    --config ./openrl_ws/cfgs/ppo.yaml \
    --seed 1 \
    --exp_name  $exp_name \
    --task go1push_mid \
    --use_tensorboard \
    --headless \
    --agent0 $agent0 \
    --agent1 $agent1 \
    --layer_N 2 \
    --hidden_size 128 \
    --baseline_mappo_rewards True

    #   --checkpoint $current_dir$checkpoint \


    # calculate success rate
    steps=()
    for ((i=1; i<=num_steps/10000000; i++)); do
        steps+=("${i}0000000")
    done
    target_dir=$current_dir/results
    last_folder=$(ls -d $target_dir/*/ | sort | tail -n 1)
    echo "last_folder: $last_folder"
    for step in "${steps[@]}"; do
        filename="rl_model_${step}_steps/module.pt"
        test_checkpoint="$last_folder/checkpoints/$filename"
        python ./openrl_ws/test.py --num_envs 300 \
                --algo "$algo" \
                --task go1push_mid \
                --checkpoint "$test_checkpoint" \
                --agent0 $agent0 \
                --agent1 $agent1 \
                --test_mode calculator \
                --headless  >> $last_folder/success_rate.txt 2>&1
    done

else
# test
test_checkpoint="/home/gvlab/new-universal-MAPush/log/MQE/go1push_mid/cuboid/run1/checkpoints/rl_model_150000000_steps/module.pt"
python ./openrl_ws/test.py --num_envs 1 \
        --algo "$algo" \
        --task go1push_mid \
        --checkpoint "$test_checkpoint" \
        --test_mode viewer \
        --agent0 $agent0 \
        --agent1 $agent1 
#       --record_video
fi
