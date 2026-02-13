
from openrl_ws.utils import make_env, get_args, MATWrapper
from openrl_ws.run_utils.run_config_saver import save_run_config
from mqe.envs.utils import custom_cfg
from openrl.utils.logger import Logger
from openrl.utils.callbacks.checkpoint_callback import CheckpointCallback
import shutil
import os

from datetime import datetime

def train(args):

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m-%d-%H")

    if args.algo == "sppo" or args.algo == "dppo":
        single_agent = True
    else:
        single_agent = False

    # Get agent types (defaults to go1 for both)
    agent0 = getattr(args, 'agent0', 'go1')
    agent1 = getattr(args, 'agent1', 'go1')
    is_hetero = (agent0 != agent1)

    # Baseline MAPPO rewards mode (default: True for MAPPO)
    # When True, uses ONLY original 7 MAPush rewards with TRUE ORIGINAL scales
    # Uses ORIGINAL distance formula WITH penalty term
    # Disables all HAPPO-specific rewards (proximity_penalty, goal_push_bonus, etc.)
    baseline_mappo_rewards = getattr(args, 'baseline_mappo_rewards', True)

    # Heavy box rewards mode (default: False)
    # When True, uses adjusted Exp 7 scales for 8kg box training
    # Uses 3x target, 2.5x push, 2x OCB, distance penalty REMOVED
    mappo_heavybox_rewards = getattr(args, 'mappo_heavybox_rewards', False)

    # Heavy box mode overrides baseline mode
    if mappo_heavybox_rewards:
        baseline_mappo_rewards = False

    if is_hetero:
        print(f"[MAPPO Training] Agent types: HETEROGENEOUS (agent0={agent0}, agent1={agent1})")
    else:
        print(f"[MAPPO Training] Agent types: HOMOGENEOUS ({agent0})")

    if mappo_heavybox_rewards:
        print(f"[MAPPO Training] Reward mode: HEAVY BOX (Exp 7 scales, distance penalty removed)")
    elif baseline_mappo_rewards:
        print(f"[MAPPO Training] Reward mode: BASELINE (TRUE ORIGINAL 7 MAPush rewards)")
    else:
        print(f"[MAPPO Training] Reward mode: EXTENDED (includes HAPPO-specific rewards)")

    # Velocity task parameters (from CLI)
    vel_speed_min = getattr(args, 'vel_speed_min', None)
    vel_speed_max = getattr(args, 'vel_speed_max', None)
    vel_tracking_scale = getattr(args, 'vel_tracking_scale', None)
    vel_angular_penalty_scale = getattr(args, 'vel_angular_penalty_scale', None)

    # For velocity task, disable incompatible reward flags
    if getattr(args, 'task', '') == 'go1push_vel':
        baseline_mappo_rewards = False
        mappo_heavybox_rewards = False
        print(f"[MAPPO Training] Task: VELOCITY (go1push_vel)")

    env, env_cfg = make_env(args, custom_cfg(args, agent0=agent0, agent1=agent1, baseline_mappo_rewards=baseline_mappo_rewards, mappo_heavybox_rewards=mappo_heavybox_rewards, vel_speed_min=vel_speed_min, vel_speed_max=vel_speed_max, vel_tracking_scale=vel_tracking_scale, vel_angular_penalty_scale=vel_angular_penalty_scale), single_agent, agent0=agent0, agent1=agent1)
    
    if args.algo == "ppo":
        # or use --config ./openrl_ws/cfgs/ppo.yaml in terminal
        args.lr = 0.0005
        args.critic_lr = 0.0005
        args.episode_length = 200

    if "po" in args.algo:
        from openrl.modules.common import PPONet
        from openrl.runners.common import PPOAgent
        # initilize net and agent
        net = PPONet(env, cfg=args, device=args.rl_device)
        agent = PPOAgent(net)
        # initilize logger
        logger = Logger(
            cfg=net.cfg,
            project_name="MQE",
            scenario_name=args.task,
            wandb_entity="chrisyrniu",
            exp_name=args.exp_name,
            log_path="./log",
            use_wandb=args.use_wandb,
            use_tensorboard=args.use_tensorboard,
        )
        run_dir = str(logger.run_dir)

        # add config to log
        if args.task in ("go1push_mid", "go1push_vel"):
            source_folder = "./task/"+args.exp_name+"/"
            target_folder = run_dir + "/task/"
            if os.path.exists(source_folder):
                shutil.copytree(source_folder, target_folder)

        # Save run configuration (command, hyperparameters, agent types, etc.)
        save_run_config(run_dir, args, env_cfg)

        if getattr(args, "checkpoint") is not None:
            if os.path.exists(args.checkpoint):
                agent.load(args.checkpoint)
                print("*******************************************************************************************************")
                print("loaded checkpoint: ", args.checkpoint)
                print("*******************************************************************************************************")
            else:
                print("*******************************************************************************************************")
                print("checkpoint not found: ", args.checkpoint)
                print("-------                  NOT LOADED!           --------------------------------------------------------")
                print("*******************************************************************************************************")
        else:
            print("*******************************************************************************************************")
            print("---------       no checkpoint provided       ----------------------------------------------------------")
            print("*******************************************************************************************************")

        # initilize callback
        callback=CheckpointCallback(save_freq=10000000, save_path= run_dir + "/checkpoints", name_prefix="rl_model", save_replay_buffer=False, verbose=2)
        agent.train(
            total_time_steps=args.train_timesteps,
            logger=logger,
            callback=callback,
        )
    else:
        agent.train(total_time_steps=args.train_timesteps)

    # move run_folder to result folder if training mid layer
    if args.task in ("go1push_mid", "go1push_vel"):
        source_folder = run_dir
        target_folder = "./results/"+start_time_str+"_"+args.exp_name+"/"
        shutil.copytree(source_folder, target_folder)

if __name__ == '__main__':
    args = get_args()
    print("start training...")
    train(args)
