
from script.utils.utils import make_env, get_args, MATWrapper ,set_seed, get_args, parse_sim_params, load_cfg
# from openrl_ws.utils import make_env, get_args, MATWrapper
from script.utils.process_sarl import process_sarl
from script.utils.process_marl import process_MultiAgentRL, get_AgentIndex
from mqe.envs.utils import custom_cfg
import shutil
import os

from datetime import datetime

MARL_ALGOS = ["mappo", "happo", "hatrpo","maddpg","ippo"]
SARL_ALGOS = ["ppo","ddpg","sac","td3","trpo"]

def train(args):

    # cfg_parser = create_config_parser()
    # cfg = cfg_parser.parse_args()

    start_time = datetime.now()
    start_time_str = start_time.strftime("%m/%d/%Y-%H:%M:%S")

    assert args.algo in MARL_ALGOS + SARL_ALGOS, \
        "Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo, \
            maddpg,sac,td3,trpo,ppo,ddpg]"
    algo = args.algo
    if args.algo in MARL_ALGOS: 
        # maddpg exists a bug now 
        args.task_type = "MultiAgent"
        algo = "MultiAgentRL"
        single_agent = False
        env, env_cfg = make_env(args, custom_cfg(args), single_agent)
        runner = eval('process_{}'.format(algo))(args, env, cfg_train, args.model_dir)
        # if args.model_dir != "":
        #     runner.eval(1000)
        # else:
        #     runner.run()
        return
    elif args.algo in SARL_ALGOS:
        algo = "sarl"
        single_agent = True
    
    env, env_cfg = make_env(args, custom_cfg(args), single_agent)
    # runner = eval('process_{}'.format(algo))(args, env, cfg_train, logdir)
    # iterations = cfg_train["learn"]["max_iterations"]
    if args.max_iterations > 0:
        iterations = args.max_iterations

    # runner.train(train_epoch=iterations)


if __name__ == '__main__':
    args = get_args()
    # cfg_train, logdir = load_cfg(args)
    # sim_params = parse_sim_params(args, cfg, cfg_train)
    # set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train(args)