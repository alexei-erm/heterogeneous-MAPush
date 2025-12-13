
from __future__ import print_function, division, absolute_import

from typing import Any, Dict, Optional, Union
import isaacgym
import os
import sys
import yaml
import random
import numpy as np
import torch
import gym
from gym import spaces

from mqe.envs.utils import make_mqe_env

from openrl.configs.config import create_config_parser
from isaacgym import gymutil
from typing import List
from openrl.configs.utils import ProcessYamlAction

from abc import ABC, abstractmethod
import math
import numpy as np
import argparse
from bisect import bisect

from isaacgym import gymapi
from isaacgym.gymutil import parse_device_str

from mqe.envs.go1.go1_config import Go1Cfg
from openrl.envs.vec_env import BaseVecEnv

def make_env(args, custom_cfg=None, single_agent=False):
    env, env_cfg = make_mqe_env(args.task, args, custom_cfg=custom_cfg)

    if single_agent:
        env = SingleAgentWrapper(env)

    return mqe_openrl_wrapper(env), env_cfg

class mqe_openrl_wrapper(gym.Wrapper):
    def __init__(self, env):
        self.env = env
        self.agent_num = self.env.num_agents
        self.parallel_env_num = self.env.num_envs
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        """Reset all environments."""
        obs = self.env.reset()
        return obs.cpu().numpy()

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        actions = torch.from_numpy(0.5 * actions).cuda().clip(-1, 1)

        obs, reward, termination, info = self.env.step(actions)

        obs = obs.cpu().numpy()
        rewards = reward.cpu().unsqueeze(-1).numpy()
        dones = termination.cpu().unsqueeze(-1).repeat(1, self.agent_num).numpy().astype(bool)

        infos = []
        for i in range(dones.shape[0]):
            infos.append({})

        return obs, rewards, dones, infos

    def close(self, **kwargs):
        return self.env.close()

    @property
    def use_monitor(self):
        return False

    def batch_rewards(self, buffer):

        step_count = self.env.reward_buffer["step count"]
        reward_dict = {"average step reward": 0}
        for k in self.env.reward_buffer.keys():
            if k == "step count":
                continue
            reward_dict[k] = self.env.reward_buffer[k] / (self.num_envs * step_count)
            if hasattr(self.env, "single_agent_reward_scale"):
                reward_dict[k] *= self.env.single_agent_reward_scale
            if "reward" in k or "punishment" in k:
                reward_dict["average step reward"] += reward_dict[k]
            self.env.reward_buffer[k] = 0
        self.env.reward_buffer["step count"] = 0
        return reward_dict

class MATWrapper(gym.Wrapper):
    @property
    def observation_space(
        self,
    ):
        """Return the :attr:`Env` :attr:`observation_space` unless overwritten then the wrapper :attr:`observation_space` is used."""
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation_space = observation_space["policy"]
        return observation_space

    def observation(self, observation):
        if self._observation_space is None:
            observation_space = self.env.observation_space
        else:
            observation_space = self._observation_space

        if (
            "critic" in observation_space.spaces.keys()
            and "policy" in observation_space.spaces.keys()
        ):
            observation = observation["policy"]
        return observation
    
    def reset(self, **kwargs):
        """Reset all environments."""
        return self.env.reset(**kwargs)

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        return self.env.step(actions, extra_data)

class SingleAgentWrapper(gym.Wrapper):
    
    def __init__(self, env):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        super().__init__(env)

        self.num_envs = self.env.num_envs * self.env.num_agents
        self.num_agents = 1

        self.single_agent_reward_scale = self.env.num_agents

    def reset(self, **kwargs):
        """Reset all environments."""
        obs = self.env.reset(**kwargs)
        return obs.reshape(self.num_envs, 1, -1)

    def step(self, actions, extra_data: Optional[Dict[str, Any]] = None):
        """Step all environments."""
        actions = actions.reshape(self.env.num_envs, self.env.num_agents, -1)
        obs, reward, termination, info = self.env.step(actions)
        return obs.reshape(self.num_envs, 1, -1), reward.reshape(self.num_envs, 1), torch.stack([termination, termination], dim=1).reshape(self.num_envs), info

def set_seed(seed, torch_deterministic=False):
    if seed == -1 and torch_deterministic:
        seed = 42
    elif seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch_deterministic:
        # refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_deterministic(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    return seed


def retrieve_cfg(args):
    return os.path.join(args.logdir, "{}/{}/{}".format(args.task, args.algo, args.algo)), "cfg/{}/config.yaml".format(args.algo)


def load_cfg(args, use_rlg_config=False):
    with open(os.path.join(os.getcwd(), args.cfg_train), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)


    logdir = args.logdir
    if use_rlg_config:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["params"]["torch_deterministic"] = True

        exp_name = cfg_train["params"]["config"]['name']

        if args.experiment != 'Base':
            if args.metadata:
                exp_name = "{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])

            else:
                exp_name = args.experiment

        # Override config name
        cfg_train["params"]["config"]['name'] = exp_name

        if args.resume > 0:
            cfg_train["params"]["load_checkpoint"] = True

        if args.checkpoint != "Base":
            cfg_train["params"]["load_path"] = args.checkpoint

        # Set maximum number of training iterations (epochs)
        if args.max_iterations > 0:
            cfg_train["params"]["config"]['max_epochs'] = args.max_iterations

        seed = cfg_train["params"].get("seed", -1)
        if args.seed is not None:
            seed = args.seed
        cfg_train["params"]["seed"] = seed

    else:

        # Set deterministic mode
        if args.torch_deterministic:
            cfg_train["torch_deterministic"] = True

        # Override seed if passed on the command line
        if args.seed is not None:
            cfg_train["seed"] = args.seed

        log_id = args.logdir
        if args.experiment != 'Base':
            if args.metadata:
                log_id = args.logdir + "_{}_{}_{}_{}".format(args.experiment, args.task_type, args.device, str(args.physics_engine).split("_")[-1])
            else:
                log_id = args.logdir + "_{}".format(args.experiment)

        logdir = os.path.realpath(log_id)
        # os.makedirs(logdir, exist_ok=True)

    return cfg_train, logdir


def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params


def get_args(benchmark=False, use_rlg_config=False, task_name="", algo=""):
    custom_parameters = [
        {"name": "--test", "action": "store_true", "default": False,
            "help": "Run trained policy, no training"},
        {"name": "--play", "action": "store_true", "default": False,
            "help": "Run trained policy, the same as test, can be used only by rl_games RL library"},
        {"name": "--resume", "type": int, "default": 0,
            "help": "Resume training or start testing from a checkpoint"},
        {"name": "--checkpoint", "type": str, "default": "Base",
            "help": "Path to the saved weights, only for rl_games RL library"},
        {"name": "--headless", "action": "store_true", "default": False,
            "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False,
            "help": "Use horovod for multi-gpu training, have effect only with rl_games RL library"},
        {"name": "--task", "type": str, "default": "Humanoid",
            "help": "Can be BallBalance, Cartpole, CartpoleYUp, Ant, Humanoid, Anymal, FrankaCabinet, Quadcopter, ShadowHand, Ingenuity"},
        {"name": "--task_type", "type": str,
            "default": "Python", "help": "Choose Python or C++"},
        {"name": "--rl_device", "type": str, "default": "cuda:0",
            "help": "Choose CPU or GPU device for inferencing policy network"},
        {"name": "--logdir", "type": str, "default": "logs/"},
        {"name": "--experiment", "type": str, "default": "Base",
            "help": "Experiment name. If used with --metadata flag an additional information about physics engine, sim device, pipeline and domain randomization will be added to the name"},
        {"name": "--metadata", "action": "store_true", "default": False,
            "help": "Requires --experiment flag, adds physics engine, sim device, pipeline info and if domain randomization is used to the experiment name provided by user"},
        {"name": "--cfg_train", "type": str,
            "default": "Base"},
        {"name": "--cfg_env", "type": str, "default": "Base"},
        {"name": "--num_envs", "type": int, "default": 0,
            "help": "Number of environments to create - override config file"},
        {"name": "--episode_length", "type": int, "default": 0,
            "help": "Episode length, by default is read from yaml config"},
        {"name": "--seed", "type": int, "default": 0, "help": "Random seed"},
        {"name": "--max_iterations", "type": int, "default": -1,
            "help": "Set a maximum number of training iterations"},
        {"name": "--steps_num", "type": int, "default": -1,
            "help": "Set number of simulation steps per 1 PPO iteration. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--minibatch_size", "type": int, "default": -1,
            "help": "Set batch size for PPO optimization step. Supported only by rl_games. If not -1 overrides the config settings."},
        {"name": "--randomize", "action": "store_true", "default": False,
            "help": "Apply physics domain randomization"},
        {"name": "--torch_deterministic", "action": "store_true", "default": False,
            "help": "Apply additional PyTorch settings for more deterministic behaviour"},
        {"name": "--algo", "type": str, "default": "happo",
            "help": "Choose an algorithm"},
        {"name": "--model_dir", "type": str, "default": "",
            "help": "Choose a model dir"},
        {"name": "--datatype", "type": str, "default": "random",
            "help": "Choose an offline datatype"},
        {"name": "--record_video", "action": "store_true", "default": False},]

    if benchmark:
        custom_parameters += [{"name": "--num_proc", "type": int, "default": 1, "help": "Number of child processes to launch"},
                              {"name": "--random_actions", "action": "store_true",
                                  "help": "Run benchmark with random actions instead of inferencing"},
                              {"name": "--bench_len", "type": int, "default": 10,
                                  "help": "Number of timing reports"},
                              {"name": "--bench_file", "action": "store", "help": "Filename to store benchmark results"}]

    # parse arguments
    args = gymutil.parse_arguments(
        description="RL Policy",
        custom_parameters=custom_parameters)
    
    if task_name != "":
        args.task = task_name
    if algo != "":
        args.algo = algo

    # allignment with examples
    args.device_id = args.compute_device_id
    args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'

    if args.test:
        args.play = args.test
        args.train = False
    elif args.play:
        args.train = False
    else:
        args.train = True

    logdir, cfg_train = retrieve_cfg(args)

    if use_rlg_config == False:
        if args.horovod:
            print("Distributed multi-gpu training with Horovod is not supported by rl-pytorch. Use rl_games for distributed training.")
        if args.steps_num != -1:
            print("Setting number of simulation steps per iteration from command line is not supported by rl-pytorch.")
        if args.minibatch_size != -1:
            print("Setting minibatch size from command line is not supported by rl-pytorch.")
        if args.checkpoint != "Base":
            raise ValueError("--checkpoint is not supported by rl-pytorch. Please use --resume <iteration number>")

    # use custom parameters if provided by user
    if args.logdir == "logs/":
        args.logdir = logdir

    if args.cfg_train == "Base":
        args.cfg_train = cfg_train

    return args
