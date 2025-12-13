<!-- # Learning Multi-Agent Collaborative Manipulation for Long-Horizon Quadrupedal Pushing -->
<h2 align="center">
Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing
</h2>

<p align="center">
    <a href="https://collaborative-mapush.github.io/">Website</a> |
    <a href="https://arxiv.org/pdf/2411.07104">Paper</a>
</p>
<p align="center">
  <img src="resources/images/teaser1.gif" alt="Descriptive Alt Text" width="600"/>
</p>
<p align="center">
  <img src="resources/images/teaser2.gif" alt="Descriptive Alt Text" width="600"/>
</p>

## Overview ##
This codebase includes the implementation of our hierarchical MARL framework for multi-quadruped pushing tasks in Issac Gym. The codebase is developed based on [MQE](https://github.com/ziyanx02/multiagent-quadruped-environment).


## Installation ##
1. Create a new Python virtual env or conda environment with Python 3.8.
    ```
    conda create -n mapush python=3.8
    ```
2. Install PyTorch and Isaac Gym.
    - Install an compatible PyTorch version from https://pytorch.org/.
    - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym.
        ```
        tar -xf IsaacGym_Preview_4_Package.tar.gz
        cd isaacgym/python && pip install -e .
        ```
3. Check Isaac Gym is available by running
    - `cd examples && python 1080_balls_of_solitude.py`
4. Install MAPush and required packages. Direct to the directory of this repository and run
    - `pip install -e .`
5. Refer to [Trouble Shooting](#trouble-shooting) if you meet issues!

<!-- ## Code Structure ##

Environment for each task is defined by:
- a class for controlling objects involved in the task. `./mqe/envs/go1/go1.py` is a base class for Unitree Go1 robot with locomotion policy implemented in [walk-these-ways](https://github.com/Improbable-AI/walk-these-ways). `./mqe/envs/npc/` includes several classes created for different interactive objects.
- a wrapper to specify observations, actions, rewards, and infos. `./mqe/envs/wrappers/` includes several wrappers for reference.
- a config file to specify all the configuration about the environment, including configs for simulation, terrain registration, robot assets, etc. Config files use inheritance. `./mqe/envs/configs/` includes the config files of pre-defined tasks. To explore more available configurations, please check config files in `./mqe/envs/base/`, `./mqe/envs/field/` and `./mqe/envs/go1/`, there should be no intersections between these config files for clearance.

Blocks used in terrain registration is defined in `./mqe/utils/terrain/barrier_track.py`. -->

## Usage ##

### 1. Mid-Level Controller ###

#### 1.1 Training ####

- **1.1.1 Command Line**
  - Command: `source task/<object>/train.sh False`
  - `<object>` options: `cuboid`, `Tblock`, or `cylinder`
  - Example: `source task/cuboid/train.sh False`

- **1.1.2 Running Process**
  - `task/<object>/train.sh` updates `mqe/envs/configs/go1_push_config.py` based on `task/<object>/config.py`. It then starts training by running `./openrl_ws/train.py`. Training logs are temporarily stored in `./log/`. Once training completes, the final output (including model checkpoints, TensorBoard data, and task settings) is saved in `./results/<mm-dd-hh_object>`.
  - Afterward, `task/<object>/train.sh` calls `./openrl_ws/test.py` with `<test_mode calculator>` to compute the success rate for each checkpoint. The results are stored in `./results/<mm-dd-hh_object>/success_rate.txt`.

#### 1.2 Testing ####

- **1.2.1 Command Line**
  - Command: `source <save_dir>/task/train.sh True`
  - `<save_dir>` is the directory where the training results are saved.
  - Example: `source results/10-15-23_cuboid/task/train.sh True`

- **1.2.2 Running Process**
  - Choose a specific checkpoint and modify `$filename`, or add the `--record_video` flag to record the output.
  - `<save_dir>/task/train.sh` will invoke `./openrl_ws/test.py` with `<test_mode viewer>` to render a visual output of the pushing behavior.

#### 1.3 Config Revision ####

- **1.3.1 Task Setting**
  - The task configuration is saved in `./task/cuboid/config.py`. You can edit this file according to the detailed annotations provided.

- **1.3.2 Hyperparameters**
  - Mid-level network hyperparameters can be modified in `./task/<object>/train.sh`. Adjust values for `$num_envs`, `$num_steps`, `#checkpoint`, or pass additional parameters directly to `./openrl_ws/train.py`.

### 2. High-Level Controller ###

#### 2.1 Training ####

- **2.1.1 Preparation**
  - Ensure that a mid-level controller has been trained, and add the checkpoint path to `mqe/envs/configs/go1_push_upper_config.py` under `control.command_network_path`.

- **2.1.2 Command**
  - Run the following command to begin training:
    ```bash
    python ./openrl_ws/train.py --algo ppo --task go1push_upper --train_timesteps 100000000 --num_envs 500 --use_tensorboard --headless
    ```
#### 2.2 Testing ####

- **2.2.1 Command**
  - Run the following command to test the high-level controller:
    ```bash
    python ./openrl_ws/test.py --algo ppo --task go1push_upper --train_timesteps 100000000 --num_envs 10 --use_tensorboard --checkpoint your_checkpoint
    ```
  - Use `--record_video` to record the results.

- **2.2.2 Pretrained Example**
  - A pretrained high-level policy to push a 1.2m x 1.2m cube, can be found in `resources/goals_net`.

- **2.2.3 Code Location**
  - **Upper-Level Task Configuration**
    - Task configuration settings for the high-level controller are in `mqe/envs/configs/go1_push_upper_config.py`.
  - **Upper-Level Task Wrapper**
    - The wrapper for upper-level task settings is located in `mqe/envs/wrappers/go1_push_upper_wrapper.py`.



<!-- ## Different Objects ##
You could find different object at resources/objects and add it to stage at config. -->

## Trouble Shooting ##

1. If you get the following error: `ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory`, it is also possible that you need to do `export LD_LIBRARY_PATH=/PATH/TO/LIBPYTHON/DIRECTORY` / `export LD_LIBRARY_PATH=/PATH/TO/CONDA/envs/YOUR_ENV_NAME/lib`. You can also try: `sudo apt install libpython3.8`.

2. The numpy version should be no later than 1.19.5 to avoid conflict with the Isaac Gym utility files. You can also modify 'np.float' into 'np.float32' in the function 'get_axis_params' of the python file in 'isaacgym/python/isaacgym/torch_utils.py' to resolve the issue. 

3. If you get `Segmentation fault (core dumped)` while rendering frames using A100/A800, please switch to GeFoece graphic cards.

4. If you get the error `partially initialized module 'openrl.utils.callbacks.callbacks' has no attribute 'BaseCallback'`, please comment out the line `from openrl.runners.common.base_agent import BaseAgent` in `openrl/utils/callback/callback.py`.

## Citation ##

If you find our paper or repo helpful to your research, please consider citing the paper:
```
@article{mapush2024,
  title={Learning Multi-Agent Loco-Manipulation for Long-Horizon Quadrupedal Pushing},
  author={Feng, Yuming and Hong, Chuye and Niu, Yaru and Liu, Shiqi and Yang, Yuxiang and Yu, Wenhao and Zhang, Tingnan and Tan, Jie and Zhao, Ding},
  journal={arXiv preprint arXiv:2411.07104},
  year={2024}
}
```
