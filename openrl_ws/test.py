import isaacgym
from openrl_ws.utils import make_env, get_args, MATWrapper
from mqe.envs.utils import custom_cfg

from openrl.envs.common import make
from openrl.modules.common import PPONet
from openrl.runners.common import PPOAgent
import os
from pathlib import Path

import imageio
import os
from pathlib import Path
import numpy as np
import torch

import cv2


def save_video(frames, fps, filename="output.mp4"): 
    import cv2
    GIF_SAVE_DIR = Path(__file__).parent.parent / "docs/video"
    output_video_path = f"{GIF_SAVE_DIR}/{filename}"

    codec = cv2.VideoWriter_fourcc(*'mp4v')

    frame_shape = (frames.shape[2], frames.shape[3])

    frames = frames[:, :3, :, :]
    
    frames = np.transpose(frames, (0, 2, 3, 1))

    out = cv2.VideoWriter(output_video_path, codec, fps, frame_shape)

    for i in range(len(frames)):
        frame = frames[i]
        frame = frame.astype(np.uint8)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        out.write(frame)

    out.release()

    print("video has been created!")

# def save_video(frames, fps):
#     # Assuming your ndarray is named 'frames'
#     # frames.shape = (134, 4, 240, 360)

#     # Define the output video file name
#     output_video_path = 'output_video.mp4'
    
#     import cv2
#     # Define the video codec and frame rate
#     codec = cv2.VideoWriter_fourcc(*'mp4v')

#     # Get the shape of a single frame
#     frame_shape = frames.shape[2], frames.shape[3]

#     frames = frames[:, :3, :, :]
#     frames = np.transpose(frames, (0, 2, 3, 1))

#     # Create a VideoWriter object
#     out = cv2.VideoWriter(output_video_path, codec, fps, frame_shape)

#     # Iterate through each frame
#     for i in range(len(frames)):
#         # Convert frame to uint8 (assuming it's in range 0-255)
#         frame = frames[i]
#         frame = frame.astype(np.uint8)
        
#         # Transpose frame from (4, 240, 360) to (240, 360, 3) if needed
#         # frame = np.transpose(frame, (1, 2, 0))
        
#         # Write the frame to the video file
#         out.write(frame)

#     # Release the VideoWriter object
#     out.release()

#     print("Video created successfully.")

def save_gif(frames, fps, filename="output.gif"):

    # Assuming your ndarray is named 'frames'
    # frames.shape = (134, 4, 240, 360)

    # Define the output GIF file name
    GIF_SAVE_DIR = Path(__file__).parent.parent / "docs/gif"
    output_gif_path = f"{GIF_SAVE_DIR}/{filename}"

    # Convert the frames to uint8 (assuming it's in range 0-1)
    frames = np.transpose(frames, (0, 2, 3, 1))
    frames_uint8 = frames.astype(np.uint8)

    frames = [frames_uint8[i] for i in range(len(frames_uint8))]

    # Save frames as GIF
    imageio.mimsave(output_gif_path, frames, fps=fps, quality=10)

    absolute_path = os.path.abspath(output_gif_path)
    print(f"GIF created successfully at {absolute_path}")

args = get_args()
env, _ = make_env(args, custom_cfg(args))
net = PPONet(env, device="cuda")  # Create neural network.
agent = PPOAgent(net)  # Initialize the agent.

if args.algo == "jrpo" or args.algo == "ppo":
    from openrl.modules.common import PPONet
    from openrl.runners.common import PPOAgent
    net = PPONet(env, cfg=args, device=args.rl_device)
    agent = PPOAgent(net)
else:
    from openrl.modules.common import MATNet
    from openrl.runners.common import MATAgent
    env = MATWrapper(env)
    net = MATNet(env, cfg=args, device=args.rl_device)
    agent = MATAgent(net, use_wandb=args.use_wandb)

if getattr(args, "checkpoint") is not None:
    agent.load(args.checkpoint)
    print("---------------------------------------------------------------------------")
    print("Loaded checkpoint from: ", args.checkpoint)
    
test_mode = "calculator"
if getattr(args, "test_mode") is not None:
    test_mode = args.test_mode

# env.start_recording()
agent.set_env(env)  # The agent requires an interactive environment.
obs = env.reset()  # Initialize the environment to obtain initial observations and environmental information.

if test_mode == "calculator":
    while not torch.all(env.init_reset_buf):
        action, _ = agent.act(obs) 
        obs, r, done, info = env.step(action)
    success_rate=torch.mean(env.init_finished_buf.to(torch.float))
    finished_time = torch.mean(env.init_episode_length_buf * env.dt)
    collision_degree = torch.mean(env.collision_degree_buf[env.init_finished_buf]/ env.init_episode_length_buf[env.init_finished_buf])
    collaboration_degree = torch.mean(env.collaboration_degree_buf[env.init_finished_buf]/ env.init_episode_length_buf[env.init_finished_buf])
    print("success rate:",(success_rate.item()))
    print("finished time:",(finished_time.item()))
    print("collision degree:",(collision_degree.item()))
    print("collaboration degree:",(collaboration_degree.item()))
    success_rate=torch.mean(env.init_finished_buf.to(torch.float))
    finished_time = torch.mean(env.init_episode_length_buf * env.dt)
    collision_degree = torch.mean(env.collision_degree_buf[env.init_finished_buf]/ env.init_episode_length_buf[env.init_finished_buf])
    collaboration_degree = torch.mean(env.collaboration_degree_buf[env.init_finished_buf]/ env.init_episode_length_buf[env.init_finished_buf])
    print("success rate:",(success_rate.item()))
    print("finished time:",(finished_time.item()))
    print("collision degree:",(collision_degree.item()))
    print("collaboration degree:",(collaboration_degree.item()))
    print("-----------------------------------------------------")
elif test_mode=="viewer":
    if args.record_video:
        running_count = 0
        env.start_recording()
        while True:
            action, _ = agent.act(obs) 
            obs, r, done, info = env.step(action)
            if done.all():
                running_count += 1
                if torch.all(env.finished_buf):
                    print("success")
                else:
                    print("fail")
            if running_count == 1:
                frames = env.get_complete_frames()
                video_array = np.concatenate([np.expand_dims(frame, axis=0) for frame in frames ], axis=0).swapaxes(1, 3).swapaxes(2, 3)
                print(video_array.shape)
                print(np.mean(video_array))
                # save_gif(video_array, 1 / env.dt, filename="test.gif")
                save_video(video_array, 1 / env.dt, filename="test.mp4")
                break
    else:
        while True:
            action, _ = agent.act(obs) 
            obs, r, done, info = env.step(action)
            if done.all():
                if torch.all(env.finished_buf):
                    print("success")
                else:
                    print("fail")