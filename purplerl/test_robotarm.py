import time

import torch
import numpy as np
import av

from purplerl.train_robotarm import RobotArmObsEncoder, RobotArmEnvManager
from purplerl.policy import StochasticPolicy, ContinuousPolicy
from purplerl.config import CpuConfig

cfg = CpuConfig()

max_ep_len = 100

fps = 1
container = av.open("test.mp4", mode="w")

stream = container.add_stream("mpeg4", rate=fps)
stream.width = 272
stream.height = 84
stream.pix_fmt = "yuv420p"


def run_policy(env, policy: StochasticPolicy):

    obs = env.reset()
    ep_ret = 0
    ep_len = 0

    for _ in range(40):
        #env.render()
        img = np.zeros((84, 272, 3))
        img[:,:136,0] = np.array(env._training_obs(obs)[0]).squeeze()
        img[:,136:,0] = np.array(env._goal_obs(obs)[0]).squeeze()
        img[:,:,1] = img[:,:,0]
        img[:,:,2] = img[:,:,0]
        img = np.round(255 * (img * 0.5 + 0.5)).astype(np.uint8)
        img = np.clip(img, 0, 255)

        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

        a = policy.act(torch.as_tensor(obs, **cfg.tensor_args))
        obs, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d[0] or (ep_len == max_ep_len):
            print(f'EpRet {ep_ret[0]} \t EpLen {ep_len}')
            ep_ret, ep_len = 0, 0
            #obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Flush stream
    for packet in stream.encode():
        print(".", end="")
        container.mux(packet)

    # Close the file
    container.close()

    print("done")


def run():
    pass


if __name__ == '__main__':

    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('fpath', type=str)
    #args = parser.parse_args()

    file_name = "/home/cthoens/code/UnityRL/ml-agents-robots/Builds/RobotArm.x86_64"
    env_manager = RobotArmEnvManager(file_name=file_name, action_scaling=2.0, force_vulkan=False, port_=6065)
    env_manager.set_lesson(0)
    obs_encoder = RobotArmObsEncoder(env_manager)

    action_dist_net_output_shape = np.array(env_manager.action_space.shape + (2, ))
    class ActionDistNetTail(torch.nn.Module):
        def forward(self, enc_obs):
            assert(enc_obs.shape[1]==11)
            return enc_obs[:,:10].reshape((-1, np.prod(action_dist_net_output_shape)))
    action_dist_net_tail = ActionDistNetTail()
    policy= ContinuousPolicy(
        obs_encoder = obs_encoder,
        action_space = env_manager.action_space,
        action_dist_net_tail = action_dist_net_tail,
        std_scale = 0.5,
        min_std= torch.as_tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
        max_std= torch.as_tensor([0.6, 0.6, 0.6, 0.6, 0.6])
    ).to(cfg.device)

    checkpoint = torch.load("results/RobotArm/phase1/vivid-waterfall-408/checkpoint500.pt", map_location=cfg.device)
    policy.load_checkpoint(checkpoint["policy"])

    run_policy(env_manager, policy)
    env_manager.close()
