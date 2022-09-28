import time
import torch

from purplerl.train_robotarm import RobotArmObsEncoder, RobotArmEnvManager
from purplerl.policy import StochasticPolicy, ContinuousPolicy
from purplerl.config import CpuConfig

cfg = CpuConfig()

max_ep_len = 100

def run_policy(env, policy: StochasticPolicy):

    obs = env.reset()
    ep_ret = 0
    ep_len = 0

    while True:
        #env.render()
        time.sleep(0.1)

        a = policy.act(torch.as_tensor(obs, **cfg.tensor_args))
        obs, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d[0] or (ep_len == max_ep_len):
            print(f'EpRet {ep_ret[0]} \t EpLen {ep_len}')
            ep_ret, ep_len = 0, 0
            #obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

def run():
    pass


if __name__ == '__main__':

    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('fpath', type=str)
    #args = parser.parse_args()

    file_name = "/home/cthoens/code/UnityRL/ml-agents-robots/Builds/RobotArm.x86_64"
    env_manager = RobotArmEnvManager(file_name=file_name, action_scaling=2.0, port_=6065)
    env_manager.set_lesson(1)
    policy= ContinuousPolicy(
        obs_encoder = RobotArmObsEncoder(env_manager),
        action_space = env_manager.action_space,
        hidden_sizes = [64],
        std_scale = 0.5,
        min_std= torch.as_tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
        max_std= torch.as_tensor([0.6, 0.6, 0.6, 0.6, 0.6])
    ).to(cfg.device)
    checkpoint = torch.load("results/RobotArm/phase1/floral-jazz-202/checkpoint300.pt", map_location=cfg.device)
    policy.load_checkpoint(checkpoint["policy"])

    run_policy(env_manager, policy)
