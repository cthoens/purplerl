import torch

from purplerl.train_workbook import WorkbenchObsEncoder
from purplerl.workbook_env import WorkbookEnv
from purplerl.policy import StochasticPolicy, ContinuousPolicy
from purplerl.config import device, tensor_args

max_ep_len = 100

def run_policy(env, policy: StochasticPolicy):

    obs = env.reset()
    ep_ret = 0
    ep_len = 0

    while True:
        env.render()
        #time.sleep(1e1)

        a = policy.act(torch.as_tensor(obs, **tensor_args))
        obs, r, d, _ = env.step(a.cpu().numpy())
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print(f'EpRet {ep_ret} \t EpLen {ep_len}')
            obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0            

def run():
    pass


if __name__ == '__main__':
    
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('fpath', type=str)
    #args = parser.parse_args()
    
    env = WorkbookEnv()
    policy= ContinuousPolicy(
        obs_encoder=WorkbenchObsEncoder(),
        hidden_sizes=[64, 64],
        action_space = env.action_space
    ).to(device)
    checkpoint = torch.load("/home/cthoens/code/UnityRL/purplerl/results/Workbook/phase2/leafy-dew-76/checkpoint1000.pt")
    policy.load_checkpoint(checkpoint["policy"])
    
    run_policy(env, policy)
