import time
import torch

from purplerl.train_workbook import WorkbenchObsEncoder
from purplerl.workbook_env import WorkbookEnv
from purplerl.policy import StochasticPolicy, ContinuousPolicy
import purplerl.config as cfg

max_ep_len = 100

def run_policy(env, policy: StochasticPolicy):

    state = WorkbookEnv.SheetState(
        lesson = 0,
        template_idx = 0,
        flip = False,
        rot=3
    )

    obs = env.reset(sheet_state = state)
    ep_ret = 0
    ep_len = 0

    while True:
        env.render()
        time.sleep(1)

        a = policy.act(torch.as_tensor(obs, **cfg.tensor_args))
        obs, r, d, _ = env.step(a.cpu().numpy())
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print(f'EpRet {ep_ret} \t EpLen {ep_len}')
            obs, r, d, ep_ret, ep_len = env.reset(sheet_state = state), 0, False, 0, 0            

def run():
    pass


if __name__ == '__main__':
    
    #import argparse
    #parser = argparse.ArgumentParser()
    #parser.add_argument('fpath', type=str)
    #args = parser.parse_args()
    cfg.use_cpu()
    env = WorkbookEnv()
    policy= ContinuousPolicy(
        obs_encoder=WorkbenchObsEncoder(),
        hidden_sizes=[64, 64],
        action_space = env.action_space
    ).to(cfg.device)
    checkpoint = torch.load("results/Workbook/phase1/visionary-voice-195/checkpoint150.pt", map_location=cfg.device)
    policy.load_checkpoint(checkpoint["policy"])
    
    run_policy(env, policy)
