import os

import random
from purplerl.async_env_manger import ProcessEnvManager
import torch
import numpy as np

import torch
import torch.nn as nn
import wandb

from purplerl.config import CpuConfig
from purplerl.sync_experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import GymEnvManager
from purplerl.policy import ContinuousPolicy, PPO
from purplerl.tune_env import TuneEnv, TuneEnvObsEncoder

cfg = CpuConfig()

def run(dev_mode = False):
    phase_config = {
        "phase1": {
            "policy_lr": 2e-5,
            "vf_lr": 1e-4,
            "update_epochs" : 10,
            "discount": 0.95,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2
        }
    }
    active_phase = "phase1"
    config = phase_config[active_phase]
    config["phase"] = active_phase
    wandb_mode = "online" if not dev_mode else "disabled"
    with wandb.init(project="TuneWorkbook", config=config, mode=wandb_mode) as run:
        project_name = run.project
        if project_name=="": project_name = "TuneDev"
        run_training(project_name, run.name, **wandb.config)

def run_training(
    project_name,
    exp_name,
    policy_lr,
    vf_lr,
    update_epochs,
    discount,
    adv_lambda,
    phase,
    clip_ratio: float = 0.2,
    target_kl: float = 0.01,
):
    num_envs = 5
    update_batch_size = 6
    buffer_size = 6

    epochs = 200
    save_freq = 10

    env_manager= ProcessEnvManager(TuneEnv, num_envs=num_envs)

    policy= ContinuousPolicy(
        obs_encoder = TuneEnvObsEncoder(),
        action_space = env_manager.action_space,
        hidden_sizes = [128, 128],
        output_activation=nn.Sigmoid,
        mean_offset = torch.as_tensor([0.0, 0.0, 0.0]),
        std_scale = 0.2,
        min_std= torch.as_tensor([0.05, 0.05, 0.05])
    )

    experience = ExperienceBuffer(
        num_envs,
        buffer_size,
        env_manager.observation_space.shape,
        policy.action_shape,
        discount,
        tensor_args = {
            "dtype": torch.float32,
            "device": torch.device('cpu')
        }
    )

    policy_updater = PPO(
        cfg = cfg,
        policy = policy,
        experience = experience,
        hidden_sizes = [128, 128],
        policy_lr = policy_lr,
        vf_lr = vf_lr,
        update_epochs = update_epochs,
        update_batch_size=update_batch_size,
        lam = adv_lambda,
        clip_ratio = clip_ratio,
        target_kl = target_kl
    )
    wandb.watch((policy, policy_updater.value_net_tail), log='all', log_freq=20)

    out_dir = f"results/{project_name}/{phase}/{exp_name}"
    trainer = Trainer(
        cfg,
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= out_dir,
    )

    checkpoint_path = os.path.join(f"results/{project_name}", f"{phase}-resume.pt")
    if os.path.exists(checkpoint_path):
        if os.path.islink(checkpoint_path):
            print(f"Resuming from {checkpoint_path}[{os.readlink(checkpoint_path)}]")
        else:
            print(f"Resuming from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    else:
        print("****\n**** Starting from scratch !!!\n****")

    trainer.run_training()
    env_manager.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    import multiprocessing as mp
    mp.set_start_method('spawn')

    seed = 45632
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    import matplotlib
    matplotlib.use('Agg')

    import purplerl.workbook_env as env
    env.init(sheet_path="sheets-64")

    run(args.dev)
