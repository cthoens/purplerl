import os
from xmlrpc.client import Boolean
import gym
import math

import torch
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Flatten
import wandb


from purplerl.sync_experience_buffer import ExperienceBufferBase
from purplerl.trainer import MonoObsExperienceBuffer, Trainer
from purplerl.environment import GymEnvManager
from purplerl.policy import ContinuousPolicy, Vanilla, PPO
from purplerl.config import device

import os.path as osp

class WorkbenchObsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Flatten(),
            Linear(4096, 64)
        )

        self.shape: tuple[int, ...] = (65, )

    def forward(self, obs: torch.tensor):
        # Note: obs can be of shape (num_envs, sheet_shape) or (num_envs, buffer_size, sheet_shape)
        buffer_dims = list(obs.shape)[:-1]
        sheet_obs = obs[...,:-1]        
        power_obs = obs[...,-1:]
        # flatten buffer dimensions since the cnn only accepts 3D or 4D input
        x = self.cnn_layers(sheet_obs.reshape(-1, 1, 128, 128))
        # restore the buffer dimension        
        x = x.reshape(*(buffer_dims + [64]))

        return torch.concat((x, power_obs), -1)


def run(dev_mode = False):
    phase_config = {
        "phase1": {
            "policy_lr": 1e-4,
            "vf_lr": 2.5e-4,
            "policy_epochs" : 10,
            "vf_epochs": 20,
            "policy_lr_decay": 0.0,
            "vf_lr_decay": 0.0,
            "discount": 0.99,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2
        },
        "phase2": {
            "policy_lr": 1e-4,
            "vf_lr": 2.5e-4,
            "policy_epochs" : 10,
            "vf_epochs": 20,
            "policy_lr_decay": 0.0,
            "vf_lr_decay": 0.0,
            "discount": 0.99,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2
        }
    }
    active_phase = "phase1"
    config = phase_config[active_phase]
    config["phase"] = active_phase
    wandb_mode = "online" if not dev_mode else "disabled"
    with wandb.init(project="Workbook", config=config, mode=wandb_mode) as run:
        project_name = run.project
        if project_name=="": project_name = "Dev"
        run_training(project_name, run.name, **wandb.config)

def run_training(
    project_name,
    exp_name,
    policy_lr,
    policy_epochs,
    policy_lr_decay,
    vf_lr,
    vf_epochs,
    vf_lr_decay,
    discount,
    adv_lambda,
    phase,
    clip_ratio: float = 0.2,
    target_kl: float = 0.01,
):  
    num_envs = 6
    buffer_size = 1550
    epochs = 500
    save_freq = 100
    
    gym.envs.register(
        id='workbook-v0',
        entry_point='purplerl.workbook_env:WorkbookEnv',
        kwargs={
        },
    )

    env_manager= GymEnvManager('workbook-v0', num_envs=num_envs)

    policy= ContinuousPolicy(
        obs_encoder=WorkbenchObsEncoder(),
        hidden_sizes=[64, 64],
        action_space = env_manager.action_space
    ).to(device)
    wandb.watch(policy)

    def policy_lr_scheduler():
        return policy_lr *  math.exp(-policy_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])
    
    def value_net_lr_scheduler():
        return vf_lr *  math.exp(-vf_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])

    experience = MonoObsExperienceBuffer(
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
        policy = policy,
        experience = experience,
        hidden_sizes = [64, 64],
        policy_lr_scheduler = policy_lr_scheduler,
        vf_lr_scheduler = value_net_lr_scheduler,
        policy_epochs = policy_epochs,
        vf_epochs = vf_epochs,
        lam = adv_lambda,
        clip_ratio = clip_ratio,
        target_kl = target_kl,
    )
    wandb.watch(policy_updater.value_net)

    trainer = Trainer(
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= f"results/{project_name}/{phase}/{exp_name}"
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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    run(args.dev)
