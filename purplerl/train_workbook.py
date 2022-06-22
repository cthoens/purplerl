import os
import gym
import math
from purplerl.sync_experience_buffer import ExperienceBufferBase

import torch
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Flatten
import wandb

import joblib

from purplerl.trainer import MonoObsExperienceBuffer, RewardToGo, Trainer
from purplerl.environment import GymEnvManager
from purplerl.workbook_env import WorkbookEnv
from purplerl.policy import ContinuousPolicy, Vanilla

import os.path as osp

def load_pytorch_policy(fpath, name):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, name)
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)    

    return model

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

        self.shape: tuple[int, ...] = (64, )

    def forward(self, obs: torch.tensor):
        old_shape = list(obs.shape)[:-3]
        x = self.cnn_layers(obs.reshape(-1, 1, 128, 128))
        
        return x.reshape(*(old_shape + [64]))


def run():
    config = {
        "policy_lr": 5e-5,
        "policy_lr_decay": 0.0 ,
        "vf_lr": 5e-5,
        "vf_lr_decay": 0.0 
    }
    with wandb.init(project="Workbook-p2", config=config) as run:
        run_training(run.name, **wandb.config)

def run_training(
    exp_name,
    policy_lr,
    policy_lr_decay,
    vf_lr,
    vf_lr_decay
):
    project_name = "Workbook"
    phase_name = "phase2"
    batch_size = 4
    buffer_size = 500
    epochs = 500
    
    gym.envs.register(
        id='workbook-v0',
        entry_point='purplerl.workbook_env:WorkbookEnv',
        max_episode_steps=100,
        kwargs={},
    )

    env_manager= GymEnvManager('workbook-v0', batch_size=batch_size)

    policy= ContinuousPolicy(
        obs_encoder=WorkbenchObsEncoder(),
        hidden_sizes=[64, 64],
        action_space = env_manager.action_space
    )
    wandb.watch(policy)

    def policy_lr_scheduler():
        return policy_lr *  math.exp(-policy_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])
    
    def value_net_lr_scheduler():
        return vf_lr *  math.exp(-vf_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])

    trainer_checkpoint_dict = {}
    checkpoint_path = f"{project_name}/"
    resuming = os.path.exists(os.path.join(checkpoint_path, "resume.pt"))
    if resuming:
        print("Resuming")
        policy = load_pytorch_policy(checkpoint_path, "resume.pt")
        trainer_checkpoint_dict = joblib.load(osp.join(checkpoint_path, "resume.pkl"))

    experience = MonoObsExperienceBuffer(
        batch_size, 
        buffer_size, 
        env_manager.observation_space.shape, 
        policy.action_shape
    )

    # policy_updater= RewardToGo(
    #          policy=policy,
    #          experience=experience,
    #          policy_lr=1e-3,
    # )

    policy_updater = Vanilla(
        policy = policy,
        experience = experience,
        hidden_sizes = [64, 64],
        policy_lr_scheduler = policy_lr_scheduler,
        value_net_lr_scheduler = value_net_lr_scheduler
    )
    if resuming:
        policy_updater.from_checkpoint(trainer_checkpoint_dict)
    wandb.watch(policy_updater.value_net)

    trainer = Trainer(
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = 100,
        output_dir= f"{project_name}/{phase_name}/{exp_name}",
        state_dict = trainer_checkpoint_dict
    )
    trainer.run_training()

if __name__ == '__main__':
    run()
