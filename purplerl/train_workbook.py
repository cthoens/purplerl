import os
import gym
import math

import torch
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Flatten
import wandb


from purplerl.sync_experience_buffer import ExperienceBufferBase
from purplerl.trainer import MonoObsExperienceBuffer, Trainer
from purplerl.environment import GymEnvManager
from purplerl.policy import ContinuousPolicy, Vanilla
from purplerl.config import device

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
    phase_config = {
        "phase1": {
            "policy_lr": 2.5e-4,
            "policy_epochs" : 3,
            "policy_lr_decay": 0.0,
            "vf_lr": 2.5e-4,
            "vf_epochs": 10,
            "vf_lr_decay": 0.0,
            "discount": 0.99,
            "adv_lambda": 0.95
        },
        "phase2": {
            "policy_lr": 5e-5,
            "policy_epochs" : 3,
            "policy_lr_decay": 0.0,            
            "vf_lr": 5e-5,
            "vf_epochs": 10,
            "vf_lr_decay": 0.0,
            "discount": 0.99,
            "adv_lambda": 0.95
        }
    }
    active_phase = "phase2"
    config = phase_config[active_phase]
    config["phase"] = active_phase
    #, mode="disabled"
    with wandb.init(project="Workbook", config=config) as run:
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
    phase
):  
    batch_size = 4
    buffer_size = 500
    epochs = 500
    save_freq = 100
    
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
    ).to(device)
    wandb.watch(policy)

    def policy_lr_scheduler():
        return policy_lr *  math.exp(-policy_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])
    
    def value_net_lr_scheduler():
        return vf_lr *  math.exp(-vf_lr_decay * experience.stats[ExperienceBufferBase.SUCCESS_RATE])

    experience = MonoObsExperienceBuffer(
        batch_size, 
        buffer_size, 
        env_manager.observation_space.shape, 
        policy.action_shape,
        discount
    )

    policy_updater = Vanilla(
        policy = policy,
        experience = experience,
        hidden_sizes = [64, 64],
        policy_lr_scheduler = policy_lr_scheduler,
        vf_lr_scheduler = value_net_lr_scheduler,
        policy_epochs = policy_epochs,
        vf_epochs = vf_epochs,
        lam = adv_lambda
    )
    wandb.watch(policy_updater.value_net)

    trainer = Trainer(
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= f"{project_name}/{phase}/{exp_name}"
    )
    
    checkpoint_path = os.path.join(f"{project_name}", f"{phase}-resume.pt")    
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
    run()
