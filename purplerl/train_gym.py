import os

from gym.envs import registration

import random
import torch
import numpy as np

import torch
import wandb

from purplerl.experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import GymEnvManager, IdentityObsEncoder
from purplerl.policy import ContinuousPolicy, PPO
from purplerl.config import GpuConfig

cfg = GpuConfig()


def run(dev_mode = False):
    phase_config = {
        "phase1": {
            "policy_lr": 2e-4,
            "vf_lr": 2e-4,
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
    with wandb.init(project="Gym", config=config, mode=wandb_mode) as run:
        project_name = run.project
        if project_name=="": project_name = "GymDev"
        run_training(project_name, run.name, **wandb.config)

def create_trainer(
    project_name,
    exp_name,
    policy_lr,
    vf_lr,
    update_epochs,
    phase = "phase1",
    discount: float = 0.95,
    adv_lambda: float = 0.95,
    clip_ratio: float = 0.2,
    target_kl: float = 0.01,
):
    num_envs = 8
    update_batch_size = 240
    buffer_size = 240

    epochs = 2000
    save_freq = 500

    def make_env():
        return registration.make('BipedalWalker-v3', max_episode_steps=80)

    env_manager= GymEnvManager(make_env, num_envs=num_envs)

    policy= ContinuousPolicy(
        obs_encoder = IdentityObsEncoder(env_manager),
        action_space = env_manager.action_space,
        hidden_sizes = [128, 128],
        std_scale = 1.0,
    ).to(cfg.device)

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
        target_kl = target_kl,
    )
    #wandb.watch((policy, policy_updater.value_net_tail), log='all', log_freq=20)

    out_dir = f"results/{project_name}/{phase}/{exp_name}"
    trainer = Trainer(
        cfg = cfg,
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= out_dir
    )

    checkpoint_path = os.path.join(f"results/{project_name}", f"{phase}-resume.pt")
    if os.path.exists(checkpoint_path):
        if os.path.islink(checkpoint_path):
            print(f"Resuming from {checkpoint_path}[{os.readlink(checkpoint_path)}]")
        else:
            print(f"Resuming from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)
    #else:
    #    print("****\n**** Starting from scratch !!!\n****")
    return trainer

def run_training(project_name, exp_name, **kwargs):
    trainer = create_trainer(project_name, exp_name, **kwargs)
    trainer.run_training()
    trainer.env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    seed = 45632
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


    import matplotlib
    matplotlib.use('Agg')

    run(args.dev)
