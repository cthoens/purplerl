import os
import gym

import random
import torch
import numpy as np

import torch
import wandb

from purplerl.sync_experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import GymEnvManager
from purplerl.policy import ContinuousPolicy, PPO
from purplerl.config import GpuConfig
from purplerl.eval_workbook import do_eval
from purplerl.resnet import resnet18
from purplerl.vision_models import half_unet_v1
import purplerl.workbook_env as env

cfg = GpuConfig()

class WorkbenchObsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        #self.cnn_layers = resnet18(num_classes=128)
        self.cnn_layers = half_unet_v1(np.array(list(env.SHEET_OBS_SPACE.shape[1:]), np.int32))
        #self.cnn_layers = half_unet_v2()
        #self.cnn_layers = half_unet_v3()

        self.shape: tuple[int, ...] = (128+1+2, )

    def forward(self, obs: torch.tensor):
        # Note: obs can be of shape (num_envs, sheet_shape) or (num_envs, buffer_size, sheet_shape)
        buffer_dims = list(obs.shape)[:-1]
        sheet_obs = obs[...,:-3]
        extra_obs = obs[...,-3:]
        # flatten buffer dimensions since the cnn only accepts 3D or 4D input
        x = self.cnn_layers(sheet_obs.reshape(-1, *env.SHEET_OBS_SPACE.shape))
        # restore the buffer dimension
        x = x.reshape(*(buffer_dims + [128]))

        return torch.concat((x, extra_obs), -1)


def run(dev_mode = False):
    phase_config = {
        "phase1": {
            "vf_only_update": False,
            "policy_lr": 2e-5,
            "vf_lr": 2e-4,
            "update_epochs" : 10,
            "discount": 0.95,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2,
            "target_kl": 0.02,
            "target_vf_delta": 1.0,
            "lr_decay": 0.90,

            "num_envs": 64,
            "update_batch_size": 19, # 29
            "update_batch_count": 2,
            "epochs": 30
        },
        "phase2": {
            "vf_only_update": False,
            "policy_lr": 2e-5,
            "vf_lr": 2e-4,
            "update_epochs" : 10,
            "discount": 0.95,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2,
            "target_kl": 0.02,
            "target_vf_delta": 1.0,
            "lr_decay": 0.90,

            "num_envs": 64,
            "update_batch_size": 19, # 29
            "update_batch_count": 2,
            "epochs": 2000
        }
    }
    active_phase = "phase2"
    config = phase_config[active_phase]
    config["phase"] = active_phase
    wandb_mode = "online" if not dev_mode else "disabled"
    with wandb.init(project="Workbook", config=config, mode=wandb_mode) as run:
        project_name = run.project
        if project_name=="": project_name = "Dev"
        run_training(project_name, run.name, **wandb.config)

def create_trainer(
    project_name,
    exp_name,
    policy_lr,
    vf_lr,
    update_epochs,
    discount: float = 0.95,
    adv_lambda: float = 0.95,
    phase = "phase1",
    clip_ratio: float = 0.2,
    target_kl: float = 0.15,
    target_vf_delta: float = 1.0,
    lr_decay: float = 0.95,

    vf_only_update: bool = False,
    num_envs: int = 64,
    update_batch_size: int = 29,
    update_batch_count: int = 2,
    epochs: int = 3000
):
    buffer_size = update_batch_size * update_batch_count

    save_freq = 100

    env_manager= GymEnvManager(env.WorkbookEnv, num_envs=num_envs)

    policy= ContinuousPolicy(
        obs_encoder = WorkbenchObsEncoder(),
        action_space = env_manager.action_space,
        hidden_sizes = [128, 128],
        std_scale = 2.0,
        min_std= torch.as_tensor([0.5, 0.5])
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
        vf_only_update= vf_only_update,
        policy_lr = policy_lr,
        vf_lr = vf_lr,
        update_epochs = update_epochs,
        update_batch_size=update_batch_size,
        lam = adv_lambda,
        clip_ratio = clip_ratio,
        target_kl = target_kl,
        target_vf_delta = target_vf_delta,
        lr_decay = lr_decay
    )
    wandb.watch((policy, policy_updater.value_net_tail), log='all', log_freq=20)

    out_dir = f"results/{project_name}/{phase}/{exp_name}"
    trainer = Trainer(
        cfg = cfg,
        env_manager = env_manager,
        experience = experience,
        policy = policy,
        policy_updater = policy_updater,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= out_dir,
        eval_func = lambda epoch, policy_updater: do_eval(out_dir, epoch, policy_updater)
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
    return trainer

def run_training(project_name, exp_name, **kwargs):
    trainer = create_trainer(project_name, exp_name, **kwargs)
    trainer.run_training()
    trainer.env_manager.close()

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

    env.init(sheet_path="sheets-64")

    run(args.dev)
