import os
import gym

import torch
import wandb

from purplerl.sync_experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import GymEnvManager
from purplerl.policy import ContinuousPolicy, PPO
from purplerl.config import device
from purplerl.eval_workbook import do_eval
from purplerl.resnet import resnet18
from purplerl.vision_models import half_unet_v3
import purplerl.workbook_env as env

class WorkbenchObsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        #self.cnn_layers = resnet18(num_classes=128)
        #self.cnn_layers = half_unet_v1()
        #self.cnn_layers = half_unet_v2()
        self.cnn_layers = half_unet_v3()

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
            "policy_lr": 2e-5,
            "vf_lr": 2e-4,
            "update_epochs" : 10,
            "discount": 0.95,
            "adv_lambda": 0.95,
            "clip_ratio": 0.2
        },
        "phase2": {
            "policy_lr": 1e-4,
            "vf_lr": 2.5e-4,
            "update_epochs" : 10,
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
    vf_lr,
    update_epochs,
    discount,
    adv_lambda,
    phase,
    clip_ratio: float = 0.2,
    target_kl: float = 0.01,
):
    num_envs = 64
    update_batch_size = 1216 // num_envs
    buffer_size = update_batch_size * 2

    epochs = 2000
    save_freq = 50

    gym.envs.register(
        id='workbook-v0',
        entry_point='purplerl.workbook_env:WorkbookEnv',
        kwargs={
        },
    )

    env_manager= GymEnvManager('workbook-v0', num_envs=num_envs)

    policy= ContinuousPolicy(
        obs_encoder = WorkbenchObsEncoder(),
        hidden_sizes = [128, 128],
        action_space = env_manager.action_space,
        std_scale = 2.0,
        min_std= torch.as_tensor([0.5, 0.5])
    ).to(device)

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
    wandb.watch((policy, policy_updater.value_net_tail), log='all', log_freq=20)

    out_dir = f"results/{project_name}/{phase}/{exp_name}"
    trainer = Trainer(
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

    trainer.run_training()
    env_manager.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    args = parser.parse_args()

    import matplotlib
    matplotlib.use('Agg')

    env.init(sheet_path="sheets-64")

    run(args.dev)
