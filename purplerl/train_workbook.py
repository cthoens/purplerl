import gym

import torch
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Flatten

from purplerl.simple import ContinuousPolicy, ExperienceBufferBase, IdentityObsEncoder, MonoObsExperienceBuffer, RewardToGo, Trainer
from purplerl.environment import GymEnvManager, UnityEnvManager, ObsType
from purplerl.config import gpu
from purplerl.workbook_env import WorkbookEnv

import os.path as osp

def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model'+str(itr)+'.pt')
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

    from purplerl.spinup.utils.run_utils import setup_logger_kwargs

    seed = 0
    exp_name = "RobotArmTest"
    env_name = "RobotArm"
    batch_size = 4
    buffer_size = 550
    logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=f"./{env_name}")

    gym.envs.register(
        id='workbook-v0',
        entry_point='purplerl.workbook_env:WorkbookEnv',
        max_episode_steps=150,
        kwargs={},
    )

    env_manager= GymEnvManager('workbook-v0', batch_size=batch_size)

    #policy= ContinuousPolicy(
    #    obs_encoder=WorkbenchObsEncoder(),
    #    hidden_sizes=[32, 32, 32],
    #    action_space = env_manager.action_space
    #)


    policy = load_pytorch_policy("/home/cthoens/code/UnityRL/robotarm/results/RobotArm/RobotArmTest/RobotArmTest_s0/", "4000-backup")

    experience = MonoObsExperienceBuffer(
        batch_size, 
        buffer_size, 
        env_manager.observation_space.shape, 
        policy.action_shape
    )

    trainer = Trainer(logger_kwargs)
    trainer.run_training(
        env_manager= env_manager,
        experience= experience,
        policy= policy,
        policy_updater= RewardToGo(
            policy=policy,
            experience=experience,
            policy_lr=1e-3,
        ),
        #policy_updater= ValueFunction(
        #    obs_shape=None,
        #    hidden_sizes=[32, 32],
        #    policy=policy,
        #    experience=experience,
        #    policy_lr=1e-3,
        #    value_net_lr = 1e-3,
        #),
        epochs=5000,
        save_freq=1000,
    )

if __name__ == '__main__':
    run()
