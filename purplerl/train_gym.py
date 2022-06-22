import math
import time
from enum import Enum

import numpy as np
from purplerl.trainer import Trainer
import torch

from gym.spaces import Discrete, MultiDiscrete, Box

from purplerl.environment import EnvManager, GymEnvManager, IdentityObsEncoder
from purplerl.sync_experience_buffer import ExperienceBufferBase, MonoObsExperienceBuffer
from purplerl.policy import StochasticPolicy, CategoricalPolicy, ContinuousPolicy
from purplerl.policy import PolicyUpdater, RewardToGo, Vanilla
from purplerl.config import device


class Algo(Enum):
    REWARD_TO_GO = 2,
    VANILLA = 3


def to_tensor(input: np.ndarray) -> torch.tensor:
        return torch.as_tensor(input, dtype=torch.float32, device=device)


class GymTrainer(Trainer):
    def __init__(self, logger_kwargs=dict()) -> None:
        self.logger = EpochLogger(**logger_kwargs)

    def train(self,
            obs_encoder : torch.nn.Module = None,
            env: EnvManager|str ='BipedalWalker-v3',
            hidden_sizes=[64, 64],
            policy_lr=1e-2,
            value_net_lr=1e-2,
            batch_size=4,
            buffer_size=5000,
            seed=20,
            epochs=50,
            save_freq=20,
            algo: Algo|str="REWARD_TO_GO"
        ):

        params = dict(locals())
        params.pop("self", None)
        params.pop("obs_encoder", None)
        params.pop("env", None)
        self.logger.save_config(params)

        if isinstance(algo, str):
            algo = Algo[algo]

        # make environment, check spaces, get obs / act dims
        if isinstance(env, str):
            env_manager: EnvManager = GymEnvManager(env_name=env, batch_size=batch_size)
        else:
            env_manager = env

        obs_shape = list(env_manager.observation_space.shape)

        if obs_encoder is None:
            obs_encoder = IdentityObsEncoder(env_manager)

        # policy builder depends on action space
        if isinstance(env_manager.action_space, Box):
            policy = ContinuousPolicy(obs_encoder, hidden_sizes, env_manager.action_space)
        elif isinstance(env_manager.action_space, (Discrete, MultiDiscrete)):
            # A discrete action shpace has one action that can take n possible values
            policy = CategoricalPolicy(obs_encoder, hidden_sizes, env_manager.action_space)
        else:
            raise Exception("Unsupported action space")

        # make optimizer
        experience = MonoObsExperienceBuffer(batch_size, buffer_size, obs_shape, policy.action_shape)

        if algo == Algo.REWARD_TO_GO:
            policy_updater = RewardToGo(policy, experience, policy_lr)
        elif algo == Algo.VANILLA:
            policy_updater = Vanilla(obs_shape, hidden_sizes, policy, experience, policy_lr, value_net_lr)
        else:
            raise Exception("Unknown algorithm")

        self.run_training(env_manager, experience, policy, policy_updater, epochs, save_freq)


def run():
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument('--env_name', '--env', type=str, default='MountainCarContinuous-v0')
    parser.add_argument('--env_name', '--env', type=str, default='LunarLander-v2')
    #parser.add_argument('--env_name', '--env', type=str, default='BipedalWalker-v3')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--value_lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--algo', type=str, default="REWARD_TO_GO")
    parser.add_argument('--exp_name', type=str, default='simple')

    args = parser.parse_args()

    for i in range(6):
        seed = i*500

        trainer = Trainer()

        trainer.train(env=args.env_name, algo=args.algo, epochs=args.epochs, policy_lr=args.lr, value_net_lr=args.value_lr, seed=seed)

if __name__ == '__main__':
    run()
