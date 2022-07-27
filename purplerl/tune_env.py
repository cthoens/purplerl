from ast import Pass
from typing import Callable

import numpy as np
import torch
from torch.nn import Conv2d, Sequential, BatchNorm2d, ReLU, MaxPool2d, Linear, Flatten

import gym
from gym.spaces import Box

from purplerl.trainer import Trainer
from purplerl.sync_experience_buffer import ExperienceBuffer
from purplerl.policy import PPO
from purplerl.train_gym import create_trainer

class StatsAndParamsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn_layers = Sequential(
            # 12x24x1
            Conv2d(in_channels=1, out_channels=64, kernel_size=(3, TuneEnv.NUM_STATS), stride=1),
            ReLU(inplace=True),
            # 1x22x64
            Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 1), stride=1),
            ReLU(inplace=True),
            # 1x20x32
            Conv2d(in_channels=32, out_channels=24, kernel_size=(3, 1), stride=1),
            ReLU(inplace=True),
            # 1x18x24
            Flatten(),
            # 432
            Linear(432, 32)
        )

        self.shape: tuple[int, ...] = (32+TuneEnv.NUM_PARAMS, )


    def forward(self, obs: torch.tensor):
        params_obs = obs[...,:TuneEnv.NUM_PARAMS]
        stats_obs = obs[...,TuneEnv.NUM_PARAMS:]
        # flatten buffer dimensions since the cnn only accepts 3D or 4D input
        input = stats_obs.reshape(-1, *TuneEnv.STATS_OBS_SPACE.shape)
        x = self.cnn_layers(input)

        return torch.concat((x, params_obs), -1)


class PassesEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = Sequential(
            # 8x35
            Flatten(),
            # 280
        )

        self.shape: tuple[int, ...] = (280, )


    def forward(self, obs: torch.tensor):
        x = self.layers(obs)

        return x


class TuneEnvObsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.stats_and_params_encoder = StatsAndParamsEncoder()
        self.passes_encoder = PassesEncoder()

        self.shape: tuple[int, ...] = (210, )

    def forward(self, obs: torch.tensor):
        # obs: ([buffer,] envs, passes, flat stats and params)
        buffer_dims = list(obs.shape)[:-2]

        input = obs.reshape(-1, *TuneEnv.STATS_AND_PARAMS_OBS_SPACE.shape)

        # input: (-1, flat stats and params)
        x = self.stats_and_params_encoder(input)

        x = x.reshape(-1, TuneEnv.NUM_PASSES, x.shape[-1])

        # x: (-1, passes, encoded stats and params)
        x = self.passes_encoder(x)

        # x: (-1, encoded passes stats and params)
        return x.reshape(*(buffer_dims + [-1]))

class TuneEnv(gym.Env):

    STATS = [
        ExperienceBuffer.MEAN_RETURN,
        ExperienceBuffer.DISC_REWARD,
        ExperienceBuffer.SUCCESS_RATE,
        ExperienceBuffer.EPISODE_COUNT,
        PPO.POLICY_EPOCHS,
        PPO.VF_EPOCHS,
        PPO.VALUE_LOSS,
        PPO.VF_LOSS_RISE,
        PPO.KL,
        PPO.POLICY_LOSS,
        PPO.CLIP_FACTOR,
        Trainer.ENTROPY
    ]

    NUM_PARAMS = 3
    NUM_EPOCHS = 24
    NUM_STATS = len(STATS)
    NUM_PASSES = 6

    ACTION_SPACE = Box(float("-1"), float("1"), (3, ))
    STATS_OBS_SPACE =  Box(float("-1"), float("1"), (1, NUM_EPOCHS, NUM_STATS))
    PARAMS_OBS_SPACE = Box(float("-1"), float("1"), (NUM_PARAMS, ))
    STATS_AND_PARAMS_OBS_SPACE = Box(float("-1"), float("1"),
        tuple(np.prod(np.array(STATS_OBS_SPACE.shape)) + np.array(PARAMS_OBS_SPACE.shape)))
    PASSES_OBSERVATION_SPACE = Box(float("-1"), float("1"),
        (NUM_PASSES, np.prod(STATS_AND_PARAMS_OBS_SPACE.shape)))
    OBSERVATION_SPACE = PASSES_OBSERVATION_SPACE

    def __init__(self, trainer_func: Callable[[dict], Trainer] = create_trainer) -> None:
        self.trainer_func = trainer_func
        self.action_space = self.ACTION_SPACE
        self.observation_space = self.OBSERVATION_SPACE
        self.remaining_passes = self.NUM_PASSES

        self.obs = np.zeros(self.OBSERVATION_SPACE.shape, dtype=np.float32)


    def reset(self):
        self.obs = np.zeros(self.OBSERVATION_SPACE.shape, dtype=np.float32)
        self.remaining_passes = self.NUM_PASSES
        return self.obs


    def step(self, action):
        assert(self.remaining_passes > 0)

        action_min = np.array([1e-6, 1e-6, 8], dtype = np.float32)
        action_max = np.array([1e-3, 1e-3, 16], dtype = np.float32)

        action_lerp = action_min + (action_max - action_min) * action
        action_lerp = np.clip(action_lerp, action_min, action_max)

        assert((action_lerp >= action_min).all() and (action_lerp <= action_max).all())

        meta_params = {
            "project_name": "TuneWorkbook",
            "exp_name": f"policy_lr_{float(action_lerp[0]):.4e} vf_lr_{float(action_lerp[1]):.4e} epochs_{int(action_lerp[2])}",
            "policy_lr": float(action_lerp[0]),
            "vf_lr": float(action_lerp[1]),
            "update_epochs" : int(action_lerp[2]),
        }
        print(f"{self.remaining_passes}: {meta_params['exp_name']}")
        trainer: Trainer = self.trainer_func(**meta_params)

        # rotate obs
        self.obs[1:] = self.obs[0:-1]

        # add new data to obs
        param_obs = self.obs[0][:self.NUM_PARAMS]
        param_obs[...] =  (action * 2.0) - 1.0

        # Note: shape[1:] -> skip dummy channel
        stats_obs = self.obs[0][self.NUM_PARAMS:].reshape(self.STATS_OBS_SPACE.shape[1:])
        for epoch_stats_obs in stats_obs:
            trainer.run_epoch()
            flattened_stats = {}
            for entry in trainer.all_stats.values():
                flattened_stats.update(entry)
            for stat_idx, stat_key in enumerate(self.STATS):
                epoch_stats_obs[stat_idx] = flattened_stats[stat_key]

        # TODO: accumulate?
        self.remaining_passes -= 1
        DISC_REWARD_OBS_IDX = 1
        disc_reward_begin = stats_obs[:3, DISC_REWARD_OBS_IDX].mean()
        disc_reward_end = stats_obs[-3:, DISC_REWARD_OBS_IDX].mean()
        reward = np.clip((disc_reward_end - disc_reward_begin) / abs(disc_reward_begin), [-1.0], [1.0]).item()
        return self.obs, reward, self.remaining_passes==0, {}


    def render(self, mode="human"):
        pass