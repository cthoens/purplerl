import numpy as np

import torch.nn

import gym


class EnvManager:
    def reset(self) -> list[np.ndarray]:
        """
        Resets all environments

        Returns:
            first observation per environent
        """
        raise NotImplemented()

    def step(self, act: np.ndarray) -> list[(np.ndarray, np.ndarray, np.ndarray)]:
        """
        Steps all environments. Environments that reach the end of their episode are reset automatically.

        Note that this will not return the final observation of an episode, but the first observation after the reset

        Args:
            obs (Tensor): one observation per environment

        Returns:
            Tuple of
              - obs (Tensor): next observation per environent
              - reward (Tensor of float): reward per environment
              - done (Tensor of bool): true if the environments episode has ended and the environment was reset

        """
        raise NotImplemented

    def set_lesson(self, lesson):
        return False

    def close(self):
        pass


class GymEnvManager(EnvManager):
    def __init__(self, env='CartPole-v0', num_envs=2) -> None:
        if callable(env):
            self.envs = [env() for _ in range(num_envs)]
        else:
            self.envs = [gym.make(env) for _ in range(num_envs)]
        self.done = [False for _ in range(num_envs)]
        self.success = [False for _ in range(num_envs)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """
        Resets all environments

        Returns:
            first observation per environent
        """
        obs = np.array([env.reset() for env in self.envs], dtype=np.float32)
        return obs

    def step(self, act: np.ndarray):
        """
        Steps all environments. Environments that reach the end of their episode are reset automatically.

        Note that this will not return the final observation of an episode, but the first observation after the reset

        Args:
            obs (Tensor): one observation per environment

        Returns:
            Tuple of
              - obs (Tensor): next observation per environent
              - reward (Tensor of float): reward per environment
              - done (Tensor of bool): true if the environments episode has ended and the environment was reset

        """
        act = act.cpu().numpy()
        interaction =  [env.step(act.reshape(self.action_space.shape)) for env, act in zip(self.envs, act)]
        obs =  np.array([next_obs for next_obs, _, _, _ in interaction], dtype=np.float32)
        rew =  np.array([rew for _, rew, _, _ in interaction], dtype=np.float32)
        done = np.array([done for _, _, done, _ in interaction], dtype=np.bool8)
        success = np.array([info.get("success", False) for _, _, _, info in interaction], dtype=np.bool8)
        for index, (env, env_done) in enumerate(zip(self.envs, done)):
            if not env_done:
                continue
            obs[index] = env.reset()
        return obs, rew, done, success

    def set_lesson(self, lesson):
        set_lesson = getattr(self.envs[0], "set_lesson", None)
        if not callable(set_lesson):
            return False

        more_lessons = np.array([env.set_lesson(lesson) for env in self.envs])
        assert(np.all(more_lessons == more_lessons[0]))
        return more_lessons[0]


class IdentityObsEncoder(torch.nn.Module):
    def __init__(self, env: GymEnvManager) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = env.observation_space.shape

    def forward(self, obs: any):
        return obs