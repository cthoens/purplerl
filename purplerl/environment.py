from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from collections import namedtuple

import torch
import gym
from gym.spaces import Box, MultiDiscrete, Tuple as TupleSpace

from mlagents_envs.environment import UnityEnvironment, ActionTuple
from purplerl.config import gpu


class EnvManager:
    def reset(self) -> list[torch.tensor]:
        """
        Resets all environments

        Returns:
            first observation per environent
        """
        raise NotImplemented()

    def step(self, act: torch.tensor) -> list[(torch.tensor, torch.tensor, torch.tensor)]:
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

ObsType = namedtuple('ObsType', ['training', 'goal', 'jointPos', 'goalPos', 'remaining'])

class UnityEnvManager(EnvManager):
    def __init__(self, file_name, *, port_ = 5004, seed = 0, headless = True, timeout_wait = 120):
        self.stats_channel = StatsSideChannel()
        self.params_channel = EnvironmentParametersChannel()
        self.config_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=0,
            base_port=port_,
            seed=seed,
            no_graphics=headless,
            timeout_wait=timeout_wait,
            side_channels=[self.stats_channel, self.params_channel],
            log_folder="."
        )

        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        decision_steps, _ = self.env.get_steps(self.behavior_name)
        self.env_count = len(decision_steps)

        spec = self.env.behavior_specs[self.behavior_name]
        for idx, spec in enumerate(spec.observation_specs):
            if spec.name == "GoalSensor":
                self.goalSpec = spec
                self.goalIdx = idx
            elif spec.name == "TrainingSensor":
                self.trainingSpec = spec
                self.trainingIdx = idx
            elif spec.name == "JointAngels":
                self.jointPosSpec = spec
                self.jointPosIdx = idx
            elif spec.name == "GoalAngels":
                self.goalPosSpec = spec
                self.goalPosIdx = idx
            elif spec.name == "ObservableAttribute:RobotAgent.Remaining":
                self.remainigSpec = spec
                self.remainingIdx = idx
            else:
                raise Exception(f"Unknown spec: {spec.name}")

        assert(self.goalSpec and self.trainingSpec and self.jointPosSpec and self.goalPosSpec and self.remainigSpec)

        self.obs_space = TupleSpace(
            [
                # TrainingSensor
                Box(float("-inf"), float("inf"), self.trainingSpec.shape),
                # GoalSensor
                Box(float("-inf"), float("inf"), self.goalSpec.shape),
                # JointAngels
                Box(float("-inf"), float("inf"), self.jointPosSpec.shape),
                # GoalAngels
                Box(float("-inf"), float("inf"), self.goalPosSpec.shape),
                # Remaining
                Box(float("-inf"), float("inf"), self.remainigSpec.shape),
            ]
        )

        self.action_space = MultiDiscrete([3, 3, 3, 3, 3, 3])

    def reset(self):
        self.env.reset()

        obs, _ = self._get_obs()
        return obs

    def step(self, act: torch.tensor) -> list[ObsType]:
        # Tell the unity env about the actions / decision and step the environment
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        continuous_actions = np.array([[]], dtype=np.float32)
        for agent_id in decision_steps:
            discrete_actions = act[agent_id].cpu().numpy().reshape((1, -1))
            action_tuple = ActionTuple(continuous_actions, discrete_actions)

            self.env.set_action_for_agent(self.behavior_name, agent_id, action_tuple)
        self.env.step()

        # Read the observations, rewards and terminated state of the environments
        obs, rew = self._get_obs()
        _, terminal_steps = self.env.get_steps(self.behavior_name)
        done = [False] * self.env_count
        for terminal_step_index, agent_id in enumerate(terminal_steps.agent_id):
            assert(done[agent_id] == False)
            rew[agent_id] += terminal_steps.reward[terminal_step_index]
            done[agent_id] = True

        rew = torch.as_tensor(rew, dtype=torch.float32, device=gpu)
        done = torch.as_tensor(done, dtype=torch.float32, device=gpu)

        return obs, rew, done

    def close(self):
        self.env.close()
        self.env = None

    def get_success_rate(self):
        stats = self.stats_channel.get_and_reset_stats()["Environment/Success"]
        sum = 0
        for value, _ in stats:
            sum += value
        return sum / len(stats)

    def _get_obs(self) -> tuple[list[ObsType], list[float]]:
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        rew = np.zeros([self.env_count], dtype=np.float32)
        trainingObs = np.zeros([self.env_count] + list(self.trainingSpec.shape), dtype= np.float32)
        goalObs = np.zeros([self.env_count] + list(self.goalSpec.shape), dtype= np.float32)
        jointPosObs = np.zeros([self.env_count] + list(self.jointPosSpec.shape), dtype= np.float32)
        goalPosObs = np.zeros([self.env_count] + list(self.goalPosSpec.shape), dtype= np.float32)
        remainingObs = np.zeros([self.env_count] + list(self.remainigSpec.shape), dtype= np.float32)

        obs_count = 0;
        found = [None] * self.env_count
        for decision_step_index, agent_id in enumerate(decision_steps.agent_id):
            obs_count += 1
            assert(found[agent_id] == None)
            found[agent_id] == True

            trainingObs[agent_id] = decision_steps.obs[self.trainingIdx][decision_step_index]
            goalObs[agent_id] = decision_steps.obs[self.goalIdx][decision_step_index]
            jointPosObs[agent_id] = decision_steps.obs[self.jointPosIdx][decision_step_index]
            goalPosObs[agent_id] = decision_steps.obs[self.goalPosIdx][decision_step_index]
            remainingObs[agent_id] = decision_steps.obs[self.remainingIdx][decision_step_index]
            rew[agent_id] = decision_steps.reward[decision_step_index]

        assert(obs_count == self.env_count)
        obs = ObsType(
            training= torch.as_tensor(trainingObs, device=gpu),
            goal= torch.as_tensor(goalObs, device=gpu),
            jointPos= torch.as_tensor(jointPosObs, device=gpu),
            goalPos= torch.as_tensor(goalPosObs, device=gpu),
            remaining= torch.as_tensor(remainingObs, device=gpu)
        )
        return obs, rew


class GymEnvManager(EnvManager):
    def __init__(self, env_name='CartPole-v0', batch_size=2) -> None:
        self.envs = [gym.make(env_name) for _ in range(batch_size)]
        self.done = [False for _ in range(batch_size)]
        self.success = [False for _ in range(batch_size)]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset(self):
        """
        Resets all environments

        Returns:
            first observation per environent
        """
        obs = torch.as_tensor(np.array([env.reset() for env in self.envs], dtype=np.float32), device=gpu)
        return obs

    def step(self, act: torch.tensor):
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
        interaction =  [env.step(act.reshape(self.action_space.shape).cpu().numpy()) for env, act in zip(self.envs, act)]
        obs =  torch.as_tensor(np.array([next_obs for next_obs, _, _, _ in interaction], dtype=np.float32), device=gpu)
        rew =  torch.as_tensor(np.array([rew for _, rew, _, _ in interaction], dtype=np.float32), device=gpu)
        done = torch.as_tensor(np.array([done for _, _, done, _ in interaction], dtype=np.float32), device=gpu)
        success = np.array([info.get("success", None) for _, _, _, info in interaction], dtype=np.float32)
        for index, (env, env_done) in enumerate(zip(self.envs, done)):
            if not env_done:
                continue
            obs[index] = torch.as_tensor(env.reset(), device=gpu)
        return obs, rew, done, success


class IdentityObsEncoder(torch.nn.Module):
    def __init__(self, env: GymEnvManager) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = env.observation_space.shape

    def forward(self, obs: any):
        return obs