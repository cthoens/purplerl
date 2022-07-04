from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig
import torch

from purplerl.trainer import CategoricalPolicy, ExperienceBufferBase, RewardToGo, Trainer
from purplerl.environment import UnityEnvManager, ObsType
from purplerl.config import device

class RobotArmObsEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shape: tuple[int, ...] = (12, )

    def forward(self, obs: ObsType):
        return torch.concat((obs.jointPos, obs.goalPos), -1)

class ExperienceBufferEx(ExperienceBufferBase):
    def __init__(self, env: UnityEnvManager, buffer_size: int):
        super().__init__(
            num_envs = env.env_count,
            buffer_size = buffer_size,
            act_shape = env.action_space.shape
        )
        self.trainingObs = torch.zeros(self.num_envs, buffer_size, *env.trainingSpec.shape, dtype= torch.float32, device=device)
        self.goalObs = torch.zeros(self.num_envs, buffer_size, *env.goalSpec.shape, dtype= torch.float32, device=device)
        self.jointPosObs = torch.zeros(self.num_envs, buffer_size, *env.jointPosSpec.shape, dtype= torch.float32, device=device)
        self.goalPosObs = torch.zeros(self.num_envs, buffer_size, *env.goalPosSpec.shape, dtype= torch.float32, device=device)
        self.remainingObs = torch.zeros(self.num_envs, buffer_size, *env.remainigSpec.shape, dtype= torch.float32, device=device)
        self.obs = ObsType(
            training= self.trainingObs,
            goal= self.goalObs,
            jointPos= self.jointPosObs,
            goalPos= self.goalPosObs,
            remaining= self.remainingObs
        )

    def step(self, obs: ObsType, act: torch.Tensor, reward: torch.Tensor):
        # Update overall data
        self.trainingObs[:,self.next_step_index] = obs.training
        self.goalObs[:,self.next_step_index] = obs.goal
        self.jointPosObs[:,self.next_step_index] = obs.jointPos
        self.goalPosObs[:,self.next_step_index] = obs.goalPos
        self.remainingObs[:,self.next_step_index] = obs.remaining

        # Note: Call after updating to prevent self.next_step_index from getting incremented too soon
        super().step(torch.empty(0), act, reward)

    def reset(self):
        super().reset()

        self.trainingObs[...] = 0.0
        self.goalObs[...] = 0.0
        self.jointPosObs[...] = 0.0
        self.goalPosObs[...] = 0.0
        self.remainingObs[...] = 0.0


def run():

    seed = 0
    exp_name = "RobotArmTest"
    env_name = "RobotArm"

    env_manager= UnityEnvManager(
        file_name="/home/cthoens/code/UnityRL/ml-agents-robots/Builds/RobotArm.x86_64",
        headless=False
    )
    env_manager.params_channel.set_float_parameter("lesson", 0.0)
    env_manager.config_channel.set_configuration(
        EngineConfig(
            width = 544,
            height = 168,
            quality_level=1,
            time_scale = 100.0,
            target_frame_rate = 2,
            capture_frame_rate = -1
        )
    )

    policy= CategoricalPolicy(
        obs_encoder=RobotArmObsEncoder(),
        hidden_sizes=[32, 32, 32],
        action_space = env_manager.action_space
    )

    experience = ExperienceBufferEx(
        env=env_manager,
        buffer_size=1000
    )

    trainer = Trainer()
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
        epochs=500,
        save_freq=100,
    )

if __name__ == '__main__':
    run()
