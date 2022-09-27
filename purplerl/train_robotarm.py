import os

import random
import numpy as np
import torch

import wandb

from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment, ActionTuple

from gym.spaces import Box

from purplerl.experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import EnvManager
from purplerl.policy import ContinuousPolicy, PPO
from purplerl.config import GpuConfig
from purplerl.vision_models import half_unet_v1

cfg = GpuConfig()

class RobotArmEnvManager(EnvManager):
    def __init__(self, file_name, action_scaling = 2.0, *, port_ = 6064, seed = 0, timeout_wait = 120):
        self.action_scaling = action_scaling

        self.stats_channel = StatsSideChannel()
        self.params_channel = EnvironmentParametersChannel()
        self.config_channel = EngineConfigurationChannel()
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=0,
            base_port=port_,
            seed=seed,
            # Note: In headless mode visual observations deliver bank images!!!
            no_graphics=False,
            timeout_wait=timeout_wait,
            side_channels=[self.stats_channel, self.params_channel, self.config_channel],
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
            #elif spec.name == "GoalAngels":
            #    self.goalPosSpec = spec
            #    self.goalPosIdx = idx
            elif spec.name == "Remaining":
                self.remainigSpec = spec
                self.remainingIdx = idx
            else:
                raise Exception(f"Unknown spec: {spec.name}")

        assert(self.goalSpec and self.trainingSpec and self.jointPosSpec and self.remainigSpec) #and self.goalPosSpec

        # TODO: Fix channels
        # Channels are the last dimension, but torch needs them to be the first dimension
        self.training_sensor_space = Box(float("-inf"), float("inf"), (1, ) + self.trainingSpec.shape[:-1])
        self.goal_sensor_space = Box(float("-inf"), float("inf"), (1, ) + self.goalSpec.shape[:-1])
        self.joint_angels_space = Box(float("-inf"), float("inf"), self.jointPosSpec.shape)
        self.remaining_space = Box(float("-inf"), float("inf"), self.remainigSpec.shape)
        #self.goal_angels_space = Box(float("-inf"), float("inf"), self.goalPosSpec.shape)

        self.training_range = range(0, np.prod(self.training_sensor_space.shape))
        self.goal_range = range(self.training_range.stop, self.training_range.stop + np.prod(self.goal_sensor_space.shape))
        self.joint_pos_range = range(self.goal_range.stop, self.goal_range.stop + np.prod(self.joint_angels_space.shape))
        self.remaining_range = range(self.joint_pos_range.stop, self.joint_pos_range.stop + np.prod(self.remaining_space.shape))
        #self.goal_angles_range = range(self.remaining_range.stop, self.remaining_range.stop + np.prod(self.goal_angels_space.shape))

        obs_length = (
                np.prod(np.array(self.training_sensor_space.shape)) +
                np.prod(np.array(self.goal_sensor_space.shape)) +
                np.prod(np.array(self.joint_angels_space.shape)) +
                np.prod(np.array(self.remaining_space.shape))
                #np.prod(np.array(self.goal_angels_space.shape))
            ).item()

        self.observation_space = Box(float("-1"), float("1"), (obs_length, ))

        self.action_space = Box(float("-inf"), float("inf"), (5, ))

        self.set_lesson(0)
        self.config_channel.set_configuration_parameters(time_scale=20, target_frame_rate=2, capture_frame_rate=2)

    def reset(self):
        self.env.reset()

        obs, _ = self._get_obs()
        return obs

    def step(self, act: np.ndarray):
        act = act.cpu().numpy()

        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        discrete_actions = np.array([[]], dtype=np.float32)
        for agent_id in decision_steps:
            continuous_actions = self.action_scaling * act[agent_id].reshape((1, -1))
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

        success = rew > 0.0
        return obs, rew, done, success


    def close(self):
        self.env.close()
        self.env = None


    def set_lesson(self, lesson):
        if lesson > 3:
            return False;

        self.params_channel.set_float_parameter("lessonIndex", lesson)
        return True


    def _get_obs(self) -> tuple[np.array, list[float]]:
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        rew = np.zeros([self.env_count], dtype=np.float32)
        trainingObs = np.zeros([self.env_count] + list(self.trainingSpec.shape), dtype= np.float32)
        goalObs = np.zeros([self.env_count] + list(self.goalSpec.shape), dtype= np.float32)
        jointPosObs = np.zeros([self.env_count] + list(self.jointPosSpec.shape), dtype= np.float32)
        #goalPosObs = np.zeros([self.env_count] + list(self.goalPosSpec.shape), dtype= np.float32)
        remainingObs = np.zeros([self.env_count] + list(self.remainigSpec.shape), dtype= np.float32)

        obs_count = 0;
        found = [None] * self.env_count
        for step_index, agent_id in enumerate(decision_steps.agent_id):
            obs_count += 1
            assert(found[agent_id] == None)
            found[agent_id] == True

            trainingObs[agent_id] = decision_steps.obs[self.trainingIdx][step_index]
            goalObs[agent_id] = decision_steps.obs[self.goalIdx][step_index]
            jointPosObs[agent_id] = decision_steps.obs[self.jointPosIdx][step_index]
            #goalPosObs[agent_id] = decision_steps.obs[self.goalPosIdx][step_index]
            remainingObs[agent_id] = decision_steps.obs[self.remainingIdx][step_index]
            rew[agent_id] = decision_steps.reward[step_index]

        assert obs_count == self.env_count, f"{obs_count} != {self.env_count}"
        obs = np.concatenate(
            (
                trainingObs.reshape((self.env_count, -1)),
                goalObs.reshape((self.env_count, -1)),
                jointPosObs.reshape((self.env_count, -1)),
                remainingObs.reshape((self.env_count, -1)),
                #goalPosObs.reshape((self.env_count, -1)),
            ), axis=-1
        )
        return obs, rew


    def _training_obs(self, obs: torch.tensor):
        return obs[:, self.training_range].reshape(*([-1] + list(self.training_sensor_space.shape)))


    def _goal_obs(self, obs: torch.tensor):
        # TODO
        return obs[:, self.goal_range].reshape(*([-1] + list(self.goal_sensor_space.shape)))


    def _joint_angles(self, obs: torch.tensor):
        # TODO
        return obs[:, self.joint_pos_range].reshape(*([-1] + list(self.joint_angels_space.shape)))


    #def _goal_angles(self, obs: torch.tensor):
    #    # TODO
    #    return obs[:, self.goal_angles_range].reshape(*([-1] + list(self.goal_angels_space.shape)))


    def _remaining(self, obs: torch.tensor):
        # TODO
        return obs[:, self.remaining_range].reshape(*([-1] + list(self.remaining_space.shape)))

class RobotArmObsEncoder(torch.nn.Module):
    def __init__(self, env: RobotArmEnvManager) -> None:
        super().__init__()

        self.env = env

        self.cnn_layers = half_unet_v1(np.array(list(env.training_sensor_space.shape[1:]), np.int32))

        self.shape: tuple[int, ...] = (128 + 128 + 5 + 1, )

    def forward(self, obs: torch.tensor):
        # Note: obs can be of shape (num_envs, obs_shape) or (num_envs, buffer_size, obs_shape)
        buffer_dims = obs.shape[:-1]
        obs = obs.reshape((-1, obs.shape[-1], ))

        enc_obs = self._forward(obs)

        # restore the buffer dimension
        return enc_obs.reshape(buffer_dims + (-1, ))


    def _forward(self, obs: torch.tensor):
        # obs.shape == [batch_size, self.observation_space.shape]
        training_obs = self.env._training_obs(obs)
        goal_obs = self.env._goal_obs(obs)
        joint_pos = self.env._joint_angles(obs)
        remaining = self.env._remaining(obs)

        # flatten buffer dimensions since the cnn only accepts 3D or 4D input
        enc_training = self.cnn_layers(training_obs)
        with torch.no_grad():
            enc_goal = self.cnn_layers(goal_obs)

        return torch.concat((enc_training, enc_goal, joint_pos, remaining), -1)


def run(dev_mode = False):
    phase_config = {
        "phase1": {
            "new_lesson_vf_only_updates": 8,
            "lesson_timeout_episodes": 80,
            "policy_lr": 2e-4,
            "vf_lr": 2e-4,
            "update_epochs" : 7,
            "discount": 0.95,
            "adv_lambda": 0.95,
            "clip_ratio": 0.10,
            "target_kl": 0.02,
            "target_vf_delta": 1.0,
            "lr_decay": 0.90,
            "action_scaling": 2.0,

            "update_batch_size": 75,
            "update_batch_count": 2,
            "epochs": 1000
        }
    }
    active_phase = "phase1"
    config = phase_config[active_phase]
    config["phase"] = active_phase
    wandb_mode = "online" if not dev_mode else "disabled"
    with wandb.init(project="RobotArm", config=config, mode=wandb_mode) as run:
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
    action_scaling: float = 2.0,

    lesson_timeout_episodes: int = 80,
    new_lesson_vf_only_updates: int = 0,
    update_batch_size: int = 29,
    update_batch_count: int = 2,
    epochs: int = 3000
):
    buffer_size = update_batch_size * update_batch_count

    save_freq = 50

    file_name = "/home/cthoens/code/UnityRL/ml-agents-robots/Builds/RobotArm.x86_64"
    env_manager= RobotArmEnvManager(file_name, action_scaling = action_scaling)

    # TODO Tell wandb about it
    num_envs = env_manager.env_count

    policy= ContinuousPolicy(
        obs_encoder = RobotArmObsEncoder(env_manager),
        action_space = env_manager.action_space,
        hidden_sizes = [64],
        std_scale = 0.5,
        min_std= torch.as_tensor([0.2, 0.2, 0.2, 0.2, 0.2]),
        max_std= torch.as_tensor([0.6, 0.6, 0.6, 0.6, 0.6])
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
        hidden_sizes = [64],
        policy_lr = policy_lr,
        vf_lr = vf_lr,
        vf_only_updates = new_lesson_vf_only_updates,
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
        lesson_timeout_episodes = lesson_timeout_episodes,
        new_lesson_vf_only_updates = new_lesson_vf_only_updates,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= out_dir,
        #eval_func =
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

    run(args.dev)
