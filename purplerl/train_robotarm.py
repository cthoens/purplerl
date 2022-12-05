import math
import os

import random
import numpy as np

import torch
from torch.distributions.normal import Normal

import wandb

from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import UnityEnvironment, ActionTuple

from gym.spaces import Box

from purplerl.experience_buffer import ExperienceBuffer
from purplerl.trainer import Trainer
from purplerl.environment import EnvManager
from purplerl.policy import ContinuousPolicy, PPO, StochasticPolicy, mlp
from purplerl.config import GpuConfig
from purplerl.resnet_mini import resnet18

cfg = GpuConfig()

class RobotArmEnvManager(EnvManager):
    def __init__(self, file_name, action_scaling = 2.0, force_vulkan = True, *, port_ = 6064, seed = 0, timeout_wait = 120):
        super().__init__()
        self.action_scaling = action_scaling

        self.stats_channel = StatsSideChannel()
        self.params_channel = EnvironmentParametersChannel()
        self.config_channel = EngineConfigurationChannel()
        args = []
        if force_vulkan:
            args += ["-force-vulkan"]
        self.env = UnityEnvironment(
            file_name=file_name,
            worker_id=0,
            base_port=port_,
            seed=seed,
            # Note: In headless mode visual observations deliver bank images!!!
            no_graphics=False,
            timeout_wait=timeout_wait,
            side_channels=[self.stats_channel, self.params_channel, self.config_channel],
            log_folder=".",
            additional_args=args
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
            elif spec.name == "Remaining":
                self.remainigSpec = spec
                self.remainingIdx = idx
            else:
                raise Exception(f"Unknown spec: {spec.name}")

        assert(self.goalSpec and self.trainingSpec and self.remainigSpec) #and self.jointPosSpec and self.goalPosSpec

        # TODO: Fix channels
        # Channels are the last dimension, but torch needs them to be the first dimension
        self.training_sensor_space = Box(float("-inf"), float("inf"), (1, ) + self.trainingSpec.shape[:-1])
        self.goal_sensor_space = Box(float("-inf"), float("inf"), (1, ) + self.goalSpec.shape[:-1])
        self.remaining_space = Box(float("-inf"), float("inf"), self.remainigSpec.shape)
        self.joint_angels_space = Box(float("-inf"), float("inf"), self.jointPosSpec.shape)
        self.goal_angels_space = Box(float("-inf"), float("inf"), self.goalPosSpec.shape)

        self.training_range = range(0, np.prod(self.training_sensor_space.shape))
        self.goal_range = range(self.training_range.stop, self.training_range.stop + np.prod(self.goal_sensor_space.shape))
        self.remaining_range = range(self.goal_range.stop, self.goal_range.stop + np.prod(self.remaining_space.shape))
        self.joint_pos_range = range(self.remaining_range.stop, self.remaining_range.stop + np.prod(self.joint_angels_space.shape))
        self.goal_angles_range = range(self.joint_pos_range.stop, self.joint_pos_range.stop + np.prod(self.goal_angels_space.shape))

        obs_length = (
                np.prod(np.array(self.training_sensor_space.shape)) +
                np.prod(np.array(self.goal_sensor_space.shape)) +
                np.prod(np.array(self.remaining_space.shape)) +
                np.prod(np.array(self.joint_angels_space.shape)) +
                np.prod(np.array(self.goal_angels_space.shape))
            ).item()

        self.observation_space = Box(float("-1"), float("1"), (obs_length, ))

        self.action_space = Box(float("-inf"), float("inf"), (5, ))

        self.set_lesson(0)
        self.config_channel.set_configuration_parameters(time_scale=10, target_frame_rate=10, capture_frame_rate=10)

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


    def update_obs_stats(self, experience:ExperienceBuffer):
        num_actions = np.prod(self.action_space.shape)

        mean_joint_anges = self._joint_angles(experience.obs_merged).mean(-1)
        std_joint_anges = self._joint_angles(experience.obs_merged).std(-1)

        for i in range(num_actions):
            self.stats[f"Joint {i} Mean"] = mean_joint_anges[i].item()
            self.stats[f"Joint {i} Std"] = std_joint_anges[i].item()

        mean_remaining = self._remaining(experience.obs_merged).mean()
        self.stats[f"Mean Remaining"] = mean_remaining.item()


    def close(self):
        self.env.close()
        self.env = None


    def set_lesson(self, lesson):
        self.params_channel.set_float_parameter("lessonIndex", lesson)
        return True


    def _get_obs(self) -> tuple[np.array, list[float]]:
        decision_steps, _ = self.env.get_steps(self.behavior_name)

        rew = np.zeros([self.env_count], dtype=np.float32)
        trainingObs = np.zeros([self.env_count] + list(self.trainingSpec.shape), dtype= np.float32)
        goalObs = np.zeros([self.env_count] + list(self.goalSpec.shape), dtype= np.float32)
        jointPosObs = np.zeros([self.env_count] + list(self.jointPosSpec.shape), dtype= np.float32)
        goalPosObs = np.zeros([self.env_count] + list(self.goalPosSpec.shape), dtype= np.float32)
        remainingObs = np.zeros([self.env_count] + list(self.remainigSpec.shape), dtype= np.float32)

        obs_count = 0;
        found = [None] * self.env_count
        for step_index, agent_id in enumerate(decision_steps.agent_id):
            obs_count += 1
            assert(found[agent_id] == None)
            found[agent_id] == True

            #Scale range from between 0.0 and 1.0 to -1.0 to 1.0
            trainingObs[agent_id] = decision_steps.obs[self.trainingIdx][step_index] * 2.0 - 1.0
            goalObs[agent_id] = decision_steps.obs[self.goalIdx][step_index] * 2.0 - 1.0
            remainingObs[agent_id] = decision_steps.obs[self.remainingIdx][step_index]
            jointPosObs[agent_id] = decision_steps.obs[self.jointPosIdx][step_index]
            goalPosObs[agent_id] = decision_steps.obs[self.goalPosIdx][step_index]
            rew[agent_id] = decision_steps.reward[step_index]

        assert obs_count == self.env_count, f"{obs_count} != {self.env_count}"
        obs = np.concatenate(
            (
                trainingObs.reshape((self.env_count, -1)),
                goalObs.reshape((self.env_count, -1)),
                remainingObs.reshape((self.env_count, -1)),
                jointPosObs.reshape((self.env_count, -1)),
                goalPosObs.reshape((self.env_count, -1)),
            ), axis=-1
        )
        return obs, rew


    def _training_obs(self, obs: torch.tensor):
        return obs[:, self.training_range].reshape(*([-1] + list(self.training_sensor_space.shape)))


    def _goal_obs(self, obs: torch.tensor):
        return obs[:, self.goal_range].reshape(*([-1] + list(self.goal_sensor_space.shape)))


    def _joint_angles(self, obs: torch.tensor):
        return obs[:, self.joint_pos_range].reshape(*([-1] + list(self.joint_angels_space.shape)))


    def _goal_angles(self, obs: torch.tensor):
        return obs[:, self.goal_angles_range].reshape(*([-1] + list(self.goal_angels_space.shape)))


    def _remaining(self, obs: torch.tensor):
        return obs[:, self.remaining_range].reshape(*([-1] + list(self.remaining_space.shape)))

class RobotArmObsEncoder(torch.nn.Module):
    def __init__(self,
        env: RobotArmEnvManager,
        obs_outputs = 128,
        planner_net_layers=[4096, 4096],
        planner_outputs = 1024,
        planner_skip_connection = True,
    ) -> None:
        super().__init__()

        self.env = env
        self.planner_skip_connection = planner_skip_connection
        self.num_obs_outputs = obs_outputs
        self.num_combined_obs_outputs = 2 * self.num_obs_outputs
        self.num_conv_net_outputs = 2 * self.num_obs_outputs + np.prod(env.remaining_space.shape) + np.prod(env.joint_angels_space.shape) + np.prod(env.goal_angels_space.shape)
        self.num_outputs = planner_outputs

        self.training_range = range(0, self.num_obs_outputs)
        self.goal_range = range(self.training_range.stop, self.training_range.stop + self.num_obs_outputs)
        self.training_and_goal_range = range(0, self.training_range.stop + self.num_obs_outputs)
        self.remaining_range = range(self.goal_range.stop, self.goal_range.stop + np.prod(env.remaining_space.shape))
        self.joint_pos_range = range(self.remaining_range.stop, self.remaining_range.stop + np.prod(env.joint_angels_space.shape))
        self.goal_pos_range = range(self.joint_pos_range.stop, self.joint_pos_range.stop + np.prod(env.goal_angels_space.shape))

        #self.cnn_layers = half_unet_v1(np.array(list(env.training_sensor_space.shape[1:]), np.int32), num_outputs=self.num_obs_outputs)
        self.cnn_layers = resnet18(num_classes=self.num_obs_outputs)
        self.mlp = mlp([self.num_conv_net_outputs] + planner_net_layers + [self.num_outputs], activation=torch.nn.ReLU, output_activation=torch.nn.Identity)
        self.enc_obs_relu = torch.nn.ReLU(inplace=True)
        self.skip_relu = torch.nn.ReLU(inplace=True)
        self.shape: tuple[int, ...] = (self.num_outputs, )

    def forward(self, obs: torch.tensor):
        enc_obs = self._forward(obs)
        enc_obs = self.enc_obs_relu(enc_obs)
        out = enc_obs
        out = self.mlp(enc_obs)
        if self.planner_skip_connection:
            out[:, :self.num_combined_obs_outputs] += self._training_and_goal_obs(enc_obs)
        out = self.skip_relu(out)

        # restore the buffer dimension
        return out


    def _forward(self, obs: torch.tensor):
        # obs.shape == [batch_size, self.observation_space.shape]
        training_obs = self.env._training_obs(obs)
        goal_obs = self.env._goal_obs(obs)
        remaining = self.env._remaining(obs)
        joint_angles = self.env._joint_angles(obs)
        goal_angles = self.env._goal_angles(obs)
        enc_training = self.cnn_layers(training_obs)
        enc_goal = self.cnn_layers(goal_obs)

        return torch.concat((enc_training, enc_goal, remaining, joint_angles, goal_angles), -1) #joint_pos,

    def _training_obs(self, obs: torch.tensor):
        return obs[:, self.training_range]


    def _training_and_goal_obs(self, obs: torch.tensor):
        return obs[:, self.training_and_goal_range]


    def _goal_obs(self, obs: torch.tensor):
        return obs[:, self.goal_range]


    def _joint_angles(self, obs: torch.tensor):
        return obs[:, self.joint_pos_range]


    def _goal_angles(self, obs: torch.tensor):
        return obs[:, self.goal_pos_range]


    def _remaining(self, obs: torch.tensor):
        return obs[:, self.remaining_range]


class RobotArmHeuristic(torch.nn.Module):

    def __init__(self,
        real_policy: StochasticPolicy,
        action_scaling: float,
        min_std: float
    ) -> None:
        super().__init__()
        self.real_policy = real_policy
        self.action_scaling = action_scaling
        self.min_std = min_std.to(cfg.device)

        self.obs_encoder = self.real_policy.obs_encoder


    def action_dist(self, obs=None, encoded_obs=None):
        target = self.obs_encoder.env._joint_angles(obs) * 180.0
        goalAngle = self.obs_encoder.env._goal_angles(obs) * 180.0

        batch_size = encoded_obs.shape[0]
        act = torch.zeros(batch_size, 5, **cfg.tensor_args)
        act[target < goalAngle - 5.0] = 5.0
        act[target < goalAngle - 3.0] = 3.0
        act[target < goalAngle - 2.0] = 2.0
        act[target < goalAngle - 1.0] = 1.0

        act[target > goalAngle + 5.0] = -5.0
        act[target > goalAngle + 3.0] = -3.0
        act[target > goalAngle + 2.0] = -2.0
        act[target > goalAngle + 1.0] = -1.0

        #act /= self.action_scaling
        std = self.min_std.repeat(batch_size, 1)

        real_act = self.real_policy.action_dist(encoded_obs=encoded_obs)
        use_real_policy = torch.rand(batch_size) > 0.5

        act[use_real_policy] = real_act.mean[use_real_policy]
        std[use_real_policy] = real_act.stddev[use_real_policy]

        return Normal(loc=act, scale=std)


    def to(self, device):
        self.real_policy.to(device)
        return self


    def checkpoint(self):
        return self.real_policy.checkpoint()


    def load_checkpoint(self, checkpoint):
        self.real_policy.load_checkpoint(checkpoint)


def run(dev_mode:bool = False, resume_lesson: int = None, resume_checkpoint: str = None):
    if dev_mode:
        project_name = "Dev"
    else:
        project_name = "RobotArm"
    if resume_checkpoint:
        if not os.path.exists(resume_checkpoint):
            print(f"Checkpoint does not exist: {resume_checkpoint}")
            return;

    base_config = {
        "lesson_timeout_episodes": 80,
        "new_lesson_warmup_updates":8,

        #policy_update
        "discount": 0.95,
        "adv_lambda": 0.95,
        "clip_ratio": 0.03,
        "entropy_factor": 0.0,

        "policy_imp_min": 0.60,
        "policy_imp_upper_target": 1.50,
        "policy_imp_max": 2.00,
        "policy_valid_imp_lower_bound": 0.92,
        "policy_neg_lower_imp_scale_scale": 0.001,

        "policy_lr_decay": 0.90,
        "policy_initial_lr": 1e-5,
        "policy_update_epochs" : 40,
        "policy_update_batch_size":  15,
        "policy_update_batch_count": 4,

        # vf update
        "vf_imp_min": 0.0,
        "vf_imp_upper_target": 1.8,
        "vf_imp_max": 1000.0,
        "vf_valid_imp_lower_bound": 0.80,

        "vf_lr_decay": 0.90,
        "vf_initial_lr": 1e-8,
        "vf_update_epochs" : 40,
        "vf_update_batch_size":  60,

        # env
        "action_scaling": 5.0,

        "epochs": 2000,
        "resume_lesson": resume_lesson,
        "resume_checkpoint": resume_checkpoint
    }

    unified_config = base_config.copy()
    unified_config.update({
        "obs_encoder_outputs": 128,
        "planner_layers": [4096, 2048, 2048],
        "panner_outputs": 4096,
        "planner_skip_connection": True,
        "policy_split_layers": [],
        "value_net_split_layers": [],
    })

    split_config = base_config.copy()
    split_config.update({
        "obs_encoder_outputs": 128,
        "planner_layers": [2048, 2048],
        "panner_outputs": 4096,
        "planner_skip_connection": False,
        "policy_split_layers": [],
        "value_net_split_layers": [2048, 4096],
    })

    config = split_config
    print(config)
    wandb_mode = "online" if not dev_mode else "disabled"
    with wandb.init(project=project_name, config=config, mode=wandb_mode) as run:
        trainer = create_trainer(project_name, run.name, **wandb.config)
        try:
            trainer.run_training()
        finally:
            trainer.env_manager.close()

def create_trainer(
    project_name,
    exp_name,

    discount: float,
    adv_lambda: float,
    clip_ratio: float,
    entropy_factor: float,

    policy_imp_min: float,
    policy_imp_upper_target: float,
    policy_imp_max: float,
    policy_valid_imp_lower_bound: float,
    policy_neg_lower_imp_scale_scale: float,

    policy_initial_lr: float,
    policy_update_batch_size: int,
    policy_update_epochs: int,
    policy_lr_decay: float,

    vf_imp_min: float,
    vf_imp_upper_target: float,
    vf_imp_max: float,
    vf_valid_imp_lower_bound: float,

    vf_initial_lr: float,
    vf_update_batch_size: int,
    vf_update_epochs: int,
    vf_lr_decay: float,

    action_scaling: float,
    policy_update_batch_count: int,

    obs_encoder_outputs = 128,
    planner_layers = [4096, 4096],
    planner_skip_connection = True,
    panner_outputs = 4096,
    policy_split_layers = [],
    value_net_split_layers = [],

    lesson_timeout_episodes: int = 80,
    new_lesson_warmup_updates: int = 8,

    epochs: int = 2000,
    resume_lesson: int = None,
    resume_checkpoint: str = None
):
    buffer_size = policy_update_batch_size * policy_update_batch_count
    assert(vf_update_batch_size % vf_update_batch_size == 0)

    save_freq = 100

    file_name = "../env_build/RobotArm.x86_64"
    env_manager= RobotArmEnvManager(file_name, action_scaling = action_scaling)

    # TODO Tell wandb about it
    num_envs = env_manager.env_count

    obs_encoder = RobotArmObsEncoder(
        env_manager,

        obs_outputs= obs_encoder_outputs,
        planner_net_layers = planner_layers,
        planner_skip_connection = planner_skip_connection,
        planner_outputs=panner_outputs
    )

    action_dist_net_output_shape = np.array(env_manager.action_space.shape + (2, ))
    mlp_sizes = list(obs_encoder.shape) + policy_split_layers + [np.prod(action_dist_net_output_shape)]
    action_dist_net_tail = mlp(sizes=mlp_sizes)

    # action mean output of 1 maps to 5 Deg movement
    # action std range is 0.5 - 5.0 Deg
    # keep randomnes to 3 deg for 78.2% of angle changes to

    policy= ContinuousPolicy(
        obs_encoder = obs_encoder,
        action_space = env_manager.action_space,
        action_dist_net_tail = action_dist_net_tail,
        min_std = torch.as_tensor([1.0, 1.0, 1.0, 1.0, 1.0]) / action_scaling,
        max_std = torch.as_tensor([5.0, 5.0, 5.0, 5.0, 5.0]) / action_scaling,
    ).to(cfg.device)

    heuristic= RobotArmHeuristic(
        real_policy = policy,
        action_scaling= action_scaling,
        min_std = torch.as_tensor([1.0, 1.0, 1.0, 1.0, 1.0]) / action_scaling,
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

    value_net_tail = mlp(list(obs_encoder.shape) + value_net_split_layers + [1]).to(cfg.device)
    policy_updater = PPO(
        cfg = cfg,
        policy = policy,
        experience = experience,
        value_net_tail = value_net_tail,

        adv_lambda = adv_lambda,
        clip_ratio = clip_ratio,
        entropy_factor = entropy_factor,

        policy_imp_min=policy_imp_min,
        policy_imp_upper_target=policy_imp_upper_target,
        policy_imp_max=policy_imp_max,
        policy_valid_imp_lower_bound=policy_valid_imp_lower_bound,
        policy_neg_lower_imp_scale_scale=policy_neg_lower_imp_scale_scale,

        policy_initial_lr = policy_initial_lr,
        policy_update_batch_size = policy_update_batch_size,
        policy_update_epochs = policy_update_epochs,
        policy_lr_decay = policy_lr_decay,

        vf_imp_min=vf_imp_min,
        vf_imp_upper_target=vf_imp_upper_target,
        vf_imp_max=vf_imp_max,
        vf_valid_imp_lower_bound=vf_valid_imp_lower_bound,

        vf_initial_lr = vf_initial_lr,
        vf_update_batch_size = vf_update_batch_size,
        vf_update_epochs = vf_update_epochs,
        vf_lr_decay = vf_lr_decay,
    )
    wandb.watch((policy, policy_updater.value_net_tail), log='all', log_freq=20)

    out_dir = f"results/{project_name}/{exp_name}"
    trainer = Trainer(
        cfg = cfg,
        env_manager = env_manager,
        experience = experience,
        policy = heuristic,
        policy_updater = policy_updater,
        lesson_timeout_episodes = lesson_timeout_episodes,
        new_lesson_warmup_updates = new_lesson_warmup_updates,
        epochs = epochs,
        save_freq = save_freq,
        output_dir= out_dir,
        resume_lesson = resume_lesson,
        #eval_func =
    )

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        if os.path.islink(resume_checkpoint):
            print(f"Resuming from {resume_checkpoint}[{os.readlink(resume_checkpoint)}]")
        else:
            print(f"Resuming from {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    else:
        print("****\n**** Starting from scratch !!!\n****")
    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--lesson', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    seed = 45632
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    import matplotlib
    matplotlib.use('Agg')

    run(args.dev, args.lesson, args.resume)
