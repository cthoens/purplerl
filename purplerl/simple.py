import math
import time
from enum import Enum

import torch

from gym.spaces import Discrete, MultiDiscrete, Box

from purplerl.spinup.utils.logx import EpochLogger
from purplerl.environment import EnvManager, GymEnvManager, IdentityObsEncoder
from purplerl.sync_experience_buffer import ExperienceBufferBase, MonoObsExperienceBuffer
from purplerl.policy import StochasticPolicy, CategoricalPolicy, ContinuousPolicy
from purplerl.policy import PolicyUpdater, Vanilla, RewardToGo, ValueFunction


class Algo(Enum):
    VANILLA = 1,
    REWARD_TO_GO = 2,
    VALUE_NET = 3


class Trainer:
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
            policy_updater = Vanilla(policy, experience, policy_lr)
        elif algo == Algo.VALUE_NET:
            policy_updater = ValueFunction(obs_shape, hidden_sizes, policy, experience, policy_lr, value_net_lr)
        else:
            raise Exception("Unknown algorithm")

        self.run_training(env_manager, experience, policy, policy_updater, epochs, save_freq)

    def run_training(self,
            env_manager: EnvManager,
            experience: ExperienceBufferBase,
            policy: StochasticPolicy,
            policy_updater: PolicyUpdater,
            epochs=50,
            save_freq=20,
        ):
        self.logger.setup_pytorch_saver(policy)

        # collect experience
        for epoch in range(epochs):
            epoch_start_time = time.time()
            experience.reset()
            policy_updater.reset()
            action_mean_entropy = torch.empty(experience.batch_size, experience.buffer_size, dtype=torch.float32)
            with torch.no_grad():
                obs = env_manager.reset()
                for step, _ in enumerate(range(experience.buffer_size)):
                    act_dist = policy.action_dist(obs)
                    act = act_dist.sample()
                    action_mean_entropy[:, step] = act_dist.entropy().mean(-1)
                    next_obs, rew, done, success = env_manager.step(act)

                    experience.step(obs, act, rew)
                    policy_updater.step()
                    policy_updater.end_episode(done)
                    experience.end_episode(done, success)
                    obs = next_obs

                mean_return = experience.mean_return()
                success_rate = experience.success_rate()
                mean_entropy = action_mean_entropy.mean()

            # train
            policy_updater.update()
            new_lr = 1e-4 *  math.exp(-3.0 * success_rate)
            for g in policy_updater.policy_optimizer.param_groups:
                g['lr'] = new_lr

            if (epoch % save_freq == 0) or (epoch == epochs-1):
                self.logger.save_state({}, itr = epoch)

            #print(f"Epoch: {epoch}; P-Loss: {policy_updater.policy_loss:.4f}; V-Loss: {policy_updater.value_loss:.4f}; Entropy: {mean_entropy.item():.4f}; Mean reward: {mean_return:.2f}; Episode count: {experience.ep_count}")
            #print(f"Epoch: {epoch}; P-Loss: {policy_updater.policy_loss:.4f}; Mean reward: {mean_return:.4f}; Episode count: {experience.ep_count}; Success: {env_manager.get_success_rate():0.4f}; Entropy: {mean_entropy.item():.4f}")
            print(f"Epoch: {epoch}; P-Loss: {policy_updater.policy_loss:.4f}; Mean reward: {mean_return:.4f}; Success: {success_rate*100:.0f}%; Episode count: {experience.ep_count}; LR: {new_lr:e}")
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpochEpisodeCount', experience.ep_count)
            self.logger.log_tabular('Time', time.time()-epoch_start_time)
            self.logger.log_tabular('AverageEpRet', mean_return.item())
            self.logger.log_tabular('AverageEntropy', mean_entropy.item())
            policy_updater.log(self.logger)
            self.logger.dump_tabular()


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

    from purplerl.spinup.utils.run_utils import setup_logger_kwargs

    for i in range(6):
        seed = i*500

        logger_kwargs = setup_logger_kwargs(args.exp_name, seed, data_dir=f"./{args.env_name}")
        trainer = Trainer(logger_kwargs)

        trainer.train(env=args.env_name, algo=args.algo, epochs=args.epochs, policy_lr=args.lr, value_net_lr=args.value_lr, seed=seed)

if __name__ == '__main__':
    run()
