import math
import time
from enum import Enum

import numpy as np
import torch

from gym.spaces import Discrete, MultiDiscrete, Box

from purplerl.spinup.utils.logx import EpochLogger
from purplerl.environment import EnvManager, GymEnvManager, IdentityObsEncoder
from purplerl.sync_experience_buffer import ExperienceBufferBase, MonoObsExperienceBuffer
from purplerl.policy import StochasticPolicy, CategoricalPolicy, ContinuousPolicy
from purplerl.policy import PolicyUpdater, RewardToGo, Vanilla
from purplerl.config import gpu


def to_tensor(input: np.ndarray) -> torch.tensor:
        return torch.as_tensor(input, dtype=torch.float32, device=gpu)


class Trainer:
    EXPERIENCE = "Sample Trajectories"
    POLICY = "Policy"
    TRAINER = "Trainer"
    ENTROPY = "Entropy"
    LESSON = "lesson"
    EPOCH = "epoch"

    def __init__(self, 
        env_manager: EnvManager,
        experience: ExperienceBufferBase,
        policy: StochasticPolicy,
        policy_updater: PolicyUpdater,
        epochs=50,
        save_freq=20,
        state_dict={},
        logger_kwargs=dict()
    ) -> None:
        self.env_manager = env_manager
        self.experience = experience
        self.policy = policy
        self.policy_updater = policy_updater
        self.epochs=epochs
        self.save_freq=save_freq
        self.lesson = state_dict.get(self.LESSON, 0)
        self.resume_epoch = state_dict.get(self.EPOCH, 0)
        self.logger = EpochLogger(**logger_kwargs)
        self.own_stats = {
            self.ENTROPY: -1.0 
        }
        self.all_stats = {
            self.EXPERIENCE: self.experience.stats,
            self.POLICY: self.policy_updater.stats,
            self.TRAINER: self.own_stats
        }

    def save_checkpoint(self, name = None):
            if not name:
                name = self.epoch
            state_dict = {
                self.EPOCH: self.epoch,
                self.LESSON: self.lesson,
            }
            self.logger.save_state(state_dict | self.policy_updater.get_checkpoint_dict(), itr = name)

    def run_training(self):
        self.logger.setup_pytorch_saver(self.policy)

        for epoch in range(self.epochs):
            epoch += self.resume_epoch
            epoch_start_time = time.time()
            self.experience.reset()
            self.policy_updater.reset()
            action_mean_entropy = torch.empty(self.experience.batch_size, self.experience.buffer_size, dtype=torch.float32)
            with torch.no_grad():
                obs = to_tensor(self.env_manager.reset())
                for step, _ in enumerate(range(self.experience.buffer_size)):
                    act_dist = self.policy.action_dist(obs)
                    act = act_dist.sample()
                    action_mean_entropy[:, step] = act_dist.entropy().mean(-1)
                    next_obs, rew, done, success = self.env_manager.step(act.cpu().numpy())
                    next_obs = to_tensor(next_obs)

                    self.experience.step(obs, act, rew)
                    self.policy_updater.step()
                    self.policy_updater.end_episode(done)
                    self.experience.end_episode(done, success)
                    obs = next_obs

                last_obs_value_estimate = self.policy_updater.value_estimate(obs).cpu().numpy()
                self.experience.buffer_full(last_obs_value_estimate)
                self.policy_updater.buffer_full(last_obs_value_estimate)
                
                success_rate = self.experience.success_rate()
                self.own_stats[self.ENTROPY] = action_mean_entropy.mean().item()
            
            
            # train            
            self.policy_updater.update()            

            if (epoch > 0 and  epoch % self.save_freq == 0) or (epoch == self.epochs-1):
                self.save_checkpoint()

            self.logger.log_tabular('Epoch', epoch)            
            log_str = ""
            for _, stats in self.all_stats.items():
                for name, value in stats.items():
                    self.logger.log_tabular(name, value)
                
                    if type(value) == float:
                        log_str += f"{name}: {value:.4f}; "
                    else: 
                        log_str += f"{name}: {value}; "
            
            print(f"Epoch: {epoch:3}; L: {self.lesson}; {log_str}")
            self.logger.log_tabular('Time', time.time()-epoch_start_time)
            self.logger.dump_tabular()

            if success_rate == 1.0:
                self.success_count += self.experience.ep_count
                if (self.success_count > 200):
                    self.save_checkpoint(f"lesson"+lesson)
                    
                    lesson += 1
                    has_more_lessons = self.env_manager.set_lesson(lesson)
                    if has_more_lessons:
                        print("Starting next lesson")
                    else:
                        print("Training completed")
                        return
            else:
                self.success_count = 0