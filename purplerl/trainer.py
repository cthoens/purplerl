import copy
import time
import joblib
import os
import os.path as osp
import pathlib
import warnings

import numpy as np
import torch
import wandb
from purplerl.environment import EnvManager
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
        output_dir="",
        state_dict={}
    ) -> None:
        self.env_manager = env_manager
        self.experience = experience
        self.policy = policy
        self.policy_updater = policy_updater
        self.epochs=epochs
        self.epoch=0
        self.save_freq=save_freq
        self.lesson = state_dict.get(self.LESSON, 0)
        self.resume_epoch = state_dict.get(self.EPOCH, 0)+1
        self.output_dir = output_dir
        os.makedirs(output_dir)
        self.own_stats = {
            self.ENTROPY: -1.0 
        }
        self.all_stats = {
            self.EXPERIENCE: self.experience.stats,
            self.POLICY: self.policy_updater.stats,
            self.TRAINER: self.own_stats
        }

    def save_state(self, state_dict, itr):
        fname = 'vars.pkl' if itr is None else f'vars{itr}.pkl'
        joblib.dump(state_dict, osp.join(self.output_dir, fname))
        link_fname = osp.join(self.output_dir, "resume.pkl")
        pathlib.Path(link_fname).unlink(missing_ok=True)
        os.symlink(dst=link_fname, src=fname)
        
        fpath = self.output_dir
        fname = f"model{itr}.pt"
        os.makedirs(fpath, exist_ok=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.save(self.policy, osp.join(fpath, fname))
        link_path = self.output_dir
        link_name = f"resume.pt"
        link_fname = osp.join(link_path, link_name)
        pathlib.Path(link_fname).unlink(missing_ok=True)
        os.symlink(dst=link_fname, src=fname)

    def save_checkpoint(self, name = None):
            if not name:
                name = self.epoch
            state_dict = {
                self.EPOCH: self.epoch,
                self.LESSON: self.lesson,
            }
            self.save_state(state_dict | self.policy_updater.get_checkpoint_dict(), itr = name)

    def run_training(self):
        for self.epoch in range(self.epochs):
            self.epoch += self.resume_epoch
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

            if (self.epoch > 0 and  self.epoch % self.save_freq == 0) or (self.epoch == self.epochs-1):
                self.save_checkpoint()

            log_str = ""
            for _, stats in self.all_stats.items():
                for name, value in stats.items():                
                    if type(value) == float:
                        log_str += f"{name}: {value:.4f}; "
                    else: 
                        log_str += f"{name}: {value}; "
            
            print(f"Epoch: {self.epoch:3}; L: {self.lesson}; {log_str}")
            wandb.log(copy.deepcopy(self.all_stats), step=self.epoch)

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