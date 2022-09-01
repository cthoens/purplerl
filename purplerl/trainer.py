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
from purplerl.sync_experience_buffer import ExperienceBuffer
from purplerl.policy import StochasticPolicy
from purplerl.policy import PolicyUpdater


class Trainer:
    EXPERIENCE = "Sample Trajectories"
    POLICY = "Policy"
    TRAINER = "Trainer"
    ENTROPY = "Entropy"
    LESSON = "lesson"
    LESSON_START_EPOCH = "lesson_start_epoch"
    EPOCH = "epoch"

    def __init__(self,
        cfg,
        env_manager: EnvManager,
        experience: ExperienceBuffer,
        policy: StochasticPolicy,
        policy_updater: PolicyUpdater,
        epochs=50,
        save_freq=20,
        output_dir="",
        eval_func=None
    ) -> None:
        self.cfg = cfg
        self.env_manager = env_manager
        self.experience = experience
        self.policy = policy
        self.policy_updater = policy_updater
        self.epochs=epochs
        self.epoch=0
        self.save_freq=save_freq
        self.lesson = 0
        self.lesson_start_epoch = 0
        self.resume_epoch = 1 # have epochs start at 1 and not 0
        self.eval_func=eval_func
        self.output_dir = output_dir
        self.experience_duration = None
        self.update_duration = None
        self.own_stats = {
            self.ENTROPY: -1.0,
            self.LESSON: 0
        }
        self.all_stats = {
            self.EXPERIENCE: self.experience.stats,
            self.POLICY: self.policy_updater.stats,
            self.TRAINER: self.own_stats
        }


    def run_training(self):
        max_disc_reward = float('-inf')
        max_disc_reward_epoch = 0
        for self.epoch in range(self.epochs):
            self.run_epoch()
            wandb.log(copy.deepcopy(self.all_stats), step=self.epoch)
            self.log_to_console()


            epoch_disc_reward = self.experience.mean_disc_reward()
            lesson_warmup_phase = self.epoch - self.lesson_start_epoch <= 30
            if lesson_warmup_phase:
                 max_disc_reward_epoch = self.epoch + 1
                 max_disc_reward = float('-inf')
            elif epoch_disc_reward > max_disc_reward:
                 max_disc_reward = epoch_disc_reward
                 max_disc_reward_epoch = self.epoch

            if self.epoch - max_disc_reward_epoch > 100:
                 self.lesson += 1
                 self.lesson_start_epoch = self.epoch + 1
                 max_disc_reward = float('-inf')
                 max_disc_reward_epoch = self.epoch + 1
                 self.own_stats[self.LESSON] = self.lesson
                 self.save_checkpoint(f"lesson {self.lesson}.pt")

                 has_more_lessons = self.env_manager.set_lesson(self.lesson)
                 if has_more_lessons:
                     print(f"******> Starting lesson {self.lesson}")
                 else:
                     print(f"Training completed")
                     return

    def run_epoch(self):
        self.epoch += self.resume_epoch
        self.policy_updater.reset()

        # collect experience
        experience_start_time = time.time()
        self._collect_experience()
        self.experience_duration = time.time() - experience_start_time

        # train
        update_start_time = time.time()
        self.policy_updater.update()
        self.update_duration = time.time() - update_start_time

        if (self.epoch > 0 and  self.epoch % self.save_freq == 0) or (self.epoch == self.epochs):
            self.save_checkpoint()


    def log_to_console(self):
        log_str = ""
        for _, stats in self.all_stats.items():
            for name, value in stats.items():
                if type(value) == float:
                    log_str += f"{name}: {value:.4f}; "
                else:
                    log_str += f"{name}: {value}; "

        eval_start_time = time.time()
        if self.eval_func is not None:
            plot = self.eval_func(self.epoch, self.policy_updater)
            if plot:
                wandb.log({"chart": plot})
                plot.close()
        eval_duration = time.time() - eval_start_time

        print(f"Epoch: {self.epoch:3}; L: {self.lesson}; {log_str} Exp time: {self.experience_duration:.1f}; Update time: {self.update_duration:.1f}; Eval time: {eval_duration:.1f}")


    def _collect_experience(self):
        self.experience.reset()
        action_mean_entropy = torch.empty(self.experience.buffer_size, self.experience.num_envs, dtype=torch.float32)
        self.policy.requires_grad_(False)
        self.policy_updater.value_net_tail.requires_grad_(False)
        try:
            obs = torch.as_tensor(self.env_manager.reset(), **self.cfg.tensor_args)
            for step, _ in enumerate(range(self.experience.buffer_size)):
                encoded_obs = self.policy.obs_encoder(obs)
                act_dist = self.policy.action_dist(encoded_obs = encoded_obs)
                act = act_dist.sample()
                action_mean_entropy[step, :] = act_dist.entropy().mean(-1)
                next_obs, rew, done, success = self.env_manager.step(act.cpu().numpy())
                next_obs = torch.as_tensor(next_obs, **self.cfg.tensor_args)

                self.experience.step(obs, act, rew)
                self.policy_updater.step()
                self.policy_updater.end_episode(done)
                self.experience.end_episode(done, success)
                obs = next_obs

            encoded_obs = self.policy.obs_encoder(obs)
            last_obs_value_estimate = self.policy_updater.value_estimate(encoded_obs = encoded_obs).cpu().numpy()
            self.experience.buffer_full(last_obs_value_estimate)
            self.policy_updater.buffer_full(last_obs_value_estimate)

            self.own_stats[self.ENTROPY] = action_mean_entropy.mean().item()

        finally:
            self.policy.requires_grad_(True)
            self.policy_updater.value_net_tail.requires_grad_(True)


    def save_checkpoint(self, fname = None):
        full_state = {
            "policy": self.policy.checkpoint(),
            "policy_updater": self.policy_updater.checkpoint(),
            "trainer": self.checkpoint()
        }

        fpath = self.output_dir
        if not fname:
            fname = f"checkpoint{self.epoch}.pt"
        os.makedirs(fpath, exist_ok=True)
        torch.save(full_state, osp.join(fpath, fname))

        link_name = f"resume.pt"
        link_fname = osp.join(fpath, link_name)
        pathlib.Path(link_fname).unlink(missing_ok=True)
        os.symlink(dst=link_fname, src=fname)

    def load_checkpoint(self, file):
        checkpoint = torch.load(file)
        self.policy.load_checkpoint(checkpoint["policy"])
        self.policy_updater.load_checkpoint(checkpoint["policy_updater"])
        trainer_state = checkpoint["trainer"]
        self.resume_epoch = trainer_state.get(self.EPOCH, 0)+1

        self.lesson = trainer_state.get(self.LESSON, 0)
        self.lesson_start_epoch = trainer_state.get(self.LESSON_START_EPOCH, 0)
        self.env_manager.set_lesson(self.lesson)
        self.own_stats[self.LESSON] = self.lesson

    def checkpoint(self):
        state_dict = {
            self.EPOCH: self.epoch,
            self.LESSON: self.lesson,
            self.LESSON_START_EPOCH: self.lesson_start_epoch
        }
        return state_dict