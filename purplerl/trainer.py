import copy
from email import policy
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
from purplerl.experience_buffer import ExperienceBuffer
from purplerl.policy import StochasticPolicy
from purplerl.policy import PPO


class Trainer:
    EXPERIENCE = "Sample Trajectories"
    ENVIRONMENT = "Env"
    POLICY = "Policy"
    TRAINER = "Trainer"
    TIMING = "Timing"
    ENTROPY = "Entropy"
    MEAN_RETURN_EMA = "Mean Return EMA"
    LESSON_TIMEOUT = "Lesson Timeout"
    ENV_TIME = "Env Time"
    DECISION_TIME = "Decision Time"
    EXPERIENCE_TIME = "Experience Time"
    UPDATE_TIME = "Update Time"
    EVAL_TIME = "Eval Time"
    LESSON = "lesson"
    LESSON_START_EPOCH = "lesson_start_epoch"
    EPOCH = "epoch"

    def __init__(self,
        cfg,
        env_manager: EnvManager,
        experience: ExperienceBuffer,
        policy: StochasticPolicy,
        policy_updater: PPO,
        new_lesson_warmup_updates: int,
        lesson_timeout_episodes: int = 100,
        resume_lesson: int = None,
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
        self.lesson_timeout_episodes = lesson_timeout_episodes
        self.resume_lesson = resume_lesson
        self.new_lesson_warmup_updates = new_lesson_warmup_updates
        self.resume_epoch = 1 # have epochs start at 1 and not 0
        self.eval_func=eval_func
        self.output_dir = output_dir
        self.mean_return_ema = 0.0
        self.max_mean_return = float('-inf')
        self.max_mean_return_epoch = 0
        self.own_stats = {
            self.ENTROPY: -1.0,
            self.LESSON: 0,
            self.MEAN_RETURN_EMA: 0,
        }
        self.timig_stats = {
            self.ENV_TIME: 0.0,
            self.DECISION_TIME: 0.0,
            self.EXPERIENCE_TIME: 0.0,
            self.UPDATE_TIME: 0.0,
            self.EVAL_TIME: 0.0
        }
        self.all_stats = {
            self.ENVIRONMENT: self.env_manager.stats,
            self.EXPERIENCE: self.experience.stats,
            self.POLICY: self.policy_updater.stats,
            self.TRAINER: self.own_stats,
            self.TIMING: self.timig_stats
        }
        self.console_stats = {
            self.EXPERIENCE: self.experience.stats,
            self.POLICY: self.policy_updater.stats
        }


    def run_training(self):
        for self.epoch in range(self.epochs):
            self.run_epoch()

            if self.epoch<=1:
                self.mean_return_ema = self.experience.mean_return()
            else:
                self.mean_return_ema = 0.6 * self.experience.mean_return() + 0.4 * self.mean_return_ema
            self.own_stats[self.MEAN_RETURN_EMA] = self.mean_return_ema
            lesson_warmup_phase = self.epoch - self.lesson_start_epoch <= self.new_lesson_warmup_updates
            if lesson_warmup_phase:
                 self.max_mean_return_epoch = self.epoch + 1
                 self.max_mean_return = float('-inf')
            elif self.mean_return_ema > self.max_mean_return:
                 self.max_mean_return = self.mean_return_ema
                 self.max_mean_return_epoch = self.epoch

            lesson_timeout = self.epoch - self.max_mean_return_epoch
            self.own_stats[self.LESSON_TIMEOUT] = min((self.lesson_timeout_episodes - lesson_timeout) / self.lesson_timeout_episodes, 1.0)
            if lesson_timeout > self.lesson_timeout_episodes or (self.experience.success_rate() > 0.99 and lesson_timeout > 30):
                self.start_lesson(self.lesson + 1)

            wandb.log(copy.deepcopy(self.all_stats), step=self.epoch)
            self.log_to_console()

    def start_lesson(self, lesson: int):
        self.lesson = lesson
        self.lesson_start_epoch = self.epoch + 1
        self.policy_updater.remaining_warmup_updates = self.new_lesson_warmup_updates
        self.max_mean_return = float('-inf')
        self.max_mean_return_epoch = self.epoch + 1
        self.own_stats[self.LESSON] = self.lesson
        self.save_checkpoint(f"lesson {self.lesson}.pt")

        has_more_lessons = self.env_manager.set_lesson(self.lesson)
        if has_more_lessons:
            print(f"******> Starting lesson {self.lesson}")
        else:
            print(f"Training completed")
            return

    def run_epoch(self):
        self.timig_stats.update({
            self.ENV_TIME: 0.0,
            self.DECISION_TIME: 0.0,
            self.EXPERIENCE_TIME: 0.0,
            self.UPDATE_TIME: 0.0,
            self.EVAL_TIME: 0.0
        })

        self.epoch += self.resume_epoch
        self.policy_updater.reset()

        # collect experience
        self._collect_experience()

        # train
        update_start_time = time.time()
        self.policy_updater.update()
        self.timig_stats[self.UPDATE_TIME] = time.time() - update_start_time

        if (self.epoch > 0 and  self.epoch % self.save_freq == 0) or (self.epoch == self.epochs):
            self.save_checkpoint()


    def log_to_console(self):
        eval_start_time = time.time()
        if self.eval_func is not None:
            plot = self.eval_func(self.epoch, self.lesson, self.policy_updater)
            if plot:
                wandb.log({"chart": plot})
                plot.close()
        self.timig_stats[self.EVAL_TIME] = time.time() - eval_start_time

        log_str = ""
        for _, stats in self.console_stats.items():
            for name, value in stats.items():
                if type(value) == float:
                    log_str += f"{name}: {value:.3f}; "
                else:
                    log_str += f"{name}: {value}; "

        print(f"Epoch: {self.epoch:3}; L: {self.lesson}; {log_str}")


    def _collect_experience(self):
        env_time = time.time()
        self.experience.reset()
        self.timig_stats[self.ENV_TIME] += time.time() - env_time

        num_actions = np.prod(self.env_manager.action_space.shape)
        action_mean_entropy = torch.empty(self.experience.buffer_size, self.experience.num_envs, num_actions, dtype=torch.float32)
        self.policy.requires_grad_(False)
        self.policy_updater.value_net_tail.requires_grad_(False)
        try:
            obs = torch.as_tensor(self.env_manager.reset(), **self.cfg.tensor_args)
            for step, _ in enumerate(range(self.experience.buffer_size)):
                decision_time = time.time()
                encoded_obs = self.policy.obs_encoder(obs)
                act_dist = self.policy.action_dist(encoded_obs = encoded_obs)
                act = act_dist.sample()
                action_mean_entropy[step, ...] = act_dist.entropy()
                self.timig_stats[self.DECISION_TIME] += time.time() - decision_time

                env_time = time.time()
                next_obs, rew, done, success = self.env_manager.step(act)
                next_obs = torch.as_tensor(next_obs, **self.cfg.tensor_args)
                self.timig_stats[self.ENV_TIME] += time.time() - env_time

                experience_time = time.time()
                # Note: Always call pocliy_updater first so it can use experience.next_step_index
                self.policy_updater.step(encoded_obs)
                self.experience.step(obs, act, rew)
                self.policy_updater.end_episode(done)
                self.experience.end_episode(done, success)
                self.timig_stats[self.EXPERIENCE_TIME] += time.time() - experience_time
                obs = next_obs

            experience_time = time.time()
            encoded_obs = self.policy.obs_encoder(obs)
            last_obs_value_estimate = self.policy_updater.value_estimate(encoded_obs = encoded_obs).cpu().numpy()
            self.experience.buffer_full(last_obs_value_estimate)
            self.policy_updater.buffer_full(last_obs_value_estimate)
            self.timig_stats[self.EXPERIENCE_TIME] += time.time() - experience_time

            self.env_manager.update_obs_stats(self.experience)

            self.own_stats[self.ENTROPY] = action_mean_entropy.mean().item()
            for i in range(num_actions):
                self.own_stats[f"Action-{i}"] = self.experience.action[...,i].mean().item()
                self.own_stats[f"Act-Std-{i}"] = self.experience.action[...,i].std().item()
                self.own_stats[f"Act-Max-{i}"] = torch.abs(self.experience.action[...,i]).max().item()
                self.own_stats[f"{self.ENTROPY}-A{i}"] = action_mean_entropy[...,i].mean().item()

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

        if self.resume_lesson is None:
            self.lesson = trainer_state.get(self.LESSON, 0)
            self.lesson_start_epoch = trainer_state.get(self.LESSON_START_EPOCH, 0)
            self.env_manager.set_lesson(self.lesson)
            self.own_stats[self.LESSON] = self.lesson
        else:
            self.start_lesson(self.resume_lesson)


    def checkpoint(self):
        state_dict = {
            self.EPOCH: self.epoch,
            self.LESSON: self.lesson,
            self.LESSON_START_EPOCH: self.lesson_start_epoch
        }
        return state_dict