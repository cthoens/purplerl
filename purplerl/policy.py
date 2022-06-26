import itertools
from random import gammavariate
from typing import Callable
from xml.dom import InvalidStateErr
import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

from purplerl.config import device, tensor_args
from purplerl.sync_experience_buffer import ExperienceBufferBase, discount_cumsum


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1], False)]
        layers += [activation() if j < len(sizes)-2 else output_activation()]

    return nn.Sequential(*layers)


class StochasticPolicy(nn.Module):
    def __init__(self, obs_encoder: torch.nn.Sequential):
        super().__init__()
        self.obs_encoder = obs_encoder

    def action_dist(self, obs) -> torch.tensor:
        raise NotImplemented

    def act(self, obs) -> torch.tensor:
        return self.action_dist(obs).sample()

    def checkpoint(self):
        return {}


class CategoricalPolicy(StochasticPolicy):
    def __init__(self,
        obs_encoder: torch.nn.Sequential,
        hidden_sizes: list[int],
        action_space: gym.Space
    ) -> None:
        super().__init__(obs_encoder)

        if isinstance(action_space, Discrete):
            # A discrete action shpace has one action that can take n possible values
            self.distribution_input_shape = [1, action_space.n]
            self.action_shape = [1]
        elif isinstance(action_space, MultiDiscrete):
            # A discrete action shapce has one action that can take n possible values
            nvec = action_space.nvec
            assert np.all(nvec == nvec[0]), \
                "MuliDiscrete action spaces are only supported if all action have the same number of options"
            n_actions = len(nvec)
            n_options = nvec[0]
            self.distribution_input_shape =  [n_actions, n_options]
            self.action_shape = [n_actions]
        else:
            raise Exception("Unsupported action space")

        self.logits_net = nn.Sequential(
            self.obs_encoder,
            mlp(sizes = list(obs_encoder.shape) + hidden_sizes + [np.prod(self.distribution_input_shape)] ),
        ).to(device)


    # make function to compute action distribution
    def action_dist(self, obs):
        logits = self.logits_net(obs)
        logits = logits.reshape(list(logits.shape[:-1]) + self.distribution_input_shape)
        return Categorical(logits=logits)


    def checkpoint(self):
        return super().get_checkpoint_dict() | {
            'logits_net_state_dict': self.logits_net.state_dict(),
        }


    def load_checkpoint(self, checkpoint):
        self.logits_net.load_state_dict(checkpoint['logits_net_state_dict'])

class ContinuousPolicy(StochasticPolicy):
    def __init__(self,
        obs_encoder: torch.nn.Module,
        hidden_sizes: list[int],
        action_space: list[int]
    ) -> None:
        super().__init__(obs_encoder)
        self.mean_net_output_shape = action_space.shape
        self.action_shape = action_space.shape
        self.mean_net = nn.Sequential(
            self.obs_encoder,
            mlp(sizes=list(obs_encoder.shape) + hidden_sizes + list(self.mean_net_output_shape))
        )
        log_std_init = -0.5 * torch.ones(*self.mean_net_output_shape, **tensor_args)
        self.log_std = torch.nn.Parameter(log_std_init)


    # make function to compute action distaction_spaceaction_spaceribution
    def action_dist(self, obs):
        dist_mean = self.mean_net(obs)
        dist_std = torch.exp(self.log_std)
        return Normal(loc=dist_mean, scale=dist_std)


    def checkpoint(self):
        return super().state_dict() | {
            'mean_net_state_dict': self.mean_net.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.mean_net.load_state_dict(checkpoint['mean_net_state_dict'])
        self.log_std.data = checkpoint['log_std']



class PolicyUpdater:
    POLICY_LOSS = "Policy Loss"
    POLICY_LR = "Policy LR"

    def __init__(self,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs: int = 3
    ) -> None:
        self.policy = policy
        self.experience = experience
        self.policy_lr_scheduler = policy_lr_scheduler
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr_scheduler())
        self.policy_epochs = policy_epochs
        self.stats = {}

        self.weight = torch.zeros(experience.batch_size, experience.buffer_size, **tensor_args)


    def step(self):
        """
        Called after a new environment step was added to the experience buffer. Called before end_episode().
        """
        pass


    def end_episode(self, finished_batches: torch.Tensor):
        for batch, finished in enumerate(finished_batches):
            if not finished:
                continue

            if self.experience.next_step_index==self.experience.ep_start_index[batch]:
                raise InvalidStateErr("policy_updater.end_episode must be called before experience_buffer.end_episode!")

            self._end_episode_batch(batch)

    def _end_episode_batch(self, batch: int):
        pass

    def buffer_full(self, last_state_value_estimate: torch.tensor):
        assert(self.experience.next_step_index == self.experience.buffer_size)
        for batch in range(self.experience.batch_size):
            #  there is nothing to do if this batch just finished an episode
            if self.experience.ep_start_index[batch] == self.experience.next_step_index:
                continue

            self._finish_path_batch(batch, last_state_value_estimate[batch])

    def _finish_path_batch(self, batch, last_state_value_estimate: float):
        pass

    def reset(self):
        self.weight[...] = 0.0
        
        for stat in self.stats.keys():
            self.stats[stat] = None

    def update(self):
        new_policy_lr = self.policy_lr_scheduler()
        for g in self.policy_optimizer.param_groups:
            g['lr'] = new_policy_lr
        self.stats[self.POLICY_LR] = new_policy_lr

        for _ in range(self.policy_epochs):
            self.policy_optimizer.zero_grad()
            policy_loss = self._policy_loss()
            policy_loss.backward()
            self.policy_optimizer.step()
        self.stats[self.POLICY_LOSS] = policy_loss.item()

    def value_estimate(self, obs):
        raise NotImplemented()

    # make loss function whose gradient, for the right data, is policy gradient
    def _policy_loss(self):
        obs=self.experience.obs 
        act=self.experience.action 
        weights=self.weight
        logp = self.policy.action_dist(obs).log_prob(act).sum(-1)
        return -(logp * weights).mean()

    def checkpoint(self):
        return {
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])


class RewardToGo(PolicyUpdater):
    def __init__(self, 
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr: float = 1e-2,
        policy_epochs: int = 3        
    ) -> None:
        super().__init__(policy, experience, policy_lr, policy_epochs)
        self.weight = self.experience.discounted_reward


class Vanilla(PolicyUpdater):
    VALUE_FUNCTION_LOSS = "VF Loss"
    VALUE_FUNCTION_LR = "VF LR"
    
    def __init__(self,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs,
        vf_lr_scheduler: Callable[[], float],
        vf_epochs,
        lam: float = 0.95
    ) -> None:
        super().__init__(policy, experience, policy_lr_scheduler, policy_epochs=policy_epochs)
        self.lam = lam
        self.vf_lr_scheduler = vf_lr_scheduler
        self.vf_epochs = vf_epochs
        self.value_net = self.mean_net = nn.Sequential(
            policy.obs_encoder,
            mlp(list(policy.obs_encoder.shape) + hidden_sizes + [1])
        ).to(device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr = vf_lr_scheduler())
        self.value_loss_function = torch.nn.MSELoss()


    def update(self):
        super().update()

        new_value_net_lr = self.vf_lr_scheduler()
        for g in self.value_optimizer.param_groups:
            g['lr'] = new_value_net_lr
        self.stats[self.VALUE_FUNCTION_LR] = new_value_net_lr

        for p in self.policy.obs_encoder.parameters():
            p.requires_grad=False
        
        try:
            for i in range(self.vf_epochs):
                self.value_optimizer.zero_grad()
                self.value_loss = self._value_loss()
                self.value_loss.backward()
                self.value_optimizer.step()
            self.stats[self.VALUE_FUNCTION_LOSS] = self.value_loss.item()
        finally:
            for p in self.policy.obs_encoder.parameters():
                p.requires_grad=True

    def value_estimate(self, obs):
        return self.value_net(obs).squeeze(-1)

    def _value_loss(self):
        return self.value_loss_function(self.value_net(self.experience.obs).squeeze(-1), self.experience.discounted_reward)


    def _end_episode_batch(self, batch: int):
        self._finish_path_batch(batch)


    def _finish_path_batch(self, batch, last_state_value_estimate: float = None):        
        reached_terminal_state = last_state_value_estimate is None
        path_slice = range(self.experience.ep_start_index[batch], self.experience.next_step_index)
        rews = self.experience.step_reward[batch][path_slice]
        
        vals = np.zeros(len(path_slice)+1, dtype=np.float32)
        vals[:-1] = self.value_net(self.experience.obs[batch][path_slice]).squeeze(-1).cpu().numpy()
        vals[-1] = last_state_value_estimate if not reached_terminal_state else 0.0
                
        # estimated reward for state transition = estimated value of next state - estimated value of current state
        next_state_value = vals[1:]
        current_state_value = vals[:-1]
        expected_rewards = self.experience.discount * next_state_value - current_state_value
        # if the episode has reached a terminal state the last state has no next state. In this case use the value 
        # of the last state as value estimate of the terminal state
        if reached_terminal_state: expected_rewards[-1] = current_state_value[-1]
        deltas = rews + expected_rewards
        weight = discount_cumsum(deltas, self.experience.discount * self.lam)
        self.weight[batch][path_slice] = torch.tensor(np.array(weight), **tensor_args)


    def checkpoint(self):
        return super().checkpoint() | {            
            'value_net_state_dict': self.value_net.state_dict(),
            'value_net_optimizer_state_dict': self.value_optimizer.state_dict(),
            'value_net_loss': self.value_loss,
        }


    def load_checkpoint(self, checkpoint):
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_net_optimizer_state_dict'])
        self.value_loss = checkpoint['value_net_loss']


class PPO(Vanilla):    
    KL = "KL"
    ENTROPY = "Entropy"
    MAX_KL_REACHED = "Max KL Reached"
    CLIP_FACTOR = "Clip Factor"
    
    def __init__(self,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs,
        vf_lr_scheduler: Callable[[], float],
        vf_epochs,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
    ) -> None:
        super().__init__(hidden_sizes, policy, experience, policy_lr_scheduler, policy_epochs, vf_lr_scheduler, vf_epochs, lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl


    def update(self):
        # Policy upgrade
        self._update_policy()

        # Value function upgrade
        new_value_net_lr = self.vf_lr_scheduler()
        for g in self.value_optimizer.param_groups:
            g['lr'] = new_value_net_lr
        self.stats[self.VALUE_FUNCTION_LR] = new_value_net_lr

        for p in self.policy.obs_encoder.parameters():
            p.requires_grad=False
        
        try:
            for i in range(self.vf_epochs):
                self.value_optimizer.zero_grad()
                self.value_loss = self._value_loss()
                self.value_loss.backward()
                self.value_optimizer.step()
            self.stats[self.VALUE_FUNCTION_LOSS] = self.value_loss.item()
        finally:
            for p in self.policy.obs_encoder.parameters():
                p.requires_grad=True

    def _update_policy(self):
        new_policy_lr = self.policy_lr_scheduler()
        for g in self.policy_optimizer.param_groups:
            g['lr'] = new_policy_lr
        self.stats[self.POLICY_LR] = new_policy_lr

        obs=self.experience.obs 
        act=self.experience.action
        logp_old = self.policy.action_dist(obs).log_prob(act).sum(-1).detach()
        
        # Train policy with multiple steps of gradient descent
        self.stats[self.MAX_KL_REACHED] = 0.0
        for i in range(self.policy_epochs):
            self.policy_optimizer.zero_grad()
            policy_loss = self._policy_loss(logp_old)
            if self.stats[self.KL] > 1.5 * self.target_kl:
                self.stats[self.MAX_KL_REACHED] = 1.0
                break
            policy_loss.backward()
            self.policy_optimizer.step()
        self.stats[self.POLICY_LOSS] = policy_loss.item()


    def _policy_loss(self, logp_old):
        obs=self.experience.obs 
        act=self.experience.action 
        adv=self.weight

        # Policy loss
        action_dist = self.policy.action_dist(obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * adv

        # Useful extra info
        self.stats[self.KL] = (logp_old - logp).mean().item()
        self.stats[self.ENTROPY] = action_dist.entropy().mean().item()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        self.stats[self.CLIP_FACTOR] = torch.as_tensor(clipped, dtype=torch.float32).mean().item()

        return -(torch.min(ratio * adv, clip_adv)).mean()