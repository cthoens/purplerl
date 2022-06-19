import itertools
from random import gammavariate
from xml.dom import InvalidStateErr
import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

from purplerl.config import tensor_args
from purplerl.sync_experience_buffer import ExperienceBufferBase, discount_cumsum


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1], False)]
        layers += [activation() if j < len(sizes)-2 else output_activation()]

    return nn.Sequential(*layers)


class StochasticPolicy(nn.Module):
    def action_dist(self, obs) -> torch.tensor:
        raise NotImplemented

    def act(self, obs) -> torch.tensor:
        return self.action_dist(obs).sample()


class CategoricalPolicy(StochasticPolicy):
    def __init__(self,
        obs_encoder: torch.nn.Sequential,
        hidden_sizes: list[int],
        action_space: gym.Space
    ) -> None:
        super().__init__()

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

        self.obs_encoder = obs_encoder
        self.logits_net = nn.Sequential(
            obs_encoder,
            mlp(sizes = list(obs_encoder.shape) + hidden_sizes + [np.prod(self.distribution_input_shape)] ),
        ).cuda()

    # make function to compute action distribution
    def action_dist(self, obs):
        logits = self.logits_net(obs)
        logits = logits.reshape(list(logits.shape[:-1]) + self.distribution_input_shape)
        return Categorical(logits=logits)

class ContinuousPolicy(StochasticPolicy):
    def __init__(self,
        obs_encoder: torch.nn.Module,
        hidden_sizes: list[int],
        action_space: list[int]
    ) -> None:
        super().__init__()
        self.mean_net_output_shape = action_space.shape
        self.action_shape = action_space.shape
        self.obs_encoder = obs_encoder
        self.mean_net = nn.Sequential(
            obs_encoder,
            mlp(sizes=list(obs_encoder.shape) + hidden_sizes + list(self.mean_net_output_shape))
        ).cuda()
        log_std_init = -0.5 * torch.ones(*self.mean_net_output_shape, **tensor_args)
        self.register_parameter("log_std", torch.nn.Parameter(log_std_init, requires_grad=True).cuda())

    # make function to compute action distaction_spaceaction_spaceribution
    def action_dist(self, obs):
        dist_mean = self.mean_net(obs)
        dist_std = torch.exp(self.log_std)
        return Normal(loc=dist_mean, scale=dist_std)


class PolicyUpdater:
    def __init__(self,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr: float = 1e-2,
        policy_epochs: int = 3
    ) -> None:
        self.policy = policy
        self.experience = experience
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.policy_loss = None
        self.policy_epochs = policy_epochs

        self.weight = torch.zeros(experience.batch_size, experience.buffer_size, **tensor_args)

    def step(self):
        """
        Called after a new environment step was added to the experience buffer. Called before end_episode().
        """
        pass

    def log(self, logger):
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

    def update(self):
        for _ in range(self.policy_epochs):
            self.policy_optimizer.zero_grad()
            self.policy_loss = self._policy_loss(obs=self.experience.obs, act=self.experience.action, weights=self.weight)
            self.policy_loss.backward()
            self.policy_optimizer.step()

    def value_estimate(self, obs):
        return self.value_net(obs).squeeze(-1)

    # make loss function whose gradient, for the right data, is policy gradient
    def _policy_loss(self, obs, act, weights):
        logp = self.policy.action_dist(obs).log_prob(act).sum(-1)
        return -(logp * weights).mean()

    def get_stats(self):
        return {
            "P-Loss": self.policy_loss.item()
        }


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
    def __init__(self,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr=1e-2,
        value_net_lr = 1e-2,
        lam: float = 0.95
    ) -> None:
        super().__init__(policy, experience, policy_lr)
        self.lam = lam
        self.value_net = self.mean_net = nn.Sequential(
            policy.obs_encoder,
            mlp(list(policy.obs_encoder.shape) + hidden_sizes + [1])
        ).cuda()
        self.value_optimizier = torch.optim.Adam(self.value_net.parameters(), lr = value_net_lr)
        self.value_loss_function = torch.nn.MSELoss()
        self.value_loss = 0.0

    def update(self):
        super().update()

        for i in range(10):
            self.value_optimizier.zero_grad()
            self.value_loss = self._value_loss()
            self.value_loss.backward()
            self.value_optimizier.step()


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

    def get_stats(self):
        stats = super().get_stats()
        stats["V-Loss"] = self.value_loss.item()
        return stats

    def get_checkpoint_dict(self):
        return {            
            'value_net_state_dict': self.value_net.state_dict(),
            'value_net_optimizer_state_dict': self.value_optimizier.state_dict(),
            'loss': self.value_loss,
        }

    def from_checkpoint(self, checkpoint):
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_optimizier.load_state_dict(checkpoint['value_net_optimizer_state_dict'])
        self.value_loss = checkpoint['loss']         
        