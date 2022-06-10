import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete

import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.optim import Adam

from purplerl.config import gpu
from purplerl.sync_experience_buffer import ExperienceBufferBase


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1], False)]
        layers += [activation() if j < len(sizes)-2 else output_activation()]

    return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


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
        log_std_init = -0.5 * np.ones(self.mean_net_output_shape, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std_init)).cuda()

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

        self.weight = torch.zeros(experience.batch_size, experience.buffer_size, dtype=torch.float32, device=gpu)

    def step(self):
        """
        Called after a new environment step was added to the experience buffer. Called before end_episode().
        """
        pass

    def log(self, logger):
        pass

    def end_episode(self, finished_batches: torch.Tensor = None):
        if finished_batches is None:
            finished_batches = np.full(self.batch_size, True)

        for batch, finished in enumerate(finished_batches):
            if not finished:
                continue

            self._end_episode_batch(batch)

    def _end_episode_batch(self, batch: int):
        raise NotImplemented


    def reset(self):
        self.weight[...] = 0.0

    def update(self):
        for _ in range(self.policy_epochs):
            self.policy_optimizer.zero_grad()
            self.policy_loss = self._policy_loss(obs=self.experience.obs, act=self.experience.action, weights=self.weight)
            self.policy_loss.backward()
            self.policy_optimizer.step()

    # make loss function whose gradient, for the right data, is policy gradient
    def _policy_loss(self, obs, act, weights):
        logp = self.policy.action_dist(obs).log_prob(act).sum(-1)
        return -(logp * weights).mean()


class Vanilla(PolicyUpdater):
    def _end_episode_batch(self, batch: int):
        ep_range = range(self.experience.ep_start_index[batch], self.experience.next_step_index)
        self.weight[batch, ep_range] = self.experience.ep_cum_reward[batch]


class RewardToGo(PolicyUpdater):
    def __init__(self, 
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr: float = 1e-2,
        policy_epochs: int = 3,
        gamma: float = 0.99
    ) -> None:
        super().__init__(policy, experience, policy_lr, policy_epochs)
        self.gamma = gamma

    def _end_episode_batch(self, batch: int):
        ep_range = range(self.experience.ep_start_index[batch], self.experience.next_step_index)
        # TODO Fix .cpu().numpy()
        self.weight[batch, ep_range] = torch.as_tensor(np.array(discount_cumsum(self.experience.step_reward[batch, ep_range].cpu().numpy(), self.gamma)), dtype=torch.float32, device=gpu)

        #ep_range_1 = range(self.experience.ep_start_index[batch]+1, self.experience.next_step_index  )
        #ep_range_2 = range(self.experience.ep_start_index[batch]   , self.experience.next_step_index-1)
        #self.weight[batch, ep_range] = self.experience.ep_cum_reward[batch]
        #self.weight[batch, ep_range_1] -= self.experience.cum_reward[batch, ep_range_2]
        


class ValueFunction(PolicyUpdater):
    def __init__(self,
        obs_shape: list,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        policy_lr=1e-2,
        value_net_lr = 1e-2,
    ) -> None:
        super().__init__(policy, experience, policy_lr)
        self.value_net = self.mean_net = nn.Sequential(
            policy.obs_encoder,
            mlp(obs_shape + hidden_sizes + [1])
        )
        self.value_optimizier = torch.optim.Adam(self.value_net.parameters(), lr = value_net_lr)
        self.value_loss_function = torch.nn.MSELoss()
        self.value_loss = 0.0
        self.reward_to_go = torch.zeros(experience.batch_size, experience.buffer_size, dtype=torch.float32, device=gpu)

    def update(self):
        super().update()

        for i in range(20):
            self.value_optimizier.zero_grad()
            self.value_loss = self._value_loss()
            self.value_loss.backward()
            self.value_optimizier.step()

    def _value_loss(self):
        return self.value_loss_function(self.value_net(self.experience.obs).squeeze(-1), self.reward_to_go)

    def _end_episode_batch(self, batch: int):
        ep_range = range(self.experience.ep_start_index[batch], self.experience.next_step_index)
        ep_range_1 = range(self.experience.ep_start_index[batch]+1, self.experience.next_step_index  )
        ep_range_2 = range(self.experience.ep_start_index[batch]   , self.experience.next_step_index-1)
        self.reward_to_go[batch, ep_range] = self.experience.ep_cum_reward[batch]
        self.reward_to_go[batch, ep_range_1] -= self.experience.cum_reward[batch, ep_range_2]
        self.weight = self.reward_to_go

    def log(self, logger):
        logger.log_tabular('ValueLoss', self.value_loss)
        