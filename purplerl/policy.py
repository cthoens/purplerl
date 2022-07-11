from typing import Callable, Tuple
from xml.dom import InvalidStateErr
import numpy as np

import gym
from gym.spaces import Discrete, MultiDiscrete

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader

from torch.optim import Adam

import purplerl.config as cfg
from purplerl.sync_experience_buffer import ExperienceBufferBase, discount_cumsum


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
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
        ).to(cfg.device)


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
        action_space: list[int],
        min_std: torch.tensor = None
    ) -> None:
        super().__init__(obs_encoder)
        self.mean_net_output_shape = action_space.shape + (2, )
        self.action_shape = action_space.shape
        self.min_std = min_std.to(cfg.device) if min_std is not None else torch.zeros(self.action_shape, **cfg.tensor_args)
        self.mean_net = nn.Sequential(
            self.obs_encoder,
            mlp(sizes=list(obs_encoder.shape) + hidden_sizes + [np.prod(np.array(self.mean_net_output_shape))])
        )
        #log_std_init = -0.5 * torch.ones(*self.mean_net_output_shape, **tensor_args)
        #self.log_std = torch.nn.Parameter(log_std_init)

    
    def action_dist(self, obs):
        out = self.mean_net(obs)
        shape = out.shape[:-1] + self.mean_net_output_shape
        out = out.reshape(shape)
        dist_mean = out[...,0]
        dist_std = torch.max(torch.exp(out[...,1]), self.min_std)
        return Normal(loc=dist_mean, scale=dist_std)


    def checkpoint(self):
        return super().state_dict() | {
            'mean_net_state_dict': self.mean_net.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.mean_net.load_state_dict(checkpoint['mean_net_state_dict'])
        #self.log_std.data = checkpoint['log_std']



class PolicyUpdater:
    POLICY_LOSS = "Policy Loss"
    POLICY_LR = "Policy LR"

    def __init__(self,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        batch_size,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs: int = 3
    ) -> None:
        self.policy = policy
        self.experience = experience
        self.policy_lr_scheduler = policy_lr_scheduler
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr_scheduler())
        self.policy_epochs = policy_epochs
        self.stats = {}

        self.weight = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)

        self.batch_size = batch_size
        self.obs_loader = DataLoader(TensorDataset(self.experience.obs), batch_size=self.batch_size, pin_memory=cfg.pin)
        self.action_loader = DataLoader(TensorDataset(self.experience.action), batch_size=self.batch_size, pin_memory=cfg.pin)
        self.weight_loader = DataLoader(TensorDataset(self.weight), batch_size=self.batch_size, pin_memory=cfg.pin)


    def step(self):
        """
        Called after a new environment step was added to the experience buffer. Called before end_episode().
        """
        pass


    def end_episode(self, finished_envs: torch.Tensor):
        for env_idx, finished in enumerate(finished_envs):
            if not finished:
                continue

            if self.experience.next_step_index==self.experience.ep_start_index[env_idx]:
                raise InvalidStateErr("policy_updater.end_episode must be called before experience_buffer.end_episode!")

            self._end_episode_batch(env_idx)

    def _end_episode_batch(self, env_idx: int):
        pass

    def buffer_full(self, last_state_value_estimate: torch.tensor):
        assert(self.experience.next_step_index == self.experience.buffer_size)
        for env_idx in range(self.experience.num_envs):
            #  there is nothing to do if this batch just finished an episode
            if self.experience.ep_start_index[env_idx] == self.experience.next_step_index:
                continue

            self._finish_path_batch(env_idx, last_state_value_estimate[env_idx])

    def _finish_path_batch(self, env_idx, last_state_value_estimate: float):
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
            for obs, act, weights in zip(self.obs_loader, self.action_loader, self.weight_loader):
                policy_loss = self._policy_loss(obs[0].to(cfg.device), act[0].to(cfg.device), weights[0].to(cfg.device))
                policy_loss.backward()
                batches += 1
            self.policy_optimizer.step()

        self.stats[self.POLICY_LOSS] = policy_loss.item()


    def value_estimate(self, obs):
        pass


    # make loss function whose gradient, for the right data, is policy gradient
    def _policy_loss(self, obs, act, weights):
        logp = self.policy.action_dist(obs).log_prob(act).sum(-1)
        return -(logp * weights).mean()


    def checkpoint(self):
        return {
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])


class Vanilla(PolicyUpdater):
    VALUE_FUNCTION_LOSS = "VF Loss"
    VALUE_FUNCTION_LR = "VF LR"
    
    def __init__(self,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        batch_size,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs,
        vf_lr_scheduler: Callable[[], float],
        vf_epochs,
        lam: float = 0.95
    ) -> None:
        super().__init__(policy, experience, batch_size, policy_lr_scheduler, policy_epochs=policy_epochs)
        self.lam = lam
        self.vf_lr_scheduler = vf_lr_scheduler
        self.vf_epochs = vf_epochs
        self.value_net = nn.Sequential(
            policy.obs_encoder,
            mlp(list(policy.obs_encoder.shape) + hidden_sizes + [1])
        ).to(cfg.device)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr = vf_lr_scheduler())
        self.value_loss_function = torch.nn.MSELoss()

        self.discounted_reward_loader = torch.utils.data.DataLoader(TensorDataset(self.experience.discounted_reward), batch_size=self.batch_size, pin_memory=cfg.pin)


    def update(self):
        super().update()

        self._update_value_function()


    def reset(self):
        super().reset()

    def value_estimate(self, obs):
        return self.value_net(obs).squeeze(-1)

    def _update_value_function(self):
        new_value_net_lr = self.vf_lr_scheduler()
        for g in self.value_optimizer.param_groups:
            g['lr'] = new_value_net_lr
        self.stats[self.VALUE_FUNCTION_LR] = new_value_net_lr

        for p in self.policy.obs_encoder.parameters():
            p.requires_grad=False
        
        try:
            for i in range(self.vf_epochs):
                for obs, discounted_reward in zip(self.obs_loader, self.discounted_reward_loader):
                    self.value_optimizer.zero_grad()
                    obs = obs[0].to(cfg.device)
                    discounted_reward = discounted_reward[0].to(cfg.device)
                    value_loss = self._value_loss(obs, discounted_reward)
                    value_loss.backward()
                    self.value_optimizer.step()
            self.stats[self.VALUE_FUNCTION_LOSS] = value_loss.item()
        finally:
            for p in self.policy.obs_encoder.parameters():
                p.requires_grad=True


    def _value_loss(self, obs, discounted_reward):
        return self.value_loss_function(self.value_net(obs).squeeze(-1), discounted_reward)


    def _end_episode_batch(self, env_idx: int):
        self._finish_path_batch(env_idx)


    def _finish_path_batch(self, env_idx, last_state_value_estimate: float = None):        
        reached_terminal_state = last_state_value_estimate is None
        path_slice = range(self.experience.ep_start_index[env_idx], self.experience.next_step_index)
        rews = self.experience.step_reward[path_slice, env_idx]
        
        obs_device = self.experience.obs[path_slice, env_idx].to(cfg.device)

        vals = np.zeros(len(path_slice)+1, dtype=np.float32)
        vals[:-1] = self.value_net(obs_device).squeeze(-1).cpu().numpy()
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
        self.weight[path_slice, env_idx] = torch.tensor(np.array(weight), **self.experience.tensor_args)


    def checkpoint(self):
        return super().checkpoint() | {
            'value_net_state_dict': self.value_net.state_dict(),
            'value_net_optimizer_state_dict': self.value_optimizer.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_net_optimizer_state_dict'])


class PPO(Vanilla):    
    KL = "KL"
    ENTROPY = "Entropy"
    POLICY_EPOCHS = "Policy Epochs"
    CLIP_FACTOR = "Clip Factor"
    
    def __init__(self,
        hidden_sizes: list,
        policy: StochasticPolicy,
        experience: ExperienceBufferBase,
        batch_size,
        policy_lr_scheduler: Callable[[], float],
        policy_epochs,
        vf_lr_scheduler: Callable[[], float],
        vf_epochs,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
    ) -> None:
        super().__init__(hidden_sizes, policy, experience, batch_size, policy_lr_scheduler, policy_epochs, vf_lr_scheduler, vf_epochs, lam)
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.vf_only_updates = 0

        self.logp_old = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)

        self.logp_old_loader = DataLoader(TensorDataset(self.logp_old), batch_size=self.batch_size, pin_memory=cfg.pin)


    def update(self):
        new_policy_lr = self.policy_lr_scheduler()
        for g in self.policy_optimizer.param_groups:
            g['lr'] = new_policy_lr
        self.stats[self.POLICY_LR] = new_policy_lr

        new_value_net_lr = self.vf_lr_scheduler()
        for g in self.value_optimizer.param_groups:
            g['lr'] = new_value_net_lr
        self.stats[self.VALUE_FUNCTION_LR] = new_value_net_lr

        batch_start = 0
        with torch.no_grad():
            for obs, act in zip(self.obs_loader, self.action_loader):
                obs = obs[0].to(cfg.device)
                act = act[0].to(cfg.device)
                current_batch_size = obs.shape[0]
                self.logp_old[batch_start: batch_start+current_batch_size, ...] = self.policy.action_dist(obs).log_prob(act).sum(-1).cpu()
                batch_start += current_batch_size
        
        # Train policy with multiple steps of gradient descent
        max_kl_reached = False
        self.stats[self.POLICY_EPOCHS] = 0
        for i in range(max(self.policy_epochs, self.vf_epochs)):
            update_policy = self.vf_only_updates == 0 and i < self.policy_epochs and not max_kl_reached
            if update_policy:
                loss_total, kl_total, entropy_total, clip_factor_total = 0, 0, 0, 0
                loss_count, kl_count, entropy_count, clip_factor_count = 0, 0, 0, 0
                self.policy_optimizer.zero_grad()
            for obs, act, adv, logp_old, discounted_reward in zip(self.obs_loader, self.action_loader, self.weight_loader, self.logp_old_loader, self.discounted_reward_loader):
                # Policy updates                
                obs = obs[0].to(cfg.device)
                if update_policy and not max_kl_reached:
                    act = act[0].to(cfg.device)
                    adv = adv[0].to(cfg.device)
                    logp_old = logp_old[0].to(cfg.device)

                    policy_loss, kl, entropy, clip_factor = self._policy_loss(logp_old, obs, act, adv)

                    kl_total += kl.sum().item()
                    kl_count += np.prod(kl.shape).item()

                    if kl_total / kl_count > 1.5 * self.target_kl:
                        max_kl_reached = True
                    else:
                        loss_total += policy_loss.sum().item()
                        loss_count += np.prod(policy_loss.shape).item()
                        entropy_total += entropy.sum().item()
                        entropy_count += np.prod(entropy.shape).item()
                        clip_factor_total += clip_factor.sum().item()
                        clip_factor_count += np.prod(clip_factor.shape).item()

                        policy_loss.mean().backward()

                    # free up memory for the value function update
                    del act
                    del adv
                    del policy_loss
                    del logp_old
                    del kl
                    del entropy
                    del clip_factor

                # Value function update
                if i < self.vf_epochs:
                    for p in self.policy.obs_encoder.parameters():
                        p.requires_grad=False
                    
                    try:
                        self.value_optimizer.zero_grad()
                        discounted_reward = discounted_reward[0].to(cfg.device)
                        value_loss = self._value_loss(obs, discounted_reward)
                        value_loss.backward()
                        self.value_optimizer.step()
                        value_loss_total = value_loss.item()
                    finally:
                        del discounted_reward
                        del value_loss
                        for p in self.policy.obs_encoder.parameters():
                            p.requires_grad=True
                                 
            if update_policy:
                self.policy_optimizer.step()
                self.stats[self.POLICY_EPOCHS] += 1
        
        self.stats[self.VALUE_FUNCTION_LOSS] = value_loss_total
        if self.vf_only_updates == 0:
            self.stats[self.KL] = kl_total / kl_count
            if entropy_count != 0:
                self.stats[self.ENTROPY] = entropy_total / entropy_count            
                self.stats[self.CLIP_FACTOR] = clip_factor_total / clip_factor_count
                self.stats[self.POLICY_LOSS] = loss_total / loss_count

        self.vf_only_updates = max(self.vf_only_updates -1, 0)


    def _policy_loss(self, logp_old, obs, act, adv) -> Tuple[torch.tensor, float, float, float]:
        # Policy loss
        action_dist = self.policy.action_dist(obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv))
        
        # Stats
        kl = logp_old - logp
        entropy = action_dist.entropy()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clip_factor = torch.as_tensor(clipped, dtype=torch.float32)
        
        return loss, kl, entropy, clip_factor