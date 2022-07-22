from typing import Tuple
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
from purplerl.sync_experience_buffer import ExperienceBuffer, discount_cumsum


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
        min_std: torch.tensor = None,
        std_scale = 2.0
    ) -> None:
        super().__init__(obs_encoder)
        self.action_dist_net_output_shape = action_space.shape + (2, )
        self.action_shape = action_space.shape
        self.std_scale = std_scale
        self.min_std = min_std.to(cfg.device) if min_std is not None else torch.zeros(self.action_shape, **cfg.tensor_args)
        self.action_dist_net_tail = mlp(sizes=list(obs_encoder.shape) + hidden_sizes + [np.prod(np.array(self.action_dist_net_output_shape))])
        self.action_dist_net = nn.Sequential(
            self.obs_encoder,
            self.action_dist_net_tail
        )

    def action_dist(self, obs=None, encoded_obs=None):
        assert((obs is None) != (encoded_obs is None))

        if obs is not None:
            out = self.action_dist_net(obs)
        else:
            out = self.action_dist_net_tail(encoded_obs)
        shape = out.shape[:-1] + self.action_dist_net_output_shape
        out = out.reshape(shape)
        dist_mean = out[...,0]
        dist_std = torch.max(self.std_scale * torch.exp(out[...,1]), self.min_std)
        return Normal(loc=dist_mean, scale=dist_std)


    def checkpoint(self):
        return super().state_dict() | {
            'action_dist_net_state_dict': self.action_dist_net.state_dict(),
            'action_dist_std_scale': self.std_scale
        }


    def load_checkpoint(self, checkpoint):
        self.action_dist_net.load_state_dict(checkpoint['action_dist_net_state_dict'])
        self.action_dist_net_tail = list(self.action_dist_net.children)[-2]
        self.std_scale = checkpoint['action_dist_std_scale']



class PolicyUpdater:
    POLICY_LOSS = "Policy Loss"
    POLICY_LR = "Policy LR"


    def __init__(self,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer
    ) -> None:
        self.policy = policy
        self.experience = experience
        self.stats = {}


    def step(self):
        """
        Called after a new environment step was added to the experience buffer. Called before end_episode().
        """
        pass


    def reset(self):
        self.weight[...] = 0.0

        for stat in self.stats.keys():
            self.stats[stat] = None


    def end_episode(self, finished_envs: torch.Tensor):
        for env_idx, finished in enumerate(finished_envs):
            if not finished:
                continue

            if self.experience.next_step_index==self.experience.ep_start_index[env_idx]:
                raise InvalidStateErr("policy_updater.end_episode must be called before experience_buffer.end_episode!")

            self._end_env_episode(env_idx)


    def buffer_full(self, last_state_value_estimate: torch.tensor):
        assert(self.experience.next_step_index == self.experience.buffer_size)
        for env_idx in range(self.experience.num_envs):
            #  there is nothing to do if this batch just finished an episode
            if self.experience.ep_start_index[env_idx] == self.experience.next_step_index:
                continue

            self._finish_env_path(env_idx, last_state_value_estimate[env_idx])


    def update(self):
        pass

    def value_estimate(self, encoded_obs):
        pass

    def checkpoint(self):
        pass

    def load_checkpoint(self, checkpoint):
        pass

    def _end_env_episode(self, env_idx: int):
        pass

    def _finish_env_path(self, env_idx, last_state_value_estimate: float):
        pass


class PPO(PolicyUpdater):
    KL = "KL"
    POLICY_EPOCHS = "Policy Epochs"
    VALUE_FUNC_EPOCHS = "VF Epochs"
    CLIP_FACTOR = "Clip Factor"
    VALUE_LOSS = "VF Loss"
    VALUE_FUNCTION_LR = "VF LR"

    def __init__(self,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer,
        hidden_sizes: list,
        policy_lr: float,
        vf_lr: float,
        update_batch_size: int,
        update_epochs: int,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
    ) -> None:
        super().__init__(policy, experience)

        self.policy_lr = policy_lr
        self.vf_lr = vf_lr
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.lam = lam

        self.value_net_tail = mlp(list(policy.obs_encoder.shape) + hidden_sizes + [1]).to(cfg.device)

        self.optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': max(vf_lr, policy_lr)},
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': policy_lr},
            {'params': self.value_net_tail.parameters(), 'lr': vf_lr}
        ])

        self.logp_old = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)
        self.weight = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)

        self.obs_loader = DataLoader(TensorDataset(self.experience.obs), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.action_loader = DataLoader(TensorDataset(self.experience.action), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.weight_loader = DataLoader(TensorDataset(self.weight), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.discounted_reward_loader = DataLoader(TensorDataset(self.experience.discounted_reward), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.logp_old_loader = DataLoader(TensorDataset(self.logp_old), batch_size=self.update_batch_size, pin_memory=cfg.pin)


    def buffer_full(self, last_state_value_estimate: torch.tensor):
        super().buffer_full(last_state_value_estimate)

        # If we have a lot more successes than failures (or opposite) in the buffer scale down the weight of the successes such
        # that the underepresented case gets a stronger influence on the gradient
        if self.experience.success_rate() > 0.5:
            self.weight[self.experience.success] *= (1.0 - self.experience.success_rate()) * 2
        else:
            self.weight[torch.logical_not(self.experience.success)] *= self.experience.success_rate() * 2


    def update(self):
        max_kl_reached = False
        value_loss_dropping = True
        self.stats[self.POLICY_EPOCHS] = 0
        self.stats[self.VALUE_FUNC_EPOCHS] = 0

        counts = np.zeros(4)
        totals = torch.zeros(4, requires_grad=False, **cfg.tensor_args)
        policy_loss_total, value_loss_total, kl_total, clip_factor_total = totals[0:1], totals[1:2], totals[2:3], totals[3:4]
        policy_loss_count, value_loss_count, kl_count, clip_factor_count = counts[0:1], counts[1:2], counts[2:3], counts[3:4]
        last_value_loss = float("inf")

        for update_epoch in range(self.update_epochs):
            logp_old_batch_start = 0
            totals[...] = 0.0
            counts[...] = 0
            self.optimizer.zero_grad(set_to_none=True)
            for obs, act, adv, logp_old, discounted_reward in zip(self.obs_loader, self.action_loader, self.weight_loader, self.logp_old_loader, self.discounted_reward_loader):
                # Policy updates
                obs = obs[0].to(cfg.device, non_blocking=True)
                act = act[0].to(cfg.device, non_blocking=True)
                adv = adv[0].to(cfg.device, non_blocking=True)
                encoded_obs = self.policy.obs_encoder(obs)
                del obs

                if update_epoch==0:
                    with torch.no_grad():
                        logp_old = self.policy.action_dist(encoded_obs=encoded_obs).log_prob(act).sum(-1)
                        current_batch_size = encoded_obs.shape[0]
                        batch_range = range(logp_old_batch_start, logp_old_batch_start+current_batch_size)
                        self.logp_old[batch_range, ...] = logp_old.cpu()
                        logp_old_batch_start += current_batch_size
                else:
                    logp_old = logp_old[0].to(cfg.device)

                if not max_kl_reached:
                    policy_loss, kl, clip_factor = self._policy_loss(logp_old, encoded_obs, act, adv)

                    kl_total += kl.sum().detach()
                    kl_count += np.prod(kl.shape).item()

                    if abs(kl_total.item() / kl_count) > 1.5 * self.target_kl:
                        max_kl_reached = True
                    else:
                        policy_loss_total += policy_loss.sum().detach()
                        policy_loss_count += np.prod(policy_loss.shape).item()
                        clip_factor_total += clip_factor.sum().detach()
                        clip_factor_count += np.prod(clip_factor.shape).item()

                        policy_loss = policy_loss.mean()

                    del kl
                    del clip_factor

                if max_kl_reached:
                    policy_loss = torch.zeros(1, device=cfg.device, requires_grad=False)


                # free up memory for the value function update
                del act
                del adv
                del logp_old

                # Value function update
                discounted_reward = discounted_reward[0].to(cfg.device, non_blocking=True)
                value_loss = self._value_loss(encoded_obs, discounted_reward)

                del encoded_obs
                del discounted_reward

                value_loss_total += value_loss.sum().detach()
                value_loss_count += np.prod(value_loss.shape).item()

                value_loss = value_loss.mean()

                if max_kl_reached and value_loss > last_value_loss:
                    value_loss_dropping = False
                    break
                last_value_loss = value_loss

                loss = policy_loss + 1.0 * value_loss
                loss.backward()

                del policy_loss
                del value_loss
                del loss

            if not value_loss_dropping:
                assert(max_kl_reached)
                break

            self.stats[self.VALUE_FUNC_EPOCHS] += 1

            if not max_kl_reached:
                self.stats[self.POLICY_EPOCHS] += 1
                self.stats[self.KL] = kl_total.item() / kl_count.item()
                if clip_factor_count != 0:
                    self.stats[self.POLICY_LOSS] = policy_loss_total.item() / policy_loss_count.item()
                    self.stats[self.CLIP_FACTOR] = clip_factor_total.item() / clip_factor_count.item()
            self.optimizer.step()

            self.stats[self.VALUE_LOSS] = value_loss_total.item() / value_loss_count.item()


    def _policy_loss(self, logp_old, encoded_obs, act, adv) -> Tuple[torch.tensor, float, float, float]:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv))

        # Stats
        kl = logp_old - logp
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clip_factor = torch.as_tensor(clipped, dtype=torch.float32)

        return loss, kl, clip_factor


    def value_estimate(self, encoded_obs):
        return self.value_net_tail(encoded_obs).squeeze(-1)


    def _value_loss(self, encoded_obs, discounted_reward):
        return torch.square(self.value_net_tail(encoded_obs).squeeze(-1) - discounted_reward)


    def _end_env_episode(self, env_idx: int):
        self._finish_env_path(env_idx)


    def _finish_env_path(self, env_idx, last_state_value_estimate: float = None):
        reached_terminal_state = last_state_value_estimate is None
        path_slice = range(self.experience.ep_start_index[env_idx], self.experience.next_step_index)
        rews = self.experience.step_reward[path_slice, env_idx]

        obs_device = self.experience.obs[path_slice, env_idx].to(cfg.device)

        vals = np.zeros(len(path_slice)+1, dtype=np.float32)
        # TODO: Don't encode obs again here
        vals[:-1] = self.value_net_tail(self.policy.obs_encoder(obs_device)).squeeze(-1).cpu().numpy()
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
        return {
            'value_net_state_dict': self.value_net_tail.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }


    def load_checkpoint(self, checkpoint):
        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])