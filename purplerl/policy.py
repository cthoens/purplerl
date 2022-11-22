from enum import Enum
import math
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

from purplerl.experience_buffer import ExperienceBuffer, discount_cumsum
import purplerl.config as config


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        layers += [nn.Linear(sizes[j], sizes[j+1], bias = True)]
        layers += [activation() if j < len(sizes)-2 else output_activation()]

    return nn.Sequential(*layers)


class LearningRateManager:
    def __init__(self, optimizer, initial_lr, decay, decay_revert_steps, min_factor=0.0) -> None:
        self.optimizer = optimizer
        self.initial_lr =  initial_lr
        self.min_factor = min_factor
        self.factor = 1.0
        self.decay = decay
        self.decay_revert_steps = decay_revert_steps

    def inc(self):
        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        factor = (1.0 / self.decay)**(1/self.decay_revert_steps)

        self.factor *= factor
        lr = self.initial_lr * self.factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    def dec(self, decy_steps = 1.0):
        self.factor *= max(self.factor * (self.decay**decy_steps), 0.01)
        lr = self.initial_lr * self.factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr



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
        action_space: gym.Space,
        hidden_sizes: list[int],
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
        )


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
        action_space: list[int],
        action_dist_net_tail: list[int],
        min_std: torch.tensor = None,
        max_std: torch.tensor = None,
    ) -> None:
        super().__init__(obs_encoder)
        self.action_dist_net_output_shape = action_space.shape + (2, )
        self.action_shape = action_space.shape
        self.action_dist_net_tail = action_dist_net_tail
        self.action_dist_net = nn.Sequential(
            self.obs_encoder,
            self.action_dist_net_tail
        )
        self.min_std = min_std if min_std is not None else torch.full(-1.0, *self.action_shape)
        self.max_std = max_std if max_std is not None else torch.full( 1.0, *self.action_shape)
        self.std_offset = (np.log(self.min_std) + np.log(self.max_std)) / 2.0
        self.std_scale = np.log(self.max_std) - self.std_offset


    def action_dist(self, obs=None, encoded_obs=None):
        assert((obs is None) != (encoded_obs is None))

        if obs is not None:
            out = self.action_dist_net(obs)
        else:
            out = self.action_dist_net_tail(encoded_obs)
        shape = out.shape[:-1] + self.action_dist_net_output_shape
        out = out.reshape(shape)
        dist_mean = out[...,0]
        dist_std = torch.clamp(torch.exp(self.std_scale * out[...,1] + self.std_offset), self.min_std, self.max_std)
        return Normal(loc=dist_mean, scale=dist_std)


    def to(self, device):
        super().to(device)

        self.std_offset = self.std_offset.to(device)
        self.std_scale = self.std_scale.to(device)
        self.min_std = self.min_std.to(device)
        self.max_std = self.max_std.to(device)

        return self


    def checkpoint(self):
        return {
            'action_dist_net_state_dict': {k: v.cpu() for k, v in self.action_dist_net.state_dict().items()},
        }


    def load_checkpoint(self, checkpoint):
        self.action_dist_net.load_state_dict(checkpoint['action_dist_net_state_dict'])
        self.action_dist_net_tail = list(self.action_dist_net.children())[-1]


class PPO():
    CLIP_FACTOR = "Clip Factor"
    POLICY_UPDATE_EPOCHS = "Policy Update Epochs"
    POLICY_LR_FACTOR = "Policy LR Factor"
    POLICY_LOSS = "Policy Loss"
    ABS_KL_LOWER = "Abs KL Lower"
    ABS_KL_UPPER = "Abs KL Upper"
    ABS_KL_MEAN = "Abs KL Mean"
    ABS_KL_STD = "Abs KL Std"
    ABS_KL_MIN = "Abs KL Min"
    ABS_KL_MAX = "Abs KL Max"

    def __init__(self,
        cfg: dict,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer,
        value_net_tail: torch.nn.Module,

        adv_lambda: float,
        clip_ratio: float,
        entropy_factor: float,

        # bounds
        policy_kl_lower_min: float,
        policy_kl_upper_target: float,
        policy_kl_upper_max: float,
        policy_valid_kl_lower_bound: float,
        policy_shifting_right_way_scale: float,

        # policy lr
        policy_initial_lr: float,
        policy_lr_decay: float,
        policy_update_batch_size: int,
        policy_update_epochs: int,

        # vf lr
        vf_initial_lr: float,
        vf_lr_decay: float,
        vf_update_batch_size: int,
        vf_update_epochs: int,

    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.experience = experience
        self.stats = {}

        self.value_net_tail = value_net_tail
        self.clip_ratio = clip_ratio
        self.adv_lambda = adv_lambda
        self.entropy_factor = entropy_factor

        self.policy_kl_lower_min = policy_kl_lower_min
        self.policy_kl_upper_target = policy_kl_upper_target
        self.policy_kl_upper_max = policy_kl_upper_max
        self.policy_valid_kl_lower_bound = policy_valid_kl_lower_bound
        self.policy_shifting_right_way_scale = policy_shifting_right_way_scale

        self.vf_kl_lower_min = -100
        self.vf_kl_upper_target = 1.8
        self.vf_kl_upper_max = 100
        self.vf_valid_kl_lower_bound = 0.80

        self.policy_update_epochs = policy_update_epochs
        self.policy_update_batch_size = policy_update_batch_size * experience.num_envs
        self.policy_optimizer = Adam([
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': policy_initial_lr},
        ])
        self.policy_lr = LearningRateManager(self.policy_optimizer, policy_initial_lr, policy_lr_decay, decay_revert_steps=10)

        self.vf_update_epochs = vf_update_epochs
        self.vf_update_batch_size = vf_update_batch_size
        self.vf_optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': vf_initial_lr},
            {'params': self.value_net_tail.parameters(), 'lr': vf_initial_lr}
        ])
        self.vf_lr = LearningRateManager(self.vf_optimizer, vf_initial_lr, vf_lr_decay, decay_revert_steps=10, min_factor=0.01)

        suffix = ""
        self.VF_UPDATE_EPOCHS = f"VF Update Epochs {suffix}"
        self.VF_LR_FACTOR = f"VF LR Factor {suffix}"
        self.VALUE_LOSS_IN_MEAN = f"VF Loss Mean {suffix}"
        self.VALUE_LOSS_IN_STD = f"VF Loss Std {suffix}"
        self.VALUE_LOSS_IN_MAX = f"VF Loss Max {suffix}"
        self.VF_DELTA = f"VF Delta {suffix} %"
        self.VALUE_NET_PENALTY = f"Value Net Penality {suffix}"

        # The extra slot at the end of used in _finish_env_path().
        # Note: This is used during experience collection to calculate advanteaged and during
        # the update to calculate loss statistics
        self.value_old_full = torch.zeros(experience.buffer_size+1, experience.num_envs, pin_memory=cfg.pin, dtype=config.dtype)
        self.value_old = self.value_old_full[:-1,...]
        self.kl = torch.zeros(experience.buffer_size, experience.num_envs, pin_memory=cfg.pin, dtype=config.dtype)
        self.weight = torch.zeros(experience.buffer_size, experience.num_envs, pin_memory=cfg.pin, **experience.tensor_args)

        self.value_old_merged = self.value_old.reshape(-1)
        self.kl_merged = self.kl.reshape(-1)
        self.weight_merged = self.weight.reshape(-1)

        self.value_delta_old = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, pin_memory=cfg.pin, **experience.tensor_args)
        self.value_delta_current = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, pin_memory=cfg.pin, **experience.tensor_args)
        self.abs_kl_sign = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, pin_memory=cfg.pin, **experience.tensor_args)
        self.policy_loss = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, pin_memory=cfg.pin, **experience.tensor_args)
        self.clip_factor = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, pin_memory=cfg.pin, **experience.tensor_args)

        self.obs_loader = DataLoader(TensorDataset(self.experience.obs_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.action_loader = DataLoader(TensorDataset(self.experience.action_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.weight_loader = DataLoader(TensorDataset(self.weight_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.logp_old_loader = DataLoader(TensorDataset(self.experience.action_logp_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)

        vf_update_batch_size = vf_update_batch_size * experience.num_envs
        self.vf_discounted_reward_loader = DataLoader(TensorDataset(experience.discounted_reward_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)

    def step(self, encoded_obs):
        # Note: Clamped to plausible range
        self.value_old[self.experience.next_step_index] = self.value_net_tail(encoded_obs).clamp(-1.0, 1.0).squeeze(-1).cpu()


    def buffer_full(self, last_state_value_estimate: torch.tensor):
        super().buffer_full(last_state_value_estimate)

        # If we have a lot more successes than failures (or opposite) in the buffer scale down the weight of the successes such
        # that the underepresented case gets a stronger influence on the gradient
        if self.experience.success_rate() > 0.5:
            # Value function predicted success: Advantage is < 1.0 and made smaller
            # Value function predicted failure: Advantage is > 1.7 and made smaller
            self.weight[self.experience.success] *= max((1.0 - self.experience.success_rate()) * 2.0, 0.20)
        else:
            self.weight[torch.logical_not(self.experience.success)] *= max(self.experience.success_rate() * 2.0, 0.20)


    def update(self):
        for key in self.stats:
            self.stats[key] = None

        epoch_value_loss = self.value_old_merged
        self.stats[self.VALUE_LOSS_IN_MEAN] = epoch_value_loss.mean().item()
        self.stats[self.VALUE_LOSS_IN_STD] = epoch_value_loss.std().item()
        self.stats[self.VALUE_LOSS_IN_MAX] = epoch_value_loss.max().item()

        has_updated = False
        policy_is_valid_end_state, vf_is_valid_end_state = False, False
        policy_gen = self._update_policy(self.policy_lr)
        vf_gen = self._update_vf(self.vf_lr)

        try:
            for update_epoch in range(self.policy_update_epochs+1):
                is_validate_epoch = update_epoch == self.policy_update_epochs

                batch_start = 0
                self.policy_optimizer.zero_grad(set_to_none=True)
                self.vf_optimizer.zero_grad(set_to_none=True)
                for obs, act, adv, logp_old, discounted_reward in zip(self.obs_loader, self.action_loader, self.weight_loader, self.logp_old_loader, self.vf_discounted_reward_loader):
                    assert(obs[0].shape[0] == self.policy_update_batch_size)
                    batch_range = range(batch_start, batch_start+self.policy_update_batch_size)
                    batch_start += self.policy_update_batch_size

                    # Prepare dataset
                    # ---------------

                    obs = obs[0].to(self.cfg.device, non_blocking=True)
                    act = act[0].to(self.cfg.device, non_blocking=True)
                    adv = adv[0].to(self.cfg.device, non_blocking=True)
                    logp_old = logp_old[0].to(self.cfg.device)

                    encoded_obs = self.policy.obs_encoder(obs)
                    del obs

                    # Policy update
                    # -------------
                    # Note: Calculdate even if update_policy == False to check for passive_policy_progression (see below)

                    #TODO
                    scale = 1.0 #if update_is_perfect else self.policy_shifting_right_way_scale
                    batch_policy_loss = self._policy_loss(batch_range, scale, logp_old, encoded_obs.detach(), act, adv)

                    # free up memory for the value function update
                    del logp_old
                    del act
                    del adv


                    # VF Update
                    # ----------
                    #encoded_obs = encoded_obs.detach()
                    discounted_reward = discounted_reward[0].to(self.cfg.device, non_blocking=True)

                    batch_value_loss  = self._value_loss(batch_range, encoded_obs, discounted_reward)

                    del discounted_reward
                    del encoded_obs

                    if not is_validate_epoch:
                        batch_policy_loss.backward()
                        batch_value_loss.backward()
                        #(batch_policy_loss + batch_value_loss).backward()

                    del batch_policy_loss
                    del batch_value_loss
                # end for: iterate through dataset batches

                if update_epoch > 0: print(f"{update_epoch:2}: ", end="")
                policy_can_continue, policy_is_valid_end_state, policy_target_reached = next(policy_gen)
                if update_epoch > 0: print(f"    /   ", end="")
                vf_can_continue, vf_is_valid_end_state, vf_target_reached = next(vf_gen)
                if update_epoch > 0: print("")

                if policy_is_valid_end_state and vf_is_valid_end_state:
                    has_updated = True
                    policy_valid_state_cp = self.policy_backtrack_point
                    vf_valid_state_cp = self.vf_backtrack_point

                if policy_target_reached and vf_target_reached:
                    break

                if not policy_can_continue or not vf_can_continue:
                    break
        finally:
            if not policy_is_valid_end_state or not vf_is_valid_end_state:
                self._load_policy_update_checkpoint(policy_valid_state_cp)
                self._load_vf_update_checkpoint(vf_valid_state_cp)
                print(f"BT: {policy_valid_state_cp['update_epoch']}")
            else:
                self.policy_lr.inc()
                self.vf_lr.inc()

            if not has_updated:
                print("!***")
                self.policy_lr.dec(decy_steps = 2.0)
                self.vf_lr.dec(decy_steps = 2.0)


    def _update_policy(self, lr: LearningRateManager):
        self.stats[self.POLICY_UPDATE_EPOCHS] = 0
        self.stats[self.POLICY_LR_FACTOR] = lr.factor

        prev_improvement_lower = 1.0
        epochs_since_oscilating = 0
        update_is_perfect = False
        update_was_perfect = False

        class Phase(Enum):
            DROP = 1,
            TENTATIVE_RISE = 2,
            TENTATIVE_DROP = 3,
            RISE = 4

        phase = Phase.DROP

        is_valid_end_state = True
        can_continue = True
        target_reached = False

        self.abs_kl_sign[:] = -1.0
        self.abs_kl_sign[self.weight_merged<0] = 1.0

        self.policy_backtrack_point = self._policy_update_checkpoint()
        self.policy_backtrack_point["prev_improvement_lower"] = 1.0
        self.policy_backtrack_point["prev_improvement_upper"] = 1.0
        self.policy_backtrack_point["update_epoch"] = 0
        self.policy_optimizer.step()

        for update_epoch in range(1, self.policy_update_epochs+1):
            yield can_continue, is_valid_end_state, target_reached

            is_validate_epoch = update_epoch == self.policy_update_epochs
            is_lr_inc_epoch = epochs_since_oscilating >= 2

            epoch_kl = self.kl
            epoch_improvement = epoch_kl.flatten() * self.abs_kl_sign
            epoch_improvement_min = epoch_improvement.min().item()
            epoch_improvement_mean = epoch_improvement.mean().item()
            epoch_improvement_std = epoch_improvement.std().item()
            epoch_improvement_max = epoch_improvement.max().item()
            epoch_improvement_lower = math.exp(epoch_improvement_mean - epoch_improvement_std)
            epoch_improvement_upper = math.exp(epoch_improvement_mean + epoch_improvement_std)

            update_is_perfect = epoch_improvement_lower >= 1.0 and epoch_improvement_upper > 1.0
            improvement_lower_out_of_bounds = epoch_improvement_lower < self.policy_kl_lower_min
            improvement_upper_out_of_bounds = epoch_improvement_upper > 1.0 and epoch_improvement_upper > self.policy_kl_upper_max
            improvement_out_of_bounds = improvement_upper_out_of_bounds or improvement_lower_out_of_bounds
            is_valid_end_state = epoch_improvement_lower >= self.policy_valid_kl_lower_bound and not improvement_out_of_bounds
            target_reached = epoch_improvement_upper > self.policy_kl_upper_target and is_valid_end_state

            # BEGIN: Check if policy is oscylating
            is_oscilating = False
            improvement_lower_dropped = epoch_improvement_lower < prev_improvement_lower
            improvement_lower_rose = not improvement_lower_dropped
            if phase == Phase.DROP:
                if improvement_lower_rose:
                    phase = Phase.TENTATIVE_RISE
                    tentative_rise_start_epoch = update_epoch
            elif phase == Phase.TENTATIVE_RISE:
                is_lr_inc_epoch = False

                if improvement_lower_rose and update_epoch - tentative_rise_start_epoch >= 2:
                    phase = Phase.RISE

                elif improvement_lower_dropped:
                    phase = Phase.DROP
                    is_oscilating = True

            elif phase == Phase.RISE:
                if improvement_lower_dropped:
                    phase = Phase.TENTATIVE_DROP
                    tentative_drop_start_epoch = update_epoch

            elif phase == Phase.TENTATIVE_DROP:
                is_lr_inc_epoch = False

                if improvement_lower_dropped and update_epoch - tentative_drop_start_epoch >= 2:
                    phase = Phase.DROP

                elif improvement_lower_rose:
                    phase = Phase.RISE
                    is_oscilating = True


            if phase == Phase.DROP:
                print(f"↓ ", end="")
            elif phase == Phase.TENTATIVE_RISE:
                print(f"? ", end="")
            elif phase == Phase.TENTATIVE_DROP:
                print(f"? ", end="")
            elif phase == Phase.RISE:
                print(f"↑ ", end="")

            if is_oscilating:
                print(f" ~", end="")
            else:
                print(f"  ", end="")
            # ---

            # END: Check if policy is oscylating
            is_lr_inc_epoch = is_lr_inc_epoch and (phase == Phase.RISE or phase == Phase.DROP) and (not update_is_perfect or update_was_perfect)
            if update_is_perfect and not update_was_perfect:
                lr.dec()
            update_was_perfect = update_was_perfect or update_is_perfect

            if is_valid_end_state:
                print(f" *", end="")
            else:
                print(f"  ", end="")
            print(f"{epoch_improvement_lower:.4f} {epoch_improvement_upper:.4f}   {math.exp(epoch_improvement_min):.4f} {math.exp(epoch_improvement_max):.4f}", end="")

            if improvement_out_of_bounds:
                if improvement_upper_out_of_bounds:
                    lr.dec(decy_steps = 1.0)
                    print(f"   ↓{self.policy_lr.factor:.3f} ", end="")
                else:
                    lr.dec(decy_steps = 2.0)
                    print(f"  ↓↓{self.policy_lr.factor:.3f} ", end="")
                can_continue = False
                is_valid_end_state = False
                continue

            elif is_oscilating:
                lr.dec(decy_steps = 1.0)
                epochs_since_oscilating = -1
                print(f"    {self.policy_lr.factor:.3f} ", end="")

            elif is_lr_inc_epoch:
                lr.inc()
                print(f"    {self.policy_lr.factor:.3f}↑", end="")

            else:
                print(f"    {self.policy_lr.factor:.3f} ", end="")

            epochs_since_oscilating+=1
            self.stats[self.POLICY_LR_FACTOR] = self.policy_lr.factor

            if not is_validate_epoch:
                self.policy_backtrack_point = self._policy_update_checkpoint()
                self.policy_backtrack_point["prev_improvement_lower"] = epoch_improvement_lower
                self.policy_backtrack_point["prev_improvement_upper"] = epoch_improvement_upper
                self.policy_backtrack_point["update_epoch"] = update_epoch
                self.policy_optimizer.step()

            prev_improvement_lower = epoch_improvement_lower

            if is_valid_end_state:
                has_updated = True
                self.stats[self.POLICY_UPDATE_EPOCHS] += 1
                self.stats[self.ABS_KL_LOWER] = epoch_improvement_lower
                self.stats[self.ABS_KL_UPPER] = epoch_improvement_upper
                self.stats[self.ABS_KL_MEAN] = math.exp(epoch_improvement_mean)
                self.stats[self.ABS_KL_STD] = math.exp(epoch_improvement_std)
                self.stats[self.ABS_KL_MIN] = math.exp(epoch_improvement_min)
                self.stats[self.ABS_KL_MAX] = math.exp(epoch_improvement_max)
                self.stats[self.POLICY_LOSS] = self.policy_loss.mean().item()
                self.stats[self.CLIP_FACTOR] = self.clip_factor.mean().item()

            if target_reached and has_updated:
                break
        yield can_continue, is_valid_end_state, target_reached


    def _update_vf(self, lr:LearningRateManager):
        self.stats[self.VF_UPDATE_EPOCHS] = 0
        self.stats[self.VF_LR_FACTOR] = lr.factor

        prev_improvement_lower = 1.0
        epochs_since_oscilation = 0
        update_is_perfect = False
        update_was_perfect = False

        class Phase(Enum):
            DROP = 1,
            TENTATIVE_RISE = 2,
            TENTATIVE_DROP = 3,
            RISE = 4

        phase = Phase.DROP

        is_valid_end_state = True
        can_continue = True
        target_reached = False

        self.value_delta_old[...] = self.value_delta_current[...]

        self.vf_backtrack_point = self._vf_update_checkpoint()
        self.vf_backtrack_point["prev_improvement_lower"] = 1.0
        self.vf_backtrack_point["prev_improvement_upper"] = 1.0
        self.vf_backtrack_point["update_epoch"] = 0
        self.vf_optimizer.step()

        for update_epoch in range(1, self.vf_update_epochs+1):
            yield can_continue, is_valid_end_state, target_reached

            is_validate_epoch = update_epoch == self.vf_update_epochs
            is_lr_inc_epoch = epochs_since_oscilation >= 2

            epoch_value_delta = self.value_delta_current

            epoch_improvement = self.value_delta_old / epoch_value_delta
            epoch_improvement_min = epoch_improvement.min().item()
            epoch_improvement_mean = epoch_improvement.mean().item()
            epoch_improvement_std = epoch_improvement.std().item()
            epoch_improvement_max = epoch_improvement.max().item()
            epoch_improvement_lower = epoch_improvement_mean - epoch_improvement_std
            epoch_improvement_upper = epoch_improvement_mean + epoch_improvement_std

            update_is_perfect = epoch_improvement_lower >= 1.0 and epoch_improvement_upper > 1.0
            improvement_lower_out_of_bounds = epoch_improvement_lower < self.vf_kl_lower_min
            improvement_upper_out_of_bounds = epoch_improvement_upper > 1.0 and epoch_improvement_upper > self.vf_kl_upper_max
            improvement_out_of_bounds = improvement_upper_out_of_bounds or improvement_lower_out_of_bounds
            is_valid_end_state = epoch_improvement_lower >= self.vf_valid_kl_lower_bound and not improvement_out_of_bounds
            target_reached = epoch_improvement_upper > self.vf_kl_upper_target and is_valid_end_state

            # BEGIN: Check if vf is oscylating
            is_oscilating = False
            improvement_lower_dropped = epoch_improvement_lower < prev_improvement_lower
            improvement_lower_rose = not improvement_lower_dropped
            if phase == Phase.DROP:
                if improvement_lower_rose:
                    phase = Phase.TENTATIVE_RISE
                    tentative_rise_start_epoch = update_epoch
            elif phase == Phase.TENTATIVE_RISE:
                is_lr_inc_epoch = False

                if improvement_lower_rose and update_epoch - tentative_rise_start_epoch >= 2:
                    phase = Phase.RISE

                elif improvement_lower_dropped:
                    phase = Phase.DROP
                    is_oscilating = True

            elif phase == Phase.RISE:
                if improvement_lower_dropped:
                    phase = Phase.TENTATIVE_DROP
                    tentative_drop_start_epoch = update_epoch

            elif phase == Phase.TENTATIVE_DROP:
                is_lr_inc_epoch = False

                if improvement_lower_dropped and update_epoch - tentative_drop_start_epoch >= 2:
                    phase = Phase.DROP

                elif improvement_lower_rose:
                    phase = Phase.RISE
                    is_oscilating = True

            if phase == Phase.DROP:
                print(f"↓ ", end="")
            elif phase == Phase.TENTATIVE_RISE:
                print(f"? ", end="")
            elif phase == Phase.TENTATIVE_DROP:
                print(f"? ", end="")
            elif phase == Phase.RISE:
                print(f"↑ ", end="")

            if is_oscilating:
                print(f" ~", end="")
            else:
                print(f"  ", end="")
            # ---

            # END: Check if vf is oscylating
            is_lr_inc_epoch = is_lr_inc_epoch and (phase == Phase.RISE or phase == Phase.DROP) and (not update_is_perfect or update_was_perfect)
            if update_is_perfect and not update_was_perfect:
                lr.dec()
            update_was_perfect = update_was_perfect or update_is_perfect

            if is_valid_end_state:
                print(f" *", end="")
            else:
                print(f"  ", end="")
            print(f"{epoch_improvement_lower:.4f} {epoch_improvement_upper:.4f}   {epoch_improvement_min:.4f} {epoch_improvement_max:.4f}", end="")

            if improvement_out_of_bounds:
                if improvement_upper_out_of_bounds:
                    lr.dec(decy_steps = 1.0)
                    print(f"   ↓{lr.factor:.3f}", end="")
                else:
                    lr.dec(decy_steps = 2.0)
                    print(f"  ↓↓{lr.factor:.3f}", end="")
                can_continue = False
                is_valid_end_state = False
                continue

            elif is_oscilating:
                lr.dec(decy_steps = 1.0)
                epochs_since_oscilation = -1
                print(f"    {lr.factor:.3f}", end="")

            elif is_lr_inc_epoch:
                lr.inc()
                print(f"    {lr.factor:.3f}↑", end="")

            else:
                print(f"    {lr.factor:.3f}", end="")

            epochs_since_oscilation+=1
            self.stats[self.VF_LR_FACTOR] = lr.factor

            if not is_validate_epoch:
                self.vf_backtrack_point = self._vf_update_checkpoint()
                self.vf_backtrack_point["prev_improvement_lower"] = epoch_improvement_lower
                self.vf_backtrack_point["prev_improvement_upper"] = epoch_improvement_upper
                self.vf_optimizer.step()

            prev_improvement_lower = epoch_improvement_lower

            if is_valid_end_state:
                self.stats[self.VF_UPDATE_EPOCHS] += 1
                self.stats[self.VF_DELTA] = epoch_improvement_mean
        # end of loop
        yield can_continue, is_valid_end_state, target_reached


    def _policy_loss(self, batch_range, positive_shift_scale, logp_old, encoded_obs, act, adv) -> torch.tensor:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_ratio = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
        clip_adv = clip_ratio * adv
        batch_policy_loss = -(torch.min(ratio * adv, clip_adv))
        self.policy_loss[batch_range] = batch_policy_loss.detach().cpu()

        # policy net penality
        policy_net_penalty = self.policy.action_dist_net_tail(encoded_obs)
        policy_net_penalty = torch.square(policy_net_penalty)
        in_bounds = policy_net_penalty <= 1.2 * 1.2
        policy_net_penalty[in_bounds] = 0.0
        policy_net_penalty[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        output_penality_sum = torch.mean(policy_net_penalty)

        # Stats
        entropy = action_dist.entropy()
        self.kl_merged[batch_range] = (logp_old - logp).detach().cpu()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        self.clip_factor[batch_range] = torch.as_tensor(clipped, dtype=torch.float32).cpu()

        batch_abs_kl = batch_policy_loss * -1.0 * self.abs_kl_sign[batch_range].to(self.cfg.device)
        batch_policy_loss[batch_abs_kl >= math.log(self.policy_kl_upper_target)] *= -1.0
        if positive_shift_scale != 1.0:
            batch_policy_loss[batch_abs_kl < 0.0] *= positive_shift_scale
        del batch_abs_kl

        batch_policy_loss = batch_policy_loss.mean() - batch_policy_loss.std()
        backprop_policy_loss = batch_policy_loss + self.entropy_factor * entropy.mean() + output_penality_sum

        self.stats["Policy Net Penalty"] = output_penality_sum.item()

        return backprop_policy_loss


    def _value_loss(self, batch_range, encoded_obs, discounted_reward):
        value_net_out = self.value_net_tail(encoded_obs).squeeze(-1)

        value_net_penality = torch.square(value_net_out)
        in_bounds = value_net_penality <= 1.2 * 1.2
        value_net_penality[in_bounds] = 0.0
        value_net_penality[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        value_net_penality_sum = torch.sum(value_net_penality)

        self.stats[self.VALUE_NET_PENALTY] = value_net_penality_sum.item()

        # If we have a lot more successes than failures (or opposite) in the buffer scale down the weight of the successes such
        # that the underepresented case gets a stronger influence on the gradient
        #if self.experience.success_rate() > 0.5:
            # Value function predicted success: Advantage is < 1.0 and made smaller
            # Value function predicted failure: Advantage is > 1.7 and made smaller
        #    batch_value_loss[self.experience.success.flatten()[batch_range]] *= max((1.0 - self.experience.success_rate()) * 2.0, 0.20)
        #else:
        #    batch_value_loss[torch.logical_not(self.experience.success.flatten()[batch_range])] *= max(self.experience.success_rate() * 2.0, 0.20)

        value_delta = value_net_out - discounted_reward
        self.value_delta_current[batch_range] = value_delta.detach().cpu()

        batch_value_loss = torch.square(value_delta).mean() - torch.square(value_delta).std()

        return batch_value_loss

    def value_estimate(self, encoded_obs):
        # Note: Clamped to plausible range
        return self.value_net_tail(encoded_obs).squeeze(-1).clamp(-1.0, 1.0)


    def _end_env_episode(self, env_idx: int):
        self._finish_env_path(env_idx)


    def _finish_env_path(self, env_idx, last_state_value_estimate: float = None):
        reached_terminal_state = last_state_value_estimate is None
        path_slice = range(self.experience.ep_start_index[env_idx], self.experience.next_step_index)
        path_slice_plus_one = range(self.experience.ep_start_index[env_idx], self.experience.next_step_index+1)
        rews = self.experience.step_reward[path_slice, env_idx]

        vals = self.value_old_full[path_slice_plus_one, env_idx].numpy()
        vals[-1] = last_state_value_estimate if not reached_terminal_state else 0.0

        # estimated reward for state transition = estimated value of next state - estimated value of current state
        next_state_value = vals[1:]
        current_state_value = vals[:-1]
        expected_rewards = self.experience.discount * next_state_value - current_state_value
        # if the episode has reached a terminal state the last state has no next state. In this case use the value
        # of the last state as value estimate of the terminal state
        if reached_terminal_state: expected_rewards[-1] = current_state_value[-1]
        deltas = rews + expected_rewards
        weight = discount_cumsum(deltas, self.experience.discount * self.adv_lambda)
        self.weight[path_slice, env_idx] = torch.tensor(np.array(weight), **self.experience.tensor_args)


    def _policy_update_checkpoint(self):
        full_state = {
            "policy": self.policy.checkpoint(),
            "policy_updater": {
                'optimizer_state_dict': self.policy_optimizer.state_dict(),
            }
        }
        opt_dict = full_state["policy_updater"]["optimizer_state_dict"]
        opt_dict["state"] = {k1: {k: v.cpu() for k, v in v1.items()} for k1, v1 in opt_dict["state"].items()}
        return full_state

    def _load_policy_update_checkpoint(self, checkpoint):
        self.policy.load_checkpoint(checkpoint["policy"])

        checkpoint = checkpoint["policy_updater"]
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        lr = self.policy_lr.initial_lr * self.policy_lr.factor
        for group in self.policy_optimizer.param_groups:
            group['lr'] = lr

    def checkpoint(self):
        return {
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'initial_lr': self.policy_lr.initial_lr,
            'lr_factor': self.policy_lr.factor,

            'value_net_state_dict': {k: v.cpu() for k, v in self.value_net_tail.state_dict().items()},
            'vf_optimizer_state_dict': self.vf_optimizer.state_dict(),
            'vf_initial_lr': self.vf_lr.initial_lr,
            'vf_lr_factor': self.vf_lr.factor,
        }


    def load_checkpoint(self, checkpoint):
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_lr.initial_lr = checkpoint['initial_lr']
        self.policy_lr.factor = checkpoint['lr_factor']

        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.vf_optimizer.load_state_dict(checkpoint['vf_optimizer_state_dict'])
        self.vf_lr.initial_lr = checkpoint['vf_initial_lr']
        self.vf_lr.factor = checkpoint['vf_lr_factor']



    def reset(self):
        self.weight[...] = 0.0

        for stat in self.stats.keys():
            self.stats[stat] = None


    def end_episode(self, finished_envs: torch.Tensor):
        for env_idx, finished in enumerate(finished_envs):
            if not finished:
                continue

            if self.experience.next_step_index==self.experience.ep_start_index[env_idx]:
                raise Exception("policy_updater.end_episode must be called before experience_buffer.end_episode!")

            self._end_env_episode(env_idx)


    def buffer_full(self, last_state_value_estimate: torch.tensor):
        assert(self.experience.next_step_index == self.experience.buffer_size)
        for env_idx in range(self.experience.num_envs):
            #  there is nothing to do if this batch just finished an episode
            if self.experience.ep_start_index[env_idx] == self.experience.next_step_index:
                continue

            self._finish_env_path(env_idx, last_state_value_estimate[env_idx])


    def _vf_update_checkpoint(self):
        result = {
            'value_net_state_dict': {k: v.cpu() for k, v in self.value_net_tail.state_dict().items()},
            'vf_optimizer_state_dict': self.vf_optimizer.state_dict()
        }
        opt_dict = result["vf_optimizer_state_dict"]
        opt_dict["state"] = {k1: {k: v.cpu() for k, v in v1.items()} for k1, v1 in opt_dict["state"].items()}
        return result


    def _load_vf_update_checkpoint(self, checkpoint):
        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.vf_optimizer.load_state_dict(checkpoint['vf_optimizer_state_dict'])

        lr = self.vf_lr.initial_lr * self.vf_lr.factor
        for group in self.vf_optimizer.param_groups:
            group['lr'] = lr


