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
        self.policy_initial_lr = policy_initial_lr
        self.policy_lr_factor = 1.0
        self.policy_lr_decay = policy_lr_decay
        self.entropy_factor = entropy_factor
        self.policy_kl_lower_min = policy_kl_lower_min
        self.policy_kl_upper_target = policy_kl_upper_target
        self.policy_kl_upper_max = policy_kl_upper_max
        self.policy_valid_kl_lower_bound = policy_valid_kl_lower_bound
        self.policy_shifting_right_way_scale = policy_shifting_right_way_scale

        self.policy_update_epochs = policy_update_epochs
        self.policy_update_batch_size = policy_update_batch_size * experience.num_envs
        self.policy_optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': self.policy_initial_lr},
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': self.policy_initial_lr},
        ])

        # The extra slot at the end of used in _finish_env_path().
        # Note: This is used during experience collection to calculate advanteaged and during
        # the update to calculate loss statistics
        self.value_full = torch.zeros(experience.buffer_size+1, experience.num_envs, dtype=config.dtype)
        self.value = self.value_full[:-1,...]
        self.kl = torch.zeros(experience.buffer_size, experience.num_envs, dtype=config.dtype)
        self.logp_old = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)
        self.weight = torch.zeros(experience.buffer_size, experience.num_envs, **experience.tensor_args)

        self.value_merged = self.value.reshape(-1)
        self.kl_merged = self.kl.reshape(-1)
        self.logp_old_merged = self.logp_old.reshape(-1)
        self.weight_merged = self.weight.reshape(-1)
        self.abs_kl_sign = torch.zeros(experience.buffer_size * experience.num_envs, requires_grad=False, **experience.tensor_args)
        self.encoded_obs = torch.zeros(experience.buffer_size * experience.num_envs, policy.obs_encoder.num_outputs, dtype=config.dtype)

        self.obs_loader = DataLoader(TensorDataset(self.experience.obs_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.action_loader = DataLoader(TensorDataset(self.experience.action_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.weight_loader = DataLoader(TensorDataset(self.weight_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)
        self.logp_old_loader = DataLoader(TensorDataset(self.logp_old_merged), batch_size=self.policy_update_batch_size, pin_memory=cfg.pin)

        vf_update_batch_size = vf_update_batch_size * experience.num_envs
        self.vf_discounted_reward_loader = DataLoader(TensorDataset(experience.discounted_reward_merged), batch_size=vf_update_batch_size, pin_memory=cfg.pin)
        self.vf_encoded_obs_loader = DataLoader(TensorDataset(self.encoded_obs), batch_size=vf_update_batch_size, pin_memory=cfg.pin)

        self.vf_optimizer = VFOptimizer(
            cfg=cfg,
            experience=experience,
            encoded_obs=self.encoded_obs,
            value_merged=self.value_merged,
            obs_loader=self.obs_loader,
            vf_encoded_obs_loader=self.vf_encoded_obs_loader,
            vf_discounted_reward_loader=self.vf_discounted_reward_loader,
            obs_encoder=self.policy.obs_encoder,
            value_net_tail=value_net_tail,
            vf_initial_lr=vf_initial_lr,
            vf_lr_decay=vf_lr_decay,
            vf_update_epochs=vf_update_epochs,
            vf_update_batch_size=vf_update_batch_size,
            stats=self.stats
        )


    def step(self, encoded_obs):
        # Note: Clamped to plausible range
        self.value[self.experience.next_step_index] = self.value_net_tail(encoded_obs).clamp(-1.0, 1.0).squeeze(-1).cpu()


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
        self._update_policy()
        self.vf_optimizer._update_vf()


    def _update_policy(self):
        self.stats[self.POLICY_UPDATE_EPOCHS] = 0
        self.stats[self.POLICY_LR_FACTOR] = self.policy_lr_factor

        policy_loss_total, clip_factor_total = 0, 0
        prev_epoch_abs_kl_upper, prev_epoch_abs_kl_lower = 1.0, 1.0
        has_updated = False
        previous_was_backtrack = False
        backtrack_count = 0
        epochs_since_backtrack = 0
        valid_state_cp = None
        epochs_since_oscilating = 0
        is_backtrack_candidate_state = True
        update_is_perfect = False
        update_was_perfect = False
        consecutive_backtracks = 0

        class Phase(Enum):
            DROP = 1,
            TENTATIVE_RISE = 2,
            RISE = 3

        phase = Phase.DROP

        self.abs_kl_sign[:] = -1.0
        self.abs_kl_sign[self.weight_merged<0] = 1.0

        try:
            backtrack_point = None
            for update_epoch in range(self.policy_update_epochs+1):
                is_first_epoch = update_epoch == 0
                is_last_update_epoch = update_epoch == self.policy_update_epochs-1
                is_validate_epoch = update_epoch == self.policy_update_epochs
                # If update 0, 1 have been successfull, increase lr for update 3 and above
                is_lr_inc_epoch = epochs_since_backtrack >= 2 and epochs_since_oscilating >= 2

                batch_start = 0
                policy_loss_total, clip_factor_total = 0, 0

                self.policy_optimizer.zero_grad(set_to_none=True)
                for obs, act, adv, logp_old in zip(self.obs_loader, self.action_loader, self.weight_loader, self.logp_old_loader):
                    assert(obs[0].shape[0] == self.policy_update_batch_size)
                    batch_range = range(batch_start, batch_start+self.policy_update_batch_size)
                    batch_start += self.policy_update_batch_size

                    # Prepare dataset
                    # ---------------

                    obs = obs[0].to(self.cfg.device, non_blocking=True)
                    act = act[0].to(self.cfg.device, non_blocking=True)
                    adv = adv[0].to(self.cfg.device, non_blocking=True)
                    encoded_obs = self.policy.obs_encoder(obs)
                    del obs

                    if is_first_epoch:
                        # Calculate the pre-update log probs. This is done here to reuse the obs that are already encoded
                        with torch.no_grad():
                            logp_old = self.policy.action_dist(encoded_obs=encoded_obs).log_prob(act).sum(-1)
                            self.logp_old_merged[batch_range] = logp_old.cpu()
                    else:
                        logp_old = logp_old[0].to(self.cfg.device)

                    # Policy update
                    # -------------
                    # Note: Calculdate even if update_policy == False to check for passive_policy_progression (see below)

                    #if self.remaining_warmup_updates > 0:
                    #   batch_policy_loss, kl, clip_factor, entropy = self._stationary_policy_loss(logp_old, encoded_obs, act, adv)
                    #else:
                    batch_policy_loss, policy_net_penality, kl, clip_factor, entropy = self._policy_loss(logp_old, encoded_obs, act, adv)

                    batch_abs_kl = batch_policy_loss * -1.0 * self.abs_kl_sign[batch_range].cuda()
                    batch_policy_loss[batch_abs_kl >= math.log(self.policy_kl_upper_target)] *= -1.0
                    if not update_is_perfect:
                        batch_policy_loss[batch_abs_kl < 0.0] *= self.policy_shifting_right_way_scale
                    del batch_abs_kl

                    self.kl_merged[batch_range] = kl.detach().cpu()
                    policy_loss_total += batch_policy_loss.sum().detach()
                    clip_factor_total += clip_factor.sum().detach()

                    batch_policy_loss = batch_policy_loss.mean()
                    backprop_policy_loss = batch_policy_loss + self.entropy_factor * entropy.mean() + policy_net_penality

                    self.stats["Policy Net Penalty"] = policy_net_penality.item()

                    # free up memory for the value function update
                    del logp_old
                    del encoded_obs
                    del act
                    del adv
                    del kl
                    del clip_factor
                    del entropy
                    del batch_policy_loss
                    del policy_net_penality

                    if not is_validate_epoch:
                        backprop_policy_loss.backward()

                    del backprop_policy_loss
                # end for: iterate through dataset batches

                epoch_kl = self.kl
                epoch_abs_kl = epoch_kl.flatten() * self.abs_kl_sign
                epoch_abs_kl_min = epoch_abs_kl.min().item()
                epoch_abs_kl_mean = epoch_abs_kl.mean().item()
                epoch_abs_kl_std = epoch_abs_kl.std().item()
                epoch_abs_kl_max = epoch_abs_kl.max().item()
                epoch_abs_kl_lower = math.exp(epoch_abs_kl_mean - epoch_abs_kl_std)
                epoch_abs_kl_upper = math.exp(epoch_abs_kl_mean + epoch_abs_kl_std)

                if is_first_epoch:
                    backtrack_point = self._policy_update_checkpoint()
                    backtrack_point["prev_epoch_abs_kl_lower"] = epoch_abs_kl_lower
                    backtrack_point["prev_epoch_abs_kl_upper"] = epoch_abs_kl_upper
                    backtrack_point["update_epoch"] = update_epoch
                    valid_state_cp = backtrack_point
                    self.policy_optimizer.step()

                    continue

                if previous_was_backtrack:
                    # Validation epoch is skipped if the last_upate_epoch was backtracked
                    assert(not is_validate_epoch)
                    self.policy_optimizer.step()
                    previous_was_backtrack = False
                    epochs_since_backtrack = 0

                    #print(f"[{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f}    {self.policy_lr_factor:.3f}]")
                    continue
                else:
                    epochs_since_backtrack += 1

                backtrack = False

                update_is_perfect = epoch_abs_kl_lower >= 1.0 and epoch_abs_kl_upper > 1.0
                kl_lower_out_of_bounds = epoch_abs_kl_lower < self.policy_kl_lower_min
                kl_upper_out_of_bounds = epoch_abs_kl_upper > 1.0 and epoch_abs_kl_upper > self.policy_kl_upper_max
                kl_out_of_bounds = kl_upper_out_of_bounds or kl_lower_out_of_bounds
                is_valid_end_state = epoch_abs_kl_lower >= self.policy_valid_kl_lower_bound and not kl_out_of_bounds
                target_reached = epoch_abs_kl_upper > self.policy_kl_upper_target and is_valid_end_state

                # BEGIN: Check if policy is oscylating
                is_backtrack_candidate_state = True
                policy_oscilating = False
                abs_kl_lower_dropped = epoch_abs_kl_lower < prev_epoch_abs_kl_lower
                abs_kl_lower_rose = not abs_kl_lower_dropped
                if phase == Phase.DROP:
                    if abs_kl_lower_rose:
                        phase = Phase.TENTATIVE_RISE
                        tentative_rise_start_epoch = update_epoch
                elif phase == Phase.TENTATIVE_RISE:
                    is_backtrack_candidate_state = False
                    is_lr_inc_epoch = False

                    if abs_kl_lower_rose and update_epoch - tentative_rise_start_epoch >= 2:
                        phase = Phase.RISE

                    elif abs_kl_lower_dropped:
                        phase = Phase.DROP
                        policy_oscilating = True

                elif phase == Phase.RISE:
                    if abs_kl_lower_dropped:
                        policy_oscilating = True

                # ---
                if phase == Phase.DROP:
                    print(f"↓ ", end="")
                elif phase == Phase.TENTATIVE_RISE:
                    print(f"? ", end="")
                elif phase == Phase.RISE:
                    print(f"↑ ", end="")
                # END: Check if policy is oscylating
                is_valid_end_state = is_valid_end_state and not policy_oscilating and phase == Phase.RISE

                is_lr_inc_epoch = is_lr_inc_epoch and phase == Phase.RISE and (not update_is_perfect or update_was_perfect)
                if update_is_perfect and not update_was_perfect:
                    self._dec_policy_lr_factor()
                update_was_perfect = update_was_perfect or update_is_perfect

                if is_valid_end_state:
                    print(f"{update_epoch:2}: *{epoch_abs_kl_lower:.4f} {epoch_abs_kl_upper:.4f}   {math.exp(epoch_abs_kl_min):.4f} {math.exp(epoch_abs_kl_max):.4f}", end="")
                else:
                    print(f"{update_epoch:2}:  {epoch_abs_kl_lower:.4f} {epoch_abs_kl_upper:.4f}   {math.exp(epoch_abs_kl_min):.4f} {math.exp(epoch_abs_kl_max):.4f}", end="")

                if kl_out_of_bounds:
                    if kl_upper_out_of_bounds:
                        self._dec_policy_lr_factor(decy_steps = 1.0)
                        print(f"   ↓{self.policy_lr_factor:.3f}")
                    else:
                        self._dec_policy_lr_factor(decy_steps = 2.0)
                        print(f"  ↓↓{self.policy_lr_factor:.3f}")
                    break

                elif policy_oscilating:
                    self._dec_policy_lr_factor(decy_steps = 1.0)
                    backtrack = True
                    epochs_since_oscilating = -1
                    print(f"   ~{self.policy_lr_factor:.3f}")

                elif is_lr_inc_epoch:
                    self._inc_policy_lr_factor()
                    print(f"    {self.policy_lr_factor:.3f}↑")

                else:
                    print(f"    {self.policy_lr_factor:.3f}")

                epochs_since_oscilating+=1
                self.stats[self.POLICY_LR_FACTOR] = self.policy_lr_factor

                previous_was_backtrack = backtrack
                if backtrack:
                    self._load_policy_update_checkpoint(backtrack_point)
                    prev_epoch_abs_kl_lower = backtrack_point["prev_epoch_abs_kl_lower"]
                    prev_epoch_abs_kl_upper = backtrack_point["prev_epoch_abs_kl_upper"]
                    print(f"BT: {backtrack_point['update_epoch']}")
                    backtrack_count += 1
                    consecutive_backtracks += 1

                    if consecutive_backtracks>=3 or is_last_update_epoch:
                        self._inc_policy_lr_factor(decay_revert_steps=2)
                        # In the next epoch only statitics will be updated, but no parameters. This
                        # is not necessary if we backtracked just before that
                        return
                    else:
                        continue
                consecutive_backtracks = 0

                if not is_validate_epoch:
                    if is_backtrack_candidate_state:
                        backtrack_point = self._policy_update_checkpoint()
                        backtrack_point["prev_epoch_abs_kl_lower"] = epoch_abs_kl_lower
                        backtrack_point["prev_epoch_abs_kl_upper"] = epoch_abs_kl_upper
                        backtrack_point["update_epoch"] = update_epoch
                    self.policy_optimizer.step()

                prev_epoch_abs_kl_upper = epoch_abs_kl_upper
                prev_epoch_abs_kl_lower = epoch_abs_kl_lower

                if is_valid_end_state and is_backtrack_candidate_state:
                    has_updated = True
                    valid_state_cp = backtrack_point
                    self.stats[self.POLICY_UPDATE_EPOCHS] += 1
                    self.stats[self.ABS_KL_LOWER] = epoch_abs_kl_lower
                    self.stats[self.ABS_KL_UPPER] = epoch_abs_kl_upper
                    self.stats[self.ABS_KL_MEAN] = math.exp(epoch_abs_kl_mean)
                    self.stats[self.ABS_KL_STD] = math.exp(epoch_abs_kl_std)
                    self.stats[self.ABS_KL_MIN] = math.exp(epoch_abs_kl_min)
                    self.stats[self.ABS_KL_MAX] = math.exp(epoch_abs_kl_max)
                    self.stats[self.POLICY_LOSS] = policy_loss_total.item() / self.experience.buffer_size
                    self.stats[self.CLIP_FACTOR] = clip_factor_total.item() / self.experience.buffer_size

                if target_reached and has_updated:
                    break
        finally:
            if not is_valid_end_state:
                self._load_policy_update_checkpoint(valid_state_cp)
                print(f"BT: {valid_state_cp['update_epoch']}")
            else:
                self._inc_policy_lr_factor()
            if not has_updated:
                print("!***")
                self._dec_policy_lr_factor(decy_steps = 2.0)



    def _policy_loss(self, logp_old, encoded_obs, act, adv) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_ratio = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio)
        clip_adv = clip_ratio * adv
        loss = -(torch.min(ratio * adv, clip_adv))

        # policy net penality
        policy_net_penalty = self.policy.action_dist_net_tail(encoded_obs)
        policy_net_penalty = torch.square(policy_net_penalty)
        in_bounds = policy_net_penalty <= 1.2 * 1.2
        policy_net_penalty[in_bounds] = 0.0
        policy_net_penalty[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        output_penality_sum = torch.mean(policy_net_penalty)

        # Stats
        kl = logp_old - logp
        entropy = action_dist.entropy()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clip_factor = torch.as_tensor(clipped, dtype=torch.float32)

        return loss, output_penality_sum, kl, clip_factor, entropy


    def _inc_policy_lr_factor(self, decay_revert_steps = 5):
        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        factor = (1.0 / self.policy_lr_decay)**(1/decay_revert_steps)

        self.policy_lr_factor *= factor
        lr = self.policy_initial_lr * self.policy_lr_factor
        for group in self.policy_optimizer.param_groups:
            group['lr'] = lr


    def _dec_policy_lr_factor(self, decy_steps = 1.0):
        self.policy_lr_factor *= self.policy_lr_decay**decy_steps
        lr = self.policy_initial_lr * self.policy_lr_factor
        for group in self.policy_optimizer.param_groups:
            group['lr'] = lr

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

        vals = self.value_full[path_slice_plus_one, env_idx].numpy()
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

        lr = self.policy_initial_lr * self.policy_lr_factor
        for group in self.policy_optimizer.param_groups:
            group['lr'] = lr

    def checkpoint(self):
        return {
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'initial_lr': self.policy_initial_lr,
            'lr_factor': self.policy_lr_factor,

            'value_net_state_dict': {k: v.cpu() for k, v in self.value_net_tail.state_dict().items()},
            'vf_optimizer_state_dict': self.vf_optimizer.vf_optimizer.state_dict(),
            'vf_initial_lr': self.vf_optimizer.vf_initial_lr,
            'vf_lr_factor': self.vf_optimizer.vf_lr_factor,
        }


    def load_checkpoint(self, checkpoint):
        self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.policy_initial_lr = checkpoint['initial_lr']
        self.policy_lr_factor = checkpoint['lr_factor']

        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.vf_optimizer.vf_optimizer.load_state_dict(checkpoint['vf_optimizer_state_dict'])
        self.vf_optimizer.vf_initial_lr = checkpoint['vf_initial_lr']
        self.vf_optimizer.vf_lr_factor = checkpoint['vf_lr_factor']



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


class VFOptimizer():
    def __init__(self,
        cfg,
        experience,
        encoded_obs,
        value_merged,
        obs_loader,
        vf_encoded_obs_loader,
        vf_discounted_reward_loader,
        obs_encoder,
        value_net_tail,
        vf_initial_lr,
        vf_lr_decay,
        vf_update_epochs,
        vf_update_batch_size,
        stats,
    ) -> None:
        self.cfg = cfg
        self.stats = stats
        self.experience = experience
        self.encoded_obs = encoded_obs
        self.value_merged = value_merged
        self.obs_loader = obs_loader
        self.vf_encoded_obs_loader = vf_encoded_obs_loader
        self.vf_discounted_reward_loader = vf_discounted_reward_loader
        self.obs_encoder = obs_encoder
        self.value_net_tail = value_net_tail
        self.vf_initial_lr = vf_initial_lr
        self.vf_lr_factor = 1.0
        self.vf_lr_decay = vf_lr_decay
        self.vf_update_epochs = vf_update_epochs
        self.vf_update_batch_size = vf_update_batch_size
        self.vf_optimizer = Adam([
            {'params': self.value_net_tail.parameters(), 'lr': self.vf_initial_lr}
        ])

        suffix = ""
        self.VF_UPDATE_EPOCHS = f"VF Update Epochs {suffix}"
        self.VF_LR_FACTOR = f"VF LR Factor {suffix}"
        self.VALUE_LOSS_IN_MEAN = f"VF Loss Mean {suffix}"
        self.VALUE_LOSS_IN_STD = f"VF Loss Std {suffix}"
        self.VALUE_LOSS_IN_MAX = f"VF Loss Max {suffix}"
        self.VF_DELTA = f"VF Delta {suffix} %"
        self.VALUE_NET_PENALTY = f"Value Net Penality {suffix}"


    def _update_vf(self):
        self.stats[self.VF_UPDATE_EPOCHS] = 0
        self.stats[self.VF_DELTA] = 0.0
        self.stats[self.VF_LR_FACTOR] = self.vf_lr_factor

        value_loss_mean_old = float("-inf")
        previous_was_backtrack = False
        backtrack_count = 0
        epochs_since_backtrack = 0
        consecutive_backtracks = 0

        batch_start = 0
        for obs in self.obs_loader:
            batch_range = range(batch_start, batch_start+self.obs_loader.batch_size)
            batch_start += self.obs_loader.batch_size

            obs = obs[0].to(self.cfg.device, non_blocking=True)

            with torch.no_grad():
                encoded_obs = self.obs_encoder(obs)
                self.encoded_obs[batch_range] = encoded_obs.detach().cpu()
            del obs
        assert(batch_start == self.encoded_obs.shape[0])

        backtrack_point = None
        for update_epoch in range(self.vf_update_epochs+1):
            is_first_epoch = update_epoch == 0
            is_last_update_epoch = update_epoch == self.vf_update_epochs-1
            is_validate_epoch = update_epoch == self.vf_update_epochs
            # If update 0, 1 have been successfull, increase lr for update 3 and above
            is_lr_inc_epoch = epochs_since_backtrack >= 2
            batch_start = 0
            self.vf_optimizer.zero_grad(set_to_none=True)
            for encoded_obs, discounted_reward in zip(self.vf_encoded_obs_loader, self.vf_discounted_reward_loader):
                assert(encoded_obs[0].shape[0] == self.vf_update_batch_size)
                batch_range = range(batch_start, batch_start+self.vf_update_batch_size)
                batch_start += self.vf_update_batch_size

                encoded_obs = encoded_obs[0].to(self.cfg.device, non_blocking=True)
                discounted_reward = discounted_reward[0].to(self.cfg.device, non_blocking=True)

                batch_value_loss, value_net_penality = self._value_loss(encoded_obs, discounted_reward)

                # If we have a lot more successes than failures (or opposite) in the buffer scale down the weight of the successes such
                # that the underepresented case gets a stronger influence on the gradient
                #if self.experience.success_rate() > 0.5:
                    # Value function predicted success: Advantage is < 1.0 and made smaller
                    # Value function predicted failure: Advantage is > 1.7 and made smaller
                #    batch_value_loss[self.experience.success.flatten()[batch_range]] *= max((1.0 - self.experience.success_rate()) * 2.0, 0.20)
                #else:
                #    batch_value_loss[torch.logical_not(self.experience.success.flatten()[batch_range])] *= max(self.experience.success_rate() * 2.0, 0.20)

                self.value_merged[batch_range] = batch_value_loss.detach().cpu() #TODO: Gpu?

                backprop_value_loss = batch_value_loss.mean() #+ value_net_penality
                if not is_validate_epoch:
                    backprop_value_loss.backward()

                self.stats[self.VALUE_NET_PENALTY] = value_net_penality.item()

                del discounted_reward
                del encoded_obs
                del value_net_penality
                del batch_value_loss
                del backprop_value_loss
            batch_start == self.value_merged.shape[0]
            # end for: iterate through dataset batches

            epoch_value_loss = self.value_merged
            epoch_value_loss_mean = epoch_value_loss.mean().item()

            if previous_was_backtrack:
                # Validation epoch is skipped if the last_upate_epoch was backtracked
                assert(not is_validate_epoch)
                self.vf_optimizer.step()
                previous_was_backtrack = False
                epochs_since_backtrack = 0

                #print(f"[{epoch_value_loss_mean:.4f}    {self.vf_lr_factor:.3f}]")
                continue
            epochs_since_backtrack += 1


            print(f"{epoch_value_loss_mean:.4f}", end="")

            if is_first_epoch:
                self.stats[self.VALUE_LOSS_IN_MEAN] = epoch_value_loss_mean
                self.stats[self.VALUE_LOSS_IN_STD] = epoch_value_loss.std().item()
                self.stats[self.VALUE_LOSS_IN_MAX] = epoch_value_loss.max().item()
                value_loss_mean_old = epoch_value_loss_mean
                last_epoch_value_loss_mean = epoch_value_loss_mean

                backtrack_point = self._vf_update_checkpoint()
                self.vf_optimizer.step()

                print(f"    {self.vf_lr_factor:.3f}")
                continue

            vf_delta = (1.0 - epoch_value_loss_mean / value_loss_mean_old) * 100.0
            vf_oscilating = last_epoch_value_loss_mean - epoch_value_loss_mean < 0.000
            vf_max_reached = epoch_value_loss_mean < 0.2

            backtrack = vf_oscilating
            previous_was_backtrack = backtrack

            if vf_oscilating:
                self._dec_vf_lr_factor(decy_steps = 1.0)
                print(f"   ↓{self.vf_lr_factor:.3f}")

            elif vf_max_reached:
                print(f"*   {self.vf_lr_factor:.3f}")

                break

            elif is_lr_inc_epoch:
                self._inc_vf_lr_factor()
                print(f"    {self.vf_lr_factor:.3f}↑")

            else:
                print(f"    {self.vf_lr_factor:.3f}")

            self.stats[self.VF_LR_FACTOR] = self.vf_lr_factor

            previous_was_backtrack = backtrack
            if backtrack:
                self._load_vf_update_checkpoint(backtrack_point)
                backtrack_count += 1
                consecutive_backtracks += 1

                if is_last_update_epoch:
                    # In the next epoch only statitics will be updated, but no parameters. This
                    # is not necessary if we backtracked just before that
                    return

                if consecutive_backtracks>=self.vf_update_epochs // 2:
                    return
                else:
                    continue
            consecutive_backtracks = 0

            if not is_validate_epoch:
                backtrack_point = self._vf_update_checkpoint()
                self.vf_optimizer.step()

            last_epoch_value_loss_mean = epoch_value_loss_mean

            self.stats[self.VF_UPDATE_EPOCHS] += 1
            self.stats[self.VF_DELTA] = vf_delta


    def _value_loss(self, encoded_obs, discounted_reward):
        value_net_out = self.value_net_tail(encoded_obs).squeeze(-1)

        value_net_penality = torch.square(value_net_out)
        in_bounds = value_net_penality <= 1.2 * 1.2
        value_net_penality[in_bounds] = 0.0
        value_net_penality[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        value_net_penality_sum = torch.sum(value_net_penality)

        return torch.square(value_net_out - discounted_reward), value_net_penality_sum


    def _inc_vf_lr_factor(self):
        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        decay_revert_steps = 2
        factor = (1.0 / self.vf_lr_decay)**(1/decay_revert_steps)

        self.vf_lr_factor *= factor
        lr = self.vf_initial_lr * self.vf_lr_factor
        for group in self.vf_optimizer.param_groups:
            group['lr'] = lr


    def _dec_vf_lr_factor(self, decy_steps = 1.0):
        self.vf_lr_factor = max(self.vf_lr_factor * (self.vf_lr_decay**decy_steps), 0.01)
        lr = self.vf_initial_lr * self.vf_lr_factor
        for group in self.vf_optimizer.param_groups:
            group['lr'] = lr


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

        lr = self.vf_initial_lr * self.vf_lr_factor
        for group in self.vf_optimizer.param_groups:
            group['lr'] = lr


