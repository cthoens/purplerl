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
    POLICY_LOSS = "Policy Loss"
    LR = "LR"
    LR_FACTOR = "LR Factor"
    KL_MEAN = "KL"
    KL_STD = "KL Std"
    KL_MAX = "KL Max"
    VALUE_LOSS_IN_MEAN = "VF Loss In"
    VALUE_LOSS_IN_STD = "VF Loss In Std"
    VALUE_LOSS_IN_MAX = "VF Loss In Max"
    VF_DELTA = "VF Delta"
    UPDATE_EPOCHS = "Update Epochs"
    VF_PRIORITY = "VF Priority"
    CLIP_FACTOR = "Clip Factor"
    BACKTRACK_POLICY = "Backtrack Policy"
    BACKTRACK_VF = "Backtrack VF"

    def __init__(self,
        cfg: dict,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer,
        value_net_tail: torch.nn.Module,
        initial_lr: float,
        update_batch_size: int,
        update_epochs: int,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.15,
        target_vf_decay: float = 0.1,
        lr_decay: float = 0.95,
        entropy_factor: float = 0.0,
        max_vf_priority: float = 4.0,
        min_vf_priority: float = 1.0/4.0,
        warmup_updates: int = 0
    ) -> None:
        self.cfg = cfg
        self.policy = policy
        self.experience = experience
        self.initial_lr = initial_lr
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size * experience.num_envs
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.target_vf_decay = target_vf_decay
        self.lam = lam
        self.lr_factor = 1.0
        self.lr_decay = lr_decay
        self.entropy_factor = entropy_factor
        self.max_vf_priority = max_vf_priority
        self.min_vf_priority = min_vf_priority
        self.vf_priority_step = pow(min_vf_priority / max_vf_priority, 1.0/14.0) #=> 14 steps to get from max to min_vf_priority
        self.vf_priority = max_vf_priority
        self.remaining_warmup_updates = warmup_updates
        self.stats = {}

        self.value_net_tail = value_net_tail
        self.optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': self.initial_lr},
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': self.initial_lr},
            {'params': self.value_net_tail.parameters(), 'lr': self.initial_lr}
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
        self.abs_kl_sign = torch.zeros(experience.buffer_size * experience.num_envs, **experience.tensor_args)

        self.obs_loader = DataLoader(TensorDataset(self.experience.obs_merged), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.action_loader = DataLoader(TensorDataset(self.experience.action_merged), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.discounted_reward_loader = DataLoader(TensorDataset(self.experience.discounted_reward_merged), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.weight_loader = DataLoader(TensorDataset(self.weight_merged), batch_size=self.update_batch_size, pin_memory=cfg.pin)
        self.logp_old_loader = DataLoader(TensorDataset(self.logp_old_merged), batch_size=self.update_batch_size, pin_memory=cfg.pin)


    def step(self, encoded_obs):
        self.value[self.experience.next_step_index] = self.value_net_tail(encoded_obs).squeeze(-1).cpu()


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
        self._do_update()


    def _do_update(self):
        for key in self.stats:
            self.stats[key] = None
        self.stats[self.UPDATE_EPOCHS] = 0
        self.stats[self.BACKTRACK_POLICY] = 0
        self.stats[self.BACKTRACK_VF] = 0
        self.stats[self.LR_FACTOR] = self.lr_factor
        self.stats[self.VF_PRIORITY] = self.vf_priority

        counts = np.zeros(2)
        totals = torch.zeros(2, requires_grad=False, **self.cfg.tensor_args)
        policy_loss_total, clip_factor_total = totals[0:1], totals[1:2]
        policy_loss_count, clip_factor_count = counts[0:1], counts[1:2]
        last_abs_epoch_kl = 0.0
        previous_was_backtrack = False
        backtrack_count = 0
        # => if the lr we previously increased in this update, roll back the increase
        lr_was_increased = False
        #last_epoch_kl = 0.0
        value_loss_mean_old = float("-inf")
        # Might have changed because of resume
        self.vf_priority = np.clip(self.vf_priority, self.min_vf_priority, self.max_vf_priority).item()
        is_warmup_update = self.remaining_warmup_updates > 0
        epochs_since_backtrack = 0

        backtrack_point = None
        for update_epoch in range(self.update_epochs+1):
            is_first_epoch = update_epoch == 0
            is_last_update_epoch = update_epoch == self.update_epochs-1
            is_validate_epoch = update_epoch == self.update_epochs
            # => if update 0, or 1 is rolled back, decrease the rl
            is_lr_dec_epoch = update_epoch <= 2
            # If update 0, 1 have been successfull, increase lr for update 3 and above
            is_lr_inc_epoch = epochs_since_backtrack >= 2
            batch_start = 0
            totals[...] = 0.0
            counts[...] = 0
            self.optimizer.zero_grad(set_to_none=True)
            for obs, act, adv, logp_old, discounted_reward in zip(self.obs_loader, self.action_loader, self.weight_loader, self.logp_old_loader, self.discounted_reward_loader):
                assert(obs[0].shape[0] == self.update_batch_size)
                batch_range = range(batch_start, batch_start+self.update_batch_size)
                batch_start += self.update_batch_size

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

                self.kl_merged[batch_range] = kl.detach().cpu()
                policy_loss_total += batch_policy_loss.sum().detach()
                policy_loss_count += np.prod(batch_policy_loss.shape).item()
                clip_factor_total += clip_factor.sum().detach()
                clip_factor_count += np.prod(clip_factor.shape).item()

                batch_policy_loss = batch_policy_loss.mean()

                # free up memory for the value function update
                del kl
                del clip_factor
                del act
                del adv
                del logp_old

                # Value function update
                # ---------------------
                # Note: Calculdate stats even if update_vf == False to check for passive_vf_progression (see below)

                discounted_reward = discounted_reward[0].to(self.cfg.device, non_blocking=True)
                batch_value_loss, value_net_penality = self._value_loss(encoded_obs, discounted_reward)
                del discounted_reward
                del encoded_obs

                self.value_merged[batch_range] = batch_value_loss.detach().cpu()
                batch_value_loss = batch_value_loss.mean()


                # Backward pass
                # -------------

                #vf_loss_factor = torch.abs(batch_policy_loss.detach() / batch_value_loss.detach())
                vf_loss_factor = self.vf_priority

                loss = batch_policy_loss + vf_loss_factor * batch_value_loss + policy_net_penality + value_net_penality + self.entropy_factor * entropy.mean()
                if not is_validate_epoch:
                    loss.backward()

                self.stats["Policy Net Penalty"] = policy_net_penality.item()
                self.stats["Value Net Penality"] = value_net_penality.item()

                del batch_policy_loss
                del batch_value_loss
                del policy_net_penality
                del value_net_penality
                del entropy
                del loss
            # end for: iterate through dataset batches

            epoch_value_loss = self.value
            epoch_value_loss_mean = epoch_value_loss.mean().item()

            if is_first_epoch:
                self.stats[self.VALUE_LOSS_IN_MEAN] = epoch_value_loss_mean
                self.stats[self.VALUE_LOSS_IN_STD] = epoch_value_loss.std().item()
                self.stats[self.VALUE_LOSS_IN_MAX] = epoch_value_loss.max().item()
                value_loss_mean_old = epoch_value_loss_mean
                last_epoch_value_loss_mean = epoch_value_loss_mean
                last_abs_epoch_kl = 0.0

                backtrack_point = self._full_checkpoint()
                self.optimizer.step()

                print(f"                --  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f}  ===  {self.vf_priority:.3f}")
                continue

            self.abs_kl_sign[:] = -1.0
            self.abs_kl_sign[self.weight_merged<0] = 1.0
            self.abs_kl_sign[self.experience.discounted_reward_merged<0] = 0

            epoch_kl = self.kl
            epoch_kl_mean = epoch_kl.mean().item()
            epoch_kl_std = epoch_kl.std().item()
            epoch_abs_kl = epoch_kl.flatten() * self.abs_kl_sign
            epoch_abs_kl_delta = epoch_abs_kl - last_abs_epoch_kl
            epoch_abs_kl_mean = epoch_abs_kl_delta.min().item()

            if previous_was_backtrack:
                # Validation epoch is skipped if the last_upate_epoch was backtracked
                assert(not is_validate_epoch)
                self.optimizer.step()
                previous_was_backtrack = False
                epochs_since_backtrack = 0

                #print(f"[{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f}  --  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f}  ===  {self.vf_priority:.3f}]")
                continue

            epochs_since_backtrack += 1
            vf_delta = value_loss_mean_old - epoch_value_loss_mean

            policy_oscilating = (epoch_abs_kl_delta < -0.05).any()
            vf_oscilating = epoch_value_loss_mean > last_epoch_value_loss_mean
            kl_reached_target = epoch_kl_mean + epoch_kl_std > self.target_kl
            kl_reached_max = epoch_kl_mean + epoch_kl_std > self.target_kl * 1.15

            backtrack = False
            if policy_oscilating and vf_oscilating:
                # => the network weights moved too much to stay in the region of descend
                # => decrease the learning rate
                self._dec_lr_factor()
                backtrack = True

                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f} <--> {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f} ↓===  {self.vf_priority:.3f}")

            elif vf_oscilating:
                # => the influence of the policy update on the gradient was too strong
                # => shift vf_priority toward the value loss (increase)
                backtrack = True

                self._inc_vf_priority()
                self._dec_lr_factor(decy_steps = 0.5)

                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f}  --> {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f} ↓===↑ {self.vf_priority:.3f}")


            elif policy_oscilating:
                # => the influence of the value function update on the gradient was too strong
                # => shift vf_priority toward the policy loss (decrease)
                backtrack = True

                self._dec_vf_priority()
                self._dec_lr_factor(decy_steps = 0.5)

                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f} <--  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f} ↓===↓ {self.vf_priority:.3f}")

            elif kl_reached_max:
                self._inc_vf_priority()
                self._dec_lr_factor(decy_steps = 0.5)

                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f} !--  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f} ↓===↑ {self.vf_priority:.3f}")
                backtrack = True
            elif kl_reached_target:
                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f} *--  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f}  ===  {self.vf_priority:.3f}")
                break

            elif is_lr_inc_epoch:
                self._inc_lr_factor()

                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f}  --  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f} ↑===  {self.vf_priority:.3f}")
            else:
                print(f"{epoch_kl_mean + epoch_kl_std:.4f} {epoch_abs_kl_mean:.4f}  --  {epoch_value_loss_mean:.4f}    {self.lr_factor:.3f}  ===  {self.vf_priority:.3f}")

            self.stats[self.LR_FACTOR] = self.lr_factor
            self.stats[self.VF_PRIORITY] = self.vf_priority

            previous_was_backtrack = backtrack
            if backtrack:
                # Keep vf_priority updates from getting reverted by the rollback
                lr_factor_backup = self.lr_factor
                vf_priority_backup = self.vf_priority
                self._load_full_checkpoint(backtrack_point)
                self.lr_factor = lr_factor_backup
                self.vf_priority = vf_priority_backup
                self._restore_lr()

                backtrack_count += 1

                if backtrack_count==2 or is_last_update_epoch:
                    # In the next epoch only statitics will be updated, but no parameters. This
                    # is not necessary if we backtracked just before that
                    return
                else:
                    continue

            last_epoch_value_loss_mean = epoch_value_loss_mean
            last_abs_epoch_kl = epoch_abs_kl

            self.stats[self.UPDATE_EPOCHS] += 1
            self.stats[self.VF_DELTA] = vf_delta
            self.stats[self.KL_MEAN] = epoch_kl_mean
            self.stats[self.KL_STD] = epoch_kl_std
            self.stats[self.KL_MAX] = epoch_kl.max().item()
            if clip_factor_count != 0:
                self.stats[self.POLICY_LOSS] = policy_loss_total.item() / policy_loss_count.item()
                self.stats[self.CLIP_FACTOR] = clip_factor_total.item() / clip_factor_count.item()

            #if self.remaining_warmup_updates>0:
            #    print(f"* {self.remaining_warmup_updates}")
            self.remaining_warmup_updates = max(self.remaining_warmup_updates - 1, 0)

            if not is_validate_epoch:
                backtrack_point = self._full_checkpoint()
                self.optimizer.step()


    def _inc_lr_factor(self):
        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        decay_revert_steps = 3
        factor = (1.0 / self.lr_decay)**(1/decay_revert_steps)

        self.lr_factor *= factor
        lr = self.initial_lr * self.lr_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    def _rollback_inc_lr_factor(self):
        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        decay_revert_steps = 3
        factor = (1.0 / self.lr_decay)**(1/decay_revert_steps)
        factor = 1.0 / factor

        self.lr_factor *= factor
        lr = self.initial_lr * self.lr_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    def _restore_lr(self):
        lr = self.initial_lr * self.lr_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    def _dec_lr_factor(self, decy_steps = 1.0):
        self.lr_factor *= self.lr_decay**decy_steps
        lr = self.initial_lr * self.lr_factor
        for group in self.optimizer.param_groups:
            group['lr'] = lr


    def _reset_lr_momentum(self):
        lr = self.initial_lr * self.lr_factor

        # reset the optimizer to make sure momentum does not keep driving
        # the parameters into the wrong direction
        self.optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': lr},
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': lr},
            {'params': self.value_net_tail.parameters(), 'lr': lr}
        ])


    def _inc_vf_priority(self, decay_revert_steps = 1):
        factor = (1.0 / self.vf_priority_step)**(1/decay_revert_steps)
        self.vf_priority = min(self.vf_priority * factor, self.max_vf_priority)


    def _dec_vf_priority(self):
        self.vf_priority = max(self.vf_priority * self.vf_priority_step, self.min_vf_priority)


    def _stationary_policy_loss(self, logp_old, encoded_obs, act, adv) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        loss = 10000.0 * torch.square(logp - logp_old)

        # Stats
        kl = logp_old - logp
        clip_factor = torch.zeros(1)
        entropy = torch.zeros(0)

        return loss, kl, clip_factor, entropy


    def _policy_loss(self, logp_old, encoded_obs, act, adv) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1.0-self.clip_ratio, 1.0+self.clip_ratio) * adv
        loss = -(torch.min(ratio * adv, clip_adv))

        # policy net penality
        policy_net_penalty = self.policy.action_dist_net_tail(encoded_obs)
        policy_net_penalty = torch.square(policy_net_penalty)
        in_bounds = policy_net_penalty <= 1.2 * 1.2
        policy_net_penalty[in_bounds] = 0.0
        policy_net_penalty[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        output_penality_sum = torch.sum(policy_net_penalty)

        # Stats
        kl = logp_old - logp
        entropy = action_dist.entropy()
        clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
        clip_factor = torch.as_tensor(clipped, dtype=torch.float32)

        return loss, output_penality_sum, kl, clip_factor, entropy


    def value_estimate(self, encoded_obs):
        return self.value_net_tail(encoded_obs).squeeze(-1)


    def _value_loss(self, encoded_obs, discounted_reward):
        value_net_out = self.value_net_tail(encoded_obs).squeeze(-1)

        value_net_penality = torch.square(value_net_out)
        in_bounds = value_net_penality <= 1.2 * 1.2
        value_net_penality[in_bounds] = 0.0
        value_net_penality[torch.logical_not(in_bounds)] -= 1.2 * 1.2
        value_net_penality_sum = torch.sum(value_net_penality)

        return torch.square(value_net_out - discounted_reward), value_net_penality_sum


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
        weight = discount_cumsum(deltas, self.experience.discount * self.lam)
        self.weight[path_slice, env_idx] = torch.tensor(np.array(weight), **self.experience.tensor_args)


    def checkpoint(self):
        return {
            'value_net_state_dict': {k: v.cpu() for k, v in self.value_net_tail.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'initial_lr': self.initial_lr,
            'lr_factor': self.lr_factor,
            'remaining_warmup_updates': self.remaining_warmup_updates,
            'vf_priority': self.vf_priority,
            'remaining_warmup_updates': self.remaining_warmup_updates
        }


    def load_checkpoint(self, checkpoint):
        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.initial_lr = checkpoint['initial_lr']
        self.lr_factor = checkpoint['lr_factor']
        self.vf_priority = checkpoint.get('vf_priority', self.vf_priority)
        self.remaining_warmup_updates = checkpoint.get('remaining_warmup_updates', 0)


    def _full_checkpoint(self):
        full_state = {
            "policy": self.policy.checkpoint(),
            "policy_updater": self.checkpoint(),
        }
        return full_state


    def _load_full_checkpoint(self, checkpoint):
        self.policy.load_checkpoint(checkpoint["policy"])
        self.load_checkpoint(checkpoint["policy_updater"])


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