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
        hidden_sizes: list[int],
        output_activation  = nn.Identity,
        mean_offset: torch.tensor = None,
        min_std: torch.tensor = None,
        std_scale = 2.0
    ) -> None:
        super().__init__(obs_encoder)
        self.action_dist_net_output_shape = action_space.shape + (2, )
        self.action_shape = action_space.shape
        self.mean_offset = mean_offset if mean_offset is not None else torch.zeros(*self.action_shape)
        self.std_scale = std_scale
        self.min_std = min_std if min_std is not None else torch.zeros(*self.action_shape)
        mlp_sizes = list(obs_encoder.shape) + hidden_sizes + [np.prod(np.array(self.action_dist_net_output_shape))]
        self.action_dist_net_tail = mlp(sizes=mlp_sizes, output_activation=output_activation)
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
        return Normal(loc=dist_mean + self.mean_offset, scale=dist_std)


    def to(self, device):
        super().to(device)

        self.mean_offset = self.mean_offset.to(device)
        self.min_std = self.min_std.to(device)

        return self


    def checkpoint(self):
        return {
            'action_dist_net_state_dict': {k: v.cpu() for k, v in self.action_dist_net.state_dict().items()},
            #'action_dist_std_scale': self.std_scale
        }


    def load_checkpoint(self, checkpoint):
        self.action_dist_net.load_state_dict(checkpoint['action_dist_net_state_dict'])
        self.action_dist_net_tail = list(self.action_dist_net.children())[-1]
        #self.std_scale = checkpoint['action_dist_std_scale']



class PolicyUpdater:
    POLICY_LOSS = "Policy Loss"
    POLICY_LR = "Policy LR"


    def __init__(self,
        cfg: dict,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer
    ) -> None:
        self.cfg = cfg
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
                raise Exception("policy_updater.end_episode must be called before experience_buffer.end_episode!")

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
    VF_EPOCHS = "VF Epochs"
    UPDATE_BALANCE = "Update Balance"
    CLIP_FACTOR = "Clip Factor"
    VALUE_LOSS = "VF Loss"
    LR_FACTOR = "LR Factor"
    POLICY_LR_FACTOR = "Policy LR Factor"
    BACKTRACK_POLICY = "Backtrack Policy"
    BACKTRACK_VF = "Backtrack VF"
    VF_DELTA = "VF Delta"

    MIN_UPDATE_BALANCE = 0.0
    MAX_UPDATE_BALANCE = 1.0
    UPDATE_BALANCE_STEP = 0.01

    def __init__(self,
        cfg: dict,
        policy: ContinuousPolicy,
        experience: ExperienceBuffer,
        hidden_sizes: list,
        vf_lr: float,
        policy_lr_factor: float, #  = 0.1
        update_batch_size: int,
        update_epochs: int,
        lam: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.15,
        target_vf_delta: float = 0.1,
        lr_decay: float = 0.95,
        lr_factor_decay: float = 0.92,
        vf_only_update: bool = False
    ) -> None:
        super().__init__(cfg, policy, experience)

        self.initial_vf_lr = vf_lr
        self.policy_lr_factor = policy_lr_factor
        self.update_epochs = update_epochs
        self.update_batch_size = update_batch_size
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.target_vf_delta = target_vf_delta
        self.lam = lam
        self.lr_factor = 1.0
        self.lr_decay = lr_decay
        self.lr_factor_decay = lr_factor_decay
        self.update_balance = (self.MAX_UPDATE_BALANCE - self.MIN_UPDATE_BALANCE) / 2.0
        self.vf_only_update = vf_only_update

        self.value_net_tail = mlp(list(policy.obs_encoder.shape) + hidden_sizes + [1]).to(cfg.device)

        self._update_lr()

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
        if self.experience.ep_success_info_count == 0 or self.experience.success_rate() == 0.0:
            return

        if self.experience.success_rate() > 0.5:
            self.weight[self.experience.success] *= (1.0 - self.experience.success_rate()) * 2
        else:
            self.weight[torch.logical_not(self.experience.success)] *= self.experience.success_rate() * 2


    def update(self):
        for _ in range(self.update_epochs // 2):
            if self._do_update():
                return

            # Note: Must be done after restoring the checkpoint
            self.lr_factor *= self.lr_decay
            self._update_lr()


    def _update_lr(self):
        vf_lr = self.initial_vf_lr * self.lr_factor
        policy_lr = vf_lr * self.policy_lr_factor

        print(f"====> {self.lr_factor:.6f} / {self.policy_lr_factor:.6f} - {policy_lr:.1e} / {vf_lr:.1e} ")

        self.optimizer = Adam([
            {'params': self.policy.obs_encoder.parameters(), 'lr': max(policy_lr, vf_lr), 'name': "OBS_ENCODER_PARAMS"},
            {'params': self.policy.action_dist_net_tail.parameters(), 'lr': policy_lr, 'name': "POLICY_PARAMS"},
            {'params': self.value_net_tail.parameters(), 'lr': vf_lr, 'name': "VALUE_NET_PARAMS"}
        ])

    def _do_update(self):
        for key in self.stats:
            self.stats[key] = None
        self.stats[self.POLICY_EPOCHS] = 0
        self.stats[self.VF_EPOCHS] = 0
        self.stats[self.BACKTRACK_POLICY] = 0
        self.stats[self.BACKTRACK_VF] = 0
        self.stats[self.LR_FACTOR] = self.lr_factor
        self.stats[self.UPDATE_BALANCE] = self.update_balance
        self.stats[self.POLICY_LR_FACTOR] = self.policy_lr_factor

        counts = np.zeros(4)
        totals = torch.zeros(4, requires_grad=False, **self.cfg.tensor_args)
        policy_loss_total, value_loss_total, kl_total, clip_factor_total = totals[0:1], totals[1:2], totals[2:3], totals[3:4]
        policy_loss_count, value_loss_count, kl_count, clip_factor_count = counts[0:1], counts[1:2], counts[2:3], counts[3:4]
        last_epoch_value_loss = float("inf")
        #last_epoch_kl = 0.0
        value_loss_old = float("-inf")

        cp = None
        for update_epoch in range(self.update_epochs+1):
            is_first_epoch = update_epoch == 0
            is_after_first_update = update_epoch == 1
            is_validate_epoch = update_epoch == self.update_epochs
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
                        self.logp_old[batch_range, ...] = logp_old.cpu()
                else:
                    logp_old = logp_old[0].to(self.cfg.device)

                # Policy update
                # -------------
                # Note: Calculdate even if update_policy == False to check for passive_policy_progression (see below)

                if self.vf_only_update:
                    batch_policy_loss, kl, clip_factor = self._stationary_policy_loss(logp_old, encoded_obs, act, adv)
                else:
                    batch_policy_loss, kl, clip_factor = self._policy_loss(logp_old, encoded_obs, act, adv)

                kl_total += kl.sum().detach()
                kl_count += np.prod(kl.shape).item()
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
                batch_value_loss = self._value_loss(encoded_obs, discounted_reward)
                del discounted_reward
                del encoded_obs

                value_loss_total += batch_value_loss.sum().detach()
                value_loss_count += np.prod(batch_value_loss.shape).item()

                batch_value_loss = batch_value_loss.mean()


                # Backward pass
                # -------------

                loss = batch_policy_loss + batch_value_loss
                if not is_validate_epoch:
                    loss.backward()

                del batch_policy_loss
                del batch_value_loss
                del loss

            # end for: iterate through dataset batches

            # whether to roll back policy and vf to state before last update
            backtrack = False

            policy_done = False
            epoch_kl = abs(kl_total.item() / kl_count).item()
            # kl_decreased = (epoch_kl < last_epoch_kl and not self.vf_only_update) or
            kl_limit_reached = epoch_kl > self.target_kl
            if kl_limit_reached:
                print("->", end="")
                self.stats[self.BACKTRACK_POLICY] = 1.0
                backtrack = True
                policy_done = True
            else:
                print("  ", end="")
            print(f"{epoch_kl:.6f} ", end="")

            vf_done = False
            epoch_value_loss = value_loss_total.item() / value_loss_count.item()
            value_loss_increased = epoch_value_loss > last_epoch_value_loss
            value_update_limit_reached = value_loss_old - epoch_value_loss > self.target_vf_delta
            if value_loss_increased or value_update_limit_reached:
                print("->", end="")
                self.stats[self.BACKTRACK_VF] = 1.0
                backtrack = True
                vf_done = True
            print(f"{epoch_value_loss:.6f}")

            if policy_done and not vf_done:
                self.update_balance = min(self.update_balance + self.UPDATE_BALANCE_STEP, self.MAX_UPDATE_BALANCE)
                self.policy_lr_factor = max(self.policy_lr_factor * self.lr_factor_decay, 0.1)
                if not is_after_first_update:
                    self._update_lr()
            if vf_done and not policy_done:
                self.policy_lr_factor = min(self.policy_lr_factor * (1.0 / self.lr_factor_decay), 10)
                if not is_after_first_update:
                    self._update_lr()
                self.update_balance = max(self.update_balance - self.UPDATE_BALANCE_STEP, self.MIN_UPDATE_BALANCE)

            if backtrack:
                assert(not is_first_epoch)
                # Keep update balance updates from getting reverted by the rollback
                policy_lr_factor_backup = self.policy_lr_factor
                update_balance_backup = self.update_balance
                self._load_full_checkpoint(cp)
                self.policy_lr_factor = policy_lr_factor_backup
                self.update_balance = update_balance_backup

                if not is_after_first_update:
                    self.lr_factor *= self.lr_decay
                    print(f"====> {self.lr_factor:.6f}")
                    for group in self.optimizer.param_groups:
                        group['lr'] = group['lr'] * self.lr_decay

                return not is_after_first_update

            last_epoch_value_loss = epoch_value_loss
            #last_epoch_kl = epoch_kl

            if is_first_epoch:
                value_loss_old = epoch_value_loss
            else:
                self.stats[self.VF_DELTA] = value_loss_old - epoch_value_loss

            if not is_first_epoch:
                self.stats[self.POLICY_EPOCHS] += 1
                self.stats[self.KL] = kl_total.item() / kl_count.item()
                if clip_factor_count != 0:
                    self.stats[self.POLICY_LOSS] = policy_loss_total.item() / policy_loss_count.item()
                    self.stats[self.CLIP_FACTOR] = clip_factor_total.item() / clip_factor_count.item()
            if not is_first_epoch:
                self.stats[self.VF_EPOCHS] += 1
                self.stats[self.VALUE_LOSS] = epoch_value_loss

            if not is_validate_epoch:
                cp = self._full_checkpoint()
                self.optimizer.step()


        # find factor such that applying applying it decay_revert_steps times reverts one decay step
        decay_revert_steps = 3
        factor = (1.0 / self.lr_decay)**(1/decay_revert_steps)

        self.lr_factor *= factor
        print(f"====> {self.lr_factor:.6f}")
        for group in self.optimizer.param_groups:
            group['lr'] = group['lr'] * factor

        return True


    def _stationary_policy_loss(self, logp_old, encoded_obs, act, adv) -> Tuple[torch.tensor, float, float, float]:
        # Policy loss
        action_dist = self.policy.action_dist(encoded_obs=encoded_obs)
        logp = action_dist.log_prob(act).sum(-1)
        loss = 10000.0 * torch.square(logp - logp_old)

        # Stats
        kl = logp_old - logp
        clip_factor = torch.zeros(1)

        return loss, kl, clip_factor


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

        obs_device = self.experience.obs[path_slice, env_idx].to(self.cfg.device)

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
            'value_net_state_dict': {k: v.cpu() for k, v in self.value_net_tail.state_dict().items()},
            'optimizer_state_dict': self.optimizer.state_dict(),
            'initial_vf_lr': self.initial_vf_lr,
            'policy_lr_factor': self.policy_lr_factor,
            'lr_factor': self.lr_factor,
            'update_balance': self.update_balance
        }


    def load_checkpoint(self, checkpoint):
        self.value_net_tail.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.initial_vf_lr = checkpoint['initial_vf_lr']
        self.policy_lr_factor = checkpoint['policy_lr_factor']
        self.lr_factor = checkpoint['lr_factor']
        self.update_balance = checkpoint['update_balance']


    def _full_checkpoint(self):
        full_state = {
            "policy": self.policy.checkpoint(),
            "policy_updater": self.checkpoint(),
        }
        return full_state


    def _load_full_checkpoint(self, checkpoint):
        self.policy.load_checkpoint(checkpoint["policy"])
        self.load_checkpoint(checkpoint["policy_updater"])