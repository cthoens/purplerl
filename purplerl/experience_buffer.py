import numpy as np
import torch

import scipy.signal


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


class ExperienceBuffer:
    MEAN_RETURN = "Mean Return"
    DISC_REWARD = "Disc Reward"
    SUCCESS_RATE = "Success Rate"
    PARTIAL_SUCCESS_RATE = "Part. Success Rate"
    PARTIAL_MEAN_RETURN = "Part. Mean Return"
    EPISODE_COUNT = "Ep Count"

    def __init__(self,
        num_envs:int,
        buffer_size: int,
        obs_shape: list[int],
        act_shape: list[int],
        discount: float = 0.99,
        tensor_args = {}
    ) -> None:
        # Saved parameters
        self.discount = discount
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.tensor_args = tensor_args
        self.stats = {
            self.MEAN_RETURN: 0.0,
            self.PARTIAL_MEAN_RETURN: 0.0,
            self.DISC_REWARD: 0.0,
            self.SUCCESS_RATE: 0.0,
            self.PARTIAL_SUCCESS_RATE: 0.0,
            self.EPISODE_COUNT: 0
        }

        # Overall statistics
        self.ep_count = 0
        self.next_step_index = 0
        self.ep_success_count = 0 # Number of successfully completed episodes
        self.ep_success_info_count = 0 # Number of times we have received info about success of failure of an episode
        self.ep_return =  np.zeros(num_envs * buffer_size, dtype=np.float32)

        self.update_partial_stats = True
        self.ep_partial_success_count = 0
        self.ep_partial_success_info_count = 0
        self.ep_partial_return_count = 0
        self.ep_partial_return_info_count = 0

        # Per bach data
        self.ep_start_index = np.zeros(num_envs, dtype=np.int32)
        self.ep_cum_reward = np.zeros(num_envs, dtype=np.float32)

        # Stored data
        self.obs = torch.zeros(buffer_size, num_envs, *obs_shape, **self.tensor_args)
        self.action = torch.zeros(buffer_size, num_envs, *act_shape, **self.tensor_args)
        # The extra slot at the end of used in buffer_full()
        self.step_reward_full = np.zeros((buffer_size+1, num_envs), dtype=np.float32)
        self.step_reward = self.step_reward_full[:-1,...]

        # Calculated data
        self.discounted_reward = torch.zeros(buffer_size, num_envs, **self.tensor_args)
        self.success = torch.zeros(buffer_size, num_envs, dtype=torch.bool)

        self.obs_merged = self.obs.reshape(buffer_size * num_envs, *obs_shape)
        self.action_merged = self.action.reshape(buffer_size * num_envs, *act_shape)
        self.discounted_reward_merged = self.discounted_reward.reshape(-1)

    def step(self, obs: torch.Tensor, act: torch.Tensor, reward: torch.Tensor):
        # Update in episode data
        self.ep_cum_reward += reward

        # Update overall data
        self.obs[self.next_step_index, :] = obs
        self.action[self.next_step_index, :] = act
        self.step_reward[self.next_step_index, :] = reward
        self.next_step_index += 1

    def end_episode(self, finished_envs: torch.Tensor = None, success: np.ndarray = None):
        # Called if at least one environment has finished an episode

        for env_idx, finished in enumerate(finished_envs):
            if not finished:
                continue

            # Calculate stats for each step in the path
            self._finish_path(env_idx, success = success[env_idx].item())

            # Stats for finished episode
            self.ep_return[self.ep_count] = self.ep_cum_reward[env_idx]
            if self.update_partial_stats:
                self.ep_partial_return_info_count += 1
                self.ep_partial_return_count += self.ep_cum_reward[env_idx].item()
                self.stats[self.PARTIAL_MEAN_RETURN] = self.partial_mean_return()

            if success is not None:
                self.ep_success_info_count += 1
                if success[env_idx]:
                    self.ep_success_count += 1.0
                self.stats[self.SUCCESS_RATE] = self.success_rate()

                if self.update_partial_stats:
                    self.ep_partial_success_info_count += 1
                    if success[env_idx]:
                        self.ep_partial_success_count += 1.0
                    self.stats[self.PARTIAL_SUCCESS_RATE] = self.partial_success_rate()

            # Reset for next episode
            self.ep_start_index[env_idx] = self.next_step_index
            self.ep_cum_reward[env_idx] = 0.0
            self.ep_count += 1
            self.stats[self.EPISODE_COUNT] = self.ep_count

    def buffer_full(self, last_state_value_estimate: torch.tensor):
        # Called when as many steps as fit into the buffer have been taken

        assert(self.next_step_index == self.buffer_size)
        for env_idx in range(self.num_envs):
            # there is nothing to do if this batch just finished an episode
            if self.ep_start_index[env_idx] == self.next_step_index:
                continue

            # Calculate stats for each step in the path of the unfinished episodes
            self._finish_path(env_idx, last_state_value_estimate[env_idx])

        self.stats[self.MEAN_RETURN] = self.mean_return()
        self.stats[self.DISC_REWARD] = self.mean_disc_reward()


    def _finish_path(self, env_idx, last_state_value_estimate:float = 0.0, success = False):
        # Calculate stats for each step in a completed path
        self.step_reward_full[self.next_step_index, env_idx] = last_state_value_estimate
        rew_range = range(self.ep_start_index[env_idx], self.next_step_index+1)
        discounted_reward = discount_cumsum(self.step_reward_full[rew_range, env_idx], self.discount)
        ep_range = range(self.ep_start_index[env_idx], self.next_step_index)
        self.discounted_reward[ep_range, env_idx] = torch.tensor(np.array(discounted_reward[:-1]), **self.tensor_args)
        self.success[ep_range, env_idx] = success


    # clears the entire buffer
    def reset(self):
        self.ep_count = 0
        self.ep_success_count = 0
        self.ep_success_info_count = 0
        self.ep_partial_success_count = 0
        self.ep_partial_success_info_count = 0
        self.ep_partial_return_count = 0
        self.ep_partial_return_info_count = 0
        self.next_step_index = 0
        self.ep_start_index[...] = 0.0
        self.ep_cum_reward[...] = 0.0
        self.ep_return[...] = 0.0

        # This is mostly to create a clean state for unit testing
        self.obs[...] = 0.0
        self.action[...] = 0.0
        self.step_reward[...] = 0.0
        self.discounted_reward[...] = 0.0

        # Reset stats
        for stat in self.stats.keys():
            self.stats[stat] = None
        self.stats[self.SUCCESS_RATE] = 0
        self.stats[self.EPISODE_COUNT] = 0


    def mean_return(self) -> float:
        return self.ep_return[:self.ep_count].mean().item()


    def mean_disc_reward(self) -> float:
        return self.discounted_reward.mean().item()


    def success_rate(self) -> float:
        return self.ep_success_count / self.ep_success_info_count


    def partial_success_rate(self) -> float:
        if self.ep_partial_success_info_count==0:
            return 0
        return self.ep_partial_success_count / self.ep_partial_success_info_count


    def partial_mean_return(self) -> float:
        if self.ep_partial_return_info_count==0:
            return -1.0
        return self.ep_partial_return_count / self.ep_partial_return_info_count