from sre_constants import SUCCESS
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


class ExperienceBufferBase:
    MEAN_RETURN = "Mean Return"
    SUCCESS_RATE = "Success Rate"
    EPISODE_COUNT = "Ep Count"
    
    def __init__(self, 
        num_envs:int, 
        buffer_size: int, 
        act_shape: list[int],
        discount: float = 0.99,
        tensor_args = {}
    ) -> None:
        # Saved parameters
        self.discount = discount
        self.num_envs = num_envs
        self.buffer_size = buffer_size
        self.tensor_args = tensor_args
        self.stats = {
            self.MEAN_RETURN: 0.0,
            self.SUCCESS_RATE: 0.0,
            self.EPISODE_COUNT: 0
        }

        # Overall statistics
        self.ep_count = 0
        self.next_step_index = 0
        self.ep_success_count = 0 # Number of successfully episodes
        self.ep_success_info_count = 0 # Number of times we have received info about success of failure of an episode
        self.ep_return =  np.zeros(num_envs * buffer_size, dtype=np.float32)

        # Per bach data
        self.ep_start_index = np.zeros(num_envs, dtype=np.int32)
        self.ep_cum_reward = np.zeros(num_envs, dtype=np.float32)

        # Stored data
        self.action = torch.zeros(num_envs, buffer_size, *act_shape, **self.tensor_args)
        # The extra slot at the end of used in buffer_full()
        self.step_reward_full = np.zeros((num_envs, buffer_size+1), dtype=np.float32)
        self.step_reward = self.step_reward_full[:,:-1]

        # Calculated data
        self.discounted_reward = torch.zeros(num_envs, buffer_size, **self.tensor_args)

    def step(self, act: torch.Tensor, reward: torch.Tensor):
        # Update in episode data
        self.ep_cum_reward += reward

        # Update overall data
        self.action[:, self.next_step_index] = act
        self.step_reward[:, self.next_step_index] = reward
        self.next_step_index += 1

    # accumulates statistics for all finished batches
    def end_episode(self, finished_envs: torch.Tensor = None, success: np.ndarray = None):
        for env_idx, finished in enumerate(finished_envs):
            if not finished:
                continue

            self._finish_path(env_idx)

            self.ep_start_index[env_idx] = self.next_step_index
            self.ep_return[self.ep_count] = self.ep_cum_reward[env_idx]
            self.ep_cum_reward[env_idx] = 0.0
            if success is not None:
                self.ep_success_info_count += 1
                if success[env_idx]:
                    self.ep_success_count += 1.0
                self.stats[self.SUCCESS_RATE] = self.success_rate()
            self.ep_count += 1
            self.stats[self.EPISODE_COUNT] = self.ep_count

    def buffer_full(self, last_state_value_estimate: torch.tensor):
        assert(self.next_step_index == self.buffer_size)
        for env_idx in range(self.num_envs):
            # there is nothing to do if this batch just finished an episode
            if self.ep_start_index[env_idx] == self.next_step_index:
                continue

            self._finish_path(env_idx, last_state_value_estimate[env_idx])
        self.stats[self.MEAN_RETURN] = self.mean_return()

    
    def _finish_path(self, env_idx, last_state_value_estimate:float = 0.0):
        self.step_reward_full[env_idx, self.next_step_index] = last_state_value_estimate
        rew_range = range(self.ep_start_index[env_idx], self.next_step_index+1)
        discounted_reward = discount_cumsum(self.step_reward_full[env_idx, rew_range], self.discount)
        ep_range = range(self.ep_start_index[env_idx], self.next_step_index)
        self.discounted_reward[env_idx, ep_range] = torch.tensor(np.array(discounted_reward[:-1]), **self.tensor_args)
 

    # clears the entire buffer
    def reset(self):
        self.ep_count = 0
        self.ep_success_count = 0
        self.ep_success_info_count = 0
        self.next_step_index = 0
        self.ep_start_index[...] = 0.0
        self.ep_cum_reward[...] = 0.0
        self.ep_return[...] = 0.0        

        # This is mostly to create a clean state for unit testing
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

    def success_rate(self) -> float:
        return self.ep_success_count / self.ep_success_info_count


class MonoObsExperienceBuffer(ExperienceBufferBase):
    def __init__(self, 
        num_envs:int, 
        buffer_size: int, 
        obs_shape: list[int], 
        act_shape: list[int],
        discount: float = 0.99,
        tensor_args: dict = {}
    ) -> None:
        super().__init__(num_envs, buffer_size, act_shape, discount, tensor_args)
        # Stored data
        self.obs = torch.zeros(num_envs, buffer_size, *obs_shape, **self.tensor_args)

    def step(self, obs: torch.Tensor, act: torch.Tensor, reward: torch.Tensor):
        self.obs[:,self.next_step_index] = obs

        # Note: Call after updating to prevent self.next_step_index from getting incremented too soon
        super().step(act, reward)

    # clears the entire buffer
    def reset(self):
        super().reset()

        self.obs[...] = 0.0