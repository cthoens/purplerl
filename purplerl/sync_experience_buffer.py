import numpy as np
import torch

import scipy.signal

from purplerl.config import tensor_args

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
    def __init__(self, 
        batch_size:int, 
        buffer_size: int, 
        act_shape: list[int],
        gamma: float = 0.99
    ) -> None:
        # Saved parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        # Overall statistics
        self.ep_count = 0
        self.next_step_index = 0
        self.ep_success_count = 0 # Number of successfully episodes
        self.ep_success_info_count = 0 # Number of times we have received info about success of failure of an episode
        self.ep_return =  np.zeros(batch_size * buffer_size, dtype=np.float32)

        # Per bach data
        self.ep_start_index = np.zeros(batch_size, dtype=np.int32)
        self.ep_cum_reward = np.zeros(batch_size, dtype=np.float32)

        # Stored data
        self.action = torch.zeros(batch_size, buffer_size, *act_shape, **tensor_args)
        self.step_reward = np.zeros((batch_size, buffer_size), dtype=np.float32)

        # Calculated data
        self.discounted_reward = torch.zeros(batch_size, buffer_size, **tensor_args)

    def step(self, obs: torch.Tensor, act: torch.Tensor, reward: torch.Tensor):
        # Update in episode data
        self.ep_cum_reward += reward

        # Update overall data
        self.action[:, self.next_step_index] = act
        self.step_reward[:, self.next_step_index] = reward
        self.next_step_index += 1

    # accumulates statistics for all finished batches
    def end_episode(self, finished_batches: torch.Tensor = None, success: np.ndarray = None):
        for batch, finished in enumerate(finished_batches):
            if not finished:
                continue

            ep_range = range(self.ep_start_index[batch], self.next_step_index)
            discounted_reward = discount_cumsum(self.step_reward[batch, ep_range], self.gamma)
            self.discounted_reward[batch, ep_range] = torch.tensor(np.array(discounted_reward), **tensor_args)


            self.ep_start_index[batch] = self.next_step_index
            self.ep_return[self.ep_count] = self.ep_cum_reward[batch]
            self.ep_cum_reward[batch] = 0.0
            if success is not None:
                self.ep_success_info_count += 1
                if success[batch]:
                    self.ep_success_count += 1.0
            self.ep_count += 1

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

    def mean_return(self) -> float:
        return self.ep_return[:self.ep_count].mean()

    def success_rate(self) -> float:
        return self.ep_success_count / self.ep_success_info_count

class MonoObsExperienceBuffer(ExperienceBufferBase):
    def __init__(self, batch_size:int, buffer_size: int, obs_shape: list[int], act_shape: list[int]) -> None:
        super().__init__(batch_size, buffer_size, act_shape)
        # Stored data
        self.obs = torch.zeros(batch_size, buffer_size, *obs_shape, **tensor_args)

    def step(self, obs: torch.Tensor, act: torch.Tensor, reward: torch.Tensor):
        self.obs[:,self.next_step_index] = obs

        # Note: Call after updating to prevent self.next_step_index from getting incremented too soon
        super().step(obs, act, reward)

    # clears the entire buffer
    def reset(self):
        super().reset()

        self.obs[...] = 0.0