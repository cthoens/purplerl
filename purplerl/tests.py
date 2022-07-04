
import unittest

import torch

from simple import ExperienceBuffer, Algo

in_obs = [
    [[0.1, 2.1], [1.2, 2.3]], # [env0/step0,  env1/step0]
    [[0.6, 2.7], [1.8, 3.1]], # [env0/step1,  env1/step1]
    [[0.1, 2.1], [1.2, 2.3]]  # [env0/step2,  env1/step2]
]
in_act = [
    [0, 1],
    [0, 0],
    [1, 1]
]
in_reward = [
    [1.0, 2.0],
    [1.2, 2.3],
    [1.8, 4.7]
]

# Version 1
in_done = [
    [False, False],
    [False, False],
    [True, True]
]
out_cum_reward = [
    [1.0, 2.0],
    [2.2, 4.3],
    [4.0, 9.0]
]
out_weight_reward_to_go = [
    [4.0, 9.0],
    [3.0, 7.0],
    [1.8, 4.7]
]
out_weight_vanilla = [
    [4.0, 9.0],
    [4.0, 9.0],
    [4.0, 9.0]
]

# Version 2
in_done2 = [
    [False, False],
    [False, True],
    [True, False]
]
out_cum_reward2 = [
    [1.0, 2.0],
    [2.2, 4.3],
    [4.0, 4.7]
]
out_weight_reward_to_go2 = [
    [4.0, 4.3],
    [3.0, 2.3],
    [1.8, 0.0]
]
out_weight_vanilla2 = [
    [4.0, 4.3],
    [4.0, 4.3],
    [4.0, 0.0]
]

class TestStringMethods(unittest.TestCase):

    def test_ExperienceBuffer(self):
        # General test of experience buffer
        #
        # - Add 3 environments steps of experience. All episodes end after 3 steps.
        # - Check buffers states.
        # - End the episode.
        # - Add 3 more steps of experience.
        # - Check buffers states.

        b = ExperienceBuffer(num_envs = 2, buffer_size=6, obs_shape=2, algo = Algo.VANILLA)
        self.run_test(b, expected_weights=out_weight_vanilla2)
        b.reset()
        self.run_test(b, expected_weights=out_weight_vanilla2)

        b = ExperienceBuffer(num_envs = 2, buffer_size=6, obs_shape=2, algo = Algo.REWARD_TO_GO)
        self.run_test(b, expected_weights=out_weight_reward_to_go2)
        b.reset()
        self.run_test(b, expected_weights=out_weight_reward_to_go2)


    def run_test(self, b: ExperienceBuffer, expected_weights):
        for step in range(3):
            b.step(
                obs=torch.tensor(in_obs[step]),
                act=torch.tensor(in_act[step]),
                reward=torch.tensor(in_reward[step]),
                done=torch.tensor(in_done2[step])
            )
            self.assertEqual(b.next_step_index, step+1)

        # Check inputs are stored a exprected
        self.assertTrue(
            torch.allclose(torch.transpose(b.obs[:, :3], 0, 1), torch.tensor(in_obs)),
            "Obs deviates from expectation"
        )
        self.assertTrue(
            torch.allclose(torch.transpose(b.action[:, :3], 0, 1), torch.tensor(in_act)),
            "Actions deviates from expectation"
        )
        self.assertTrue(
            torch.allclose(torch.transpose(b.step_reward[:, :3], 0, 1), torch.tensor(in_reward)),
            "Step reward deviates from expectation"
        )

        # Check aggreated value have expected values
        self.assertTrue(
            torch.allclose(torch.transpose(b.cum_reward[:, :3], 0, 1), torch.tensor(out_cum_reward2)),
            "Cumulative reward deviates from expectation"
        )
        self.assertTrue(
            torch.allclose(torch.transpose(b.weight[:, :3], 0, 1), torch.tensor(expected_weights)),
            "Weight deviates from expectation"
        )


if __name__ == '__main__':
    unittest.main()
