from multiprocessing import Process, Queue
import random
import os

import torch
import numpy as np

from purplerl.environment import EnvManager
from purplerl.tune_env import TuneEnv

class EnvProcess(Process):

    def run(self):
        seed = 10000 * os.getpid()
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.in_queue: Queue = self._args[0]
        self.out_queue: Queue = self._args[1]
        self.shared_obs: torch.tensor = self._args[2]
        # TODO: Make not hard coded
        self.env = TuneEnv()

        while True:
            which, action = self.in_queue.get()

            if which=='r':
                self.shared_obs[...] = torch.as_tensor(self.env.reset())
                self.out_queue.put(None)
            elif which=='s':
                obs, reward, done, info = self.env.step(action)
                self.shared_obs[...] = torch.as_tensor(obs)
                self.out_queue.put((reward, done, info, ))
            elif which=='l':
                set_lesson = getattr(self.env, "set_lesson", None)
                if not callable(set_lesson):
                    result = False
                else:
                    result = self.env.set_lesson(action)
                self.out_queue.put(result)
            elif which=='q':
                return
            else:
                print(f"Unknown commmand: {which}")


class ProcessEnvManager(EnvManager):
    def __init__(self, env_name='CartPole-v0', num_envs=2) -> None:
        self.shared_obs = torch.zeros(num_envs, *TuneEnv.OBSERVATION_SPACE.shape, dtype=torch.float32).share_memory_()
        self.in_queues = [Queue() for _ in range(num_envs)]
        self.out_queues = [Queue() for _ in range(num_envs)]
        self.envs = [EnvProcess(args=(i, o, obs)) for i, o, obs in zip(self.in_queues, self.out_queues, self.shared_obs)]
        self.done = [False for _ in range(num_envs)]
        self.success = [False for _ in range(num_envs)]
        self.observation_space = TuneEnv.OBSERVATION_SPACE
        self.action_space = TuneEnv.ACTION_SPACE

        for env in self.envs:
            env.start()

    def reset(self):
        for in_queue in self.in_queues:
            in_queue.put(('r', None))
        [out_queue.get() for out_queue in self.out_queues]
        obs = self.shared_obs.numpy()
        return obs

    def step(self, actions: np.ndarray):
        for q, act in zip(self.in_queues, actions):
            q.put(('s', act.reshape(self.action_space.shape)))
        interaction =  [q.get() for q in self.out_queues]
        obs =  obs = self.shared_obs.numpy()
        rew =  np.array([rew for rew, _, _ in interaction], dtype=np.float32)
        done = np.array([done for _, done, _ in interaction], dtype=np.float32)
        success = np.array([info.get("success", False) for _, _, info in interaction], dtype=np.bool8)
        for index, (in_queue, out_queue, env_done) in enumerate(zip(self.in_queues, self.out_queues, done)):
            if not env_done:
                continue
            in_queue.put(('r', None))
            out_queue.get()
        return obs, rew, done, success

    def set_lesson(self, lesson):
        for q in zip(self.in_queues):
            q.put(('l', lesson))
        more_lessons = np.array([q.get() for q in self.out_queues])
        assert(np.all(more_lessons == more_lessons[0]))
        return more_lessons[0]

    def close(self):
        for q in zip(self.in_queues):
            q.put(('q'))
        for env in self.envs:
            env.join()


if __name__ == '__main__':
    pass
