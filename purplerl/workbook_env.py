
import math
import os
from re import TEMPLATE, template
from typing import Optional, Tuple, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.spaces import Box
from gym.error import DependencyNotInstalled

from PIL import Image, ImageOps
from torch import true_divide

class WorkbookEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    goal_alpha = 128
    fail_alpha = 0
    block_alpha = 200
    spawn_alpha = 254
    traverse_alpha = 255
   

    def __init__(self) -> None:
        # Set these in ALL subclasses
        self.screen = None
        self.clock = None
        self.steps_left = None
        self.sheets = [
            ["l00-s01.png", "l00-s02.png", "l00-s03.png"],
            ["l01-s01.png", "l01-s02.png", "l01-s03.png", "l01-s04.png"],
            ["l02-s01.png", "l02-s02.png"],
            ["l03-s01.png",],
        ]
        self.templates = [[self._load_template(sheet) for sheet in lesson] for lesson in self.sheets]
        self.lesson = 0
        self.template = 0
        
        self.cursor = np.array([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], dtype = np.float32) * ((230.0 / 127.5) - 1.0)
        self.cursor_mask = self.cursor != 0
        self.cursor_hotspot = np.array([3, 3], dtype=np.int32)
        self.cursor_shape = np.array(list(self.cursor.shape), dtype=np.int32)
        
        self.cursor_pos = None
        self.cursor_vel = None
        
        self.action_space = Box(float("-1"), float("1"), (2, ))
        self.observation_space = Box(float("-inf"), float("inf"), (1, 128, 128, ))

    def _load_template(self, name):
        with Image.open(os.path.join("/home/cthoens/code/UnityRL/purplerl/sheets/", name)) as image:
            template = np.array(ImageOps.grayscale(image), dtype=np.float32)
            template /= 127.5
            template -= 1.0
        return template

    def set_lesson(self, lesson):
        if lesson < len(self.lessons):
            self.lesson = lesson
            return True
        else:
            return False
    
    def reset(self):
        template_idx = int(np.random.uniform(low=0.0, high=len(self.templates[self.lesson]) - 1e-7))
        self.sheet = np.array(self.templates[self.lesson][template_idx])
        if np.random.uniform(low=0.0, high=1.0) > 0.5:
            self.sheet = np.flip(self.sheet, axis=1)
        self.sheet = np.rot90(self.sheet, k = int(np.random.uniform(low=0.0, high=3.0 - 1e-7)))

        self.steps_left = 150
        self.cursor_pos = np.array([64.0, 64.0], dtype=np.float32)
        self.cursor_vel = np.zeros((2, ), dtype=np.float32)

        for _ in range(100):
            self.cursor_pos = np.random.uniform(low=0.0, high=127.0, size=2).astype(np.float32)
            if self._get_pixel() == 1.0:
                return self._get_obs()

        raise Exception("Could not find start")

    def step(self, action):
        assert self.cursor_pos is not None, "Call reset before using step method."
        
        self.cursor_vel = action
        min_coord = np.array([  0.0,   0.0], dtype=np.float32)
        max_coord = np.array([127.0, 127.0], dtype=np.float32)
        self.cursor_pos = np.clip(self.cursor_pos + self.cursor_vel, min_coord, max_coord)
        obs = self._get_obs()
        self.steps_left -= 1
        info = {}
        if self._get_pixel()==-1.0:
            reward = 1.0
            done = True
            info = {"success": True}
            self.cursor_pos = None
            self.steps_left = None
        elif self.steps_left == 0:
            reward = -1.0
            done = True
            info = {"success": False}
            self.cursor_pos = None
            self.steps_left = None
        else:
            reward = 0.0
            done = False
        return obs, reward, done, info


    def _get_obs(self):
        obs = np.array(self.sheet)
        cursor_dest_from = self._get_cursor_pos_int() - self.cursor_hotspot
        cursor_dest_to = cursor_dest_from + self.cursor_shape
        
        cursor_src_from = -np.minimum(cursor_dest_from, 0)
        cursor_src_to = self.cursor_shape - np.maximum(cursor_dest_to - self.sheet.shape, 0)
        
        cursor_dest_from = np.maximum(cursor_dest_from, 0)

        obs_slice = obs[cursor_dest_from[1] : cursor_dest_to[1], cursor_dest_from[0] : cursor_dest_to[0]]
        mask_slice = self.cursor_mask[cursor_src_from[1] : cursor_src_to[1], cursor_src_from[0] : cursor_src_to[0]]
        obs_slice[mask_slice] = ((230.0 / 127.5) - 1.0)
        return obs.reshape(*self.observation_space.shape)

    def _get_cursor_pos_int(self):
        return np.rint(self.cursor_pos).astype(np.int32)
        

    def _get_pixel(self):
        pos = self._get_cursor_pos_int()
        return self.sheet[pos[1], pos[0]]


    def render(self, mode="human"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = 128*4
        screen_height = 128*4

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        image = self._get_obs()
        image += 1.0
        image *= 127.5
        surf = pygame.surfarray.make_surface(image.astype(np.uint8).squeeze().swapaxes(0, 1))        
        surf = pygame.transform.scale(surf, (screen_width, screen_height))
        self.screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return True


def test():
    env = WorkbookEnv()
    env.reset()
    for i in range(1000):
        _, _, done, _ = env.step(np.array([-0.9, -0.1]))
        if done:
            env.reset()
        env.render()



if __name__ == '__main__':
    test()
