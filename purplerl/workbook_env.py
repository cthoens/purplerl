from collections import namedtuple
import os

import numpy as np

import gym
from gym.spaces import Box
from gym.error import DependencyNotInstalled

from PIL import Image, ImageOps

class WorkbookEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    goal_alpha = 128
    fail_alpha = 0
    block_alpha = 200
    spawn_alpha = 254
    traverse_alpha = 255

    SHEET = "sheet"
    POWER = "power"   
    SHEET_OBS_SPACE = Box(float("-1"), float("1"), (1, 128, 128, ))
    POWER_OBS_SPACE = Box(float("-1"), float("1"), (1, ))
    OBSERVATION_SPACE = Box(float("-1"), float("1"), tuple(np.prod(np.array(SHEET_OBS_SPACE.shape)) + np.array(POWER_OBS_SPACE.shape)))
    ACTION_SPACE = Box(float("-1"), float("1"), (2, ))

    MAX_ENERGY = 60
    SPAWN_DICT = {}

    def __init__(self, sheet_path="sheets") -> None:
        self.sheet_path = sheet_path
        self.screen = None
        self.clock = None
        self.energy_left = None
        self.lesson_paths = ["l00", "l01", "l02", "l10", "l20", "l30"]
        self.lesson_lengths = [3, 3, 3, 10, 10, 10]
        self.sheets = [self._get_sheets(lesson) for lesson in self.lesson_paths]
        self.templates = [[self._load_template(lesson_path, sheet) for sheet in lesson] for lesson, lesson_path in zip(self.sheets, self.lesson_paths)]
        self.lesson = 0
        self.max_episode_steps = self.lesson_lengths[self.lesson]
        self.template = 0
        self.max_speed = 2.0
        self.energy_reward_coeff = 0.5
        
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
        
        self.action_space = self.ACTION_SPACE
        self.observation_space = self.OBSERVATION_SPACE


    def set_lesson(self, lesson):
        if lesson < len(self.lesson_paths):
            self.lesson = lesson
            self.max_episode_steps = self.lesson_lengths[lesson]
            self.SPAWN_DICT = {}
            return True
        else:
            return False


    SheetState = namedtuple('SheetState', ['lesson', 'template_idx', 'flip', 'rot'])


    def reset(self, sheet_state = None):
        if sheet_state is not None:
            state = sheet_state
        else:
            state = self.SheetState(
                lesson = self.lesson,
                template_idx = int(np.random.uniform(low=0.0, high=len(self.templates[self.lesson]) - 1e-7)),
                flip = np.random.uniform(low=0.0, high=1.0) > 0.5,
                rot=int(np.random.uniform(low=0.0, high=3.0 - 1e-7))
            )
        
        self.sheet = np.array(self.templates[state.lesson][state.template_idx])
        if state.flip:
            self.sheet = np.flip(self.sheet, axis=1)
        self.sheet = np.rot90(self.sheet, k=state.rot)

        self.energy_left = self.max_episode_steps
        self.cursor_vel = np.zeros((2, ), dtype=np.float32)

        spawn_points = self.SPAWN_DICT.get(state, None)
        if spawn_points is None:
            spawn_points = self._get_spawn_points(self.sheet)
            self.SPAWN_DICT[state] = spawn_points

        idx = np.unravel_index(spawn_points[np.random.randint(low=0, high=len(spawn_points), size=1)[0]], self.SHEET_OBS_SPACE.shape)
        self.cursor_pos = idx[1:]
        assert(self._get_pixel() == 1.0)
        return self._get_obs()


    def step(self, action):
        assert self.cursor_pos is not None, "Call reset before using step method."
        
        self.cursor_vel = action
        action_speed = np.linalg.norm(self.cursor_vel)
        if action_speed > self.max_speed:
            self.cursor_vel *= self.max_speed / action_speed

        min_coord = np.array([  0.0,   0.0], dtype=np.float32)
        max_coord = np.array([127.0, 127.0], dtype=np.float32)
        self.cursor_pos = np.clip(self.cursor_pos + self.cursor_vel, min_coord, max_coord)
        obs = self._get_obs()
        info = {}
        self.energy_left -= max(action_speed, 1.0)
        if self._get_pixel()==-1.0:
            reward = 1.0 + self.energy_reward_coeff * (self.energy_left / self.MAX_ENERGY)
            done = True
            info = {"success": True}
            self.cursor_pos = None
            self.energy_left = None
        elif self.energy_left <= 0:
            reward = -1.0
            done = True
            info = {"success": False}
            self.cursor_pos = None
            self.energy_left = None
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

        obs_slice = obs[cursor_dest_from[0] : cursor_dest_to[0], cursor_dest_from[1] : cursor_dest_to[1]]
        mask_slice = self.cursor_mask[cursor_src_from[0] : cursor_src_to[0], cursor_src_from[1] : cursor_src_to[1]]
        obs_slice[mask_slice] = ((230.0 / 127.5) - 1.0)
        
        power_obs = np.array([self.energy_left / self.MAX_ENERGY], np.float32)
        return np.concatenate((obs.flatten(), power_obs))

    def _get_cursor_pos_int(self):
        return np.rint(self.cursor_pos).astype(np.int32)
        

    def _get_pixel(self):
        pos = self._get_cursor_pos_int()
        return self.sheet[pos[0], pos[1]]


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

        image = self._get_obs()[:-1].reshape(self.SHEET_OBS_SPACE.shape)
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

    def _load_template(self, lesson_path, name):
        with Image.open(os.path.join(self.sheet_path, lesson_path,  name)) as image:
            template = np.array(ImageOps.grayscale(image), dtype=np.float32)
            template /= 127.5
            template -= 1.0
        return template

    def _get_sheets(self, lesson):
        path = os.path.join(self.sheet_path, lesson)
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

    def _get_spawn_points(self, template):
        template = template.flatten()
        return np.where(template == 1.0)[0]


def test():
    import time

    env = WorkbookEnv()
    env.set_lesson(1)
    env.reset()
    for i in range(1000):
        time.sleep(0.2)
        _, _, done, _ = env.step(np.array([0.9, 0.1]))
        if done:
            env.reset()
        env.render()



if __name__ == '__main__':
    test()
