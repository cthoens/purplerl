from collections import namedtuple
import os

import numpy as np
from purplerl.environment import EnvManager
import torch

from gym.spaces import Box
from gym.error import DependencyNotInstalled

from PIL import Image, ImageOps

from purplerl.config import CpuConfig, GpuConfig

class WorbookVectorEnv(EnvManager):
    goal_alpha = 128
    fail_alpha = 0
    block_alpha = 200
    spawn_alpha = 254
    traverse_alpha = 255

    SHEET = "sheet"
    POWER = "power"

    MAX_ENERGY = 60
    SPAWN_DICT = {}

    def __init__(self,
        cfg:CpuConfig,
        sheet_path="sheets",
        num_envs=2
    ) -> None:
        init(sheet_path)

        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE

        self.cfg = cfg
        self.num_envs = num_envs
        self.sheet_path = sheet_path
        self.screen = None
        self.clock = None
        self.lesson = 0
        self.max_episode_steps = LESSON_LENGTHS[self.lesson]
        self.max_speed = 2.0
        self.energy_reward_coeff = 1.0
        self.origin_float = torch.zeros((2, ))
        self.origin_int = torch.zeros(2, dtype=torch.int32)

        self.cursor = torch.tensor([
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], **self.cfg.tensor_args) * ((230.0 / 127.5) - 1.0)
        self.cursor_mask = self.cursor != 0
        self.cursor_hotspot = torch.tensor([3, 3], dtype=torch.long)
        self.cursor_shape = torch.tensor(self.cursor.shape, dtype=torch.long)

        #CPU
        self.energy_left = torch.zeros(num_envs, requires_grad=False)
        self.cursor_pos = torch.zeros((num_envs, 2), requires_grad=False)
        self.cursor_vel = torch.zeros((num_envs, 2), requires_grad=False)
        self.reward = torch.zeros(num_envs, requires_grad=False)
        self.done = torch.zeros(num_envs, requires_grad=False, dtype=torch.bool)
        self.success = torch.zeros(num_envs, requires_grad=False, dtype=torch.bool)

        # GPU
        self.sheet = torch.zeros((num_envs, ) + SHEET_OBS_SPACE.shape, requires_grad=False, **cfg.tensor_args)
        self.obs = torch.zeros((num_envs, ) + OBSERVATION_SPACE.shape, requires_grad=False, **cfg.tensor_args)
        self.sheet_obs = self.obs[:,:64*64].reshape(-1, 1, 64, 64)
        self.power_obs = self.obs[:,64*64:64*64+1].flatten()
        self.prev_action_obs = self.obs[:,64*64+1:64*64+3]

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}


    def set_lesson(self, lesson):
        if lesson < len(LESSON_PATHS):
            if self.lesson == lesson:
                return True
            self.lesson = lesson
            self.max_episode_steps = LESSON_LENGTHS[lesson]
            self.SPAWN_DICT = {}
            return True
        else:
            return False


    SheetState = namedtuple('SheetState', ['lesson', 'template_idx', 'flip', 'rot'])


    def reset(self):
        for env_index in range(self.num_envs):
            self._reset_env(env_index)
        self._update_obs()
        assert(torch.all(self._get_pixel() == 1.0))
        return self.obs


    def _reset_env(self, env_index, sheet_state = None):
        if sheet_state is not None:
            state = sheet_state
        else:
            state = self.SheetState(
                lesson = self.lesson,
                template_idx = int(np.random.uniform(low=0.0, high=len(TEMPLATES[self.lesson]) - 1e-7)),
                flip = np.random.uniform(low=0.0, high=1.0) > 0.5,
                rot=int(np.random.uniform(low=0.0, high=4.0))
            )

        new_sheet = TEMPLATES[state.lesson][state.template_idx].clone()
        if state.flip:
            new_sheet = torch.flip(new_sheet, dims=[1])
        new_sheet = torch.rot90(new_sheet, k=state.rot)

        self.energy_left[env_index] = self.max_episode_steps
        self.cursor_vel[env_index].zero_()
        self.prev_action_obs[env_index].zero_()

        spawn_points = self.SPAWN_DICT.get(state, None)
        if spawn_points is None:
            spawn_points = self._get_spawn_points(new_sheet)
            self.SPAWN_DICT[state] = spawn_points

        self.sheet[env_index] = new_sheet.to(self.cfg.device)
        idx = np.unravel_index(spawn_points[np.random.randint(low=0, high=len(spawn_points), size=1)[0]], SHEET_OBS_SPACE.shape)
        self.cursor_pos[env_index] = torch.tensor(idx[1:]) + 0.5


    def step(self, action):
        self.prev_action_obs[...] = action

        self.cursor_vel[...] = action.cpu()
        action_speed = torch.linalg.norm(self.cursor_vel, dim=1)
        speed_exceeded = action_speed > self.max_speed
        if torch.any(speed_exceeded):
            # Multiplying the velocity of
            self.cursor_vel[speed_exceeded] *= (self.max_speed / action_speed[speed_exceeded]).reshape(-1, 1).repeat(1,2)

        min_coord = self.origin_float
        max_coord = torch.tensor([SHEET_OBS_SPACE.shape[1]-1.0, SHEET_OBS_SPACE.shape[2]-1.0])
        self.cursor_pos = torch.clip(self.cursor_pos + self.cursor_vel, min_coord, max_coord)

        self.energy_left -= torch.max(action_speed, torch.tensor(1.0))

        self._update_obs()

        out_of_energy = self.energy_left <= 0.0
        self.success[...] = self._get_pixel() == -1.0
        self.done[...] = self.success.logical_or(out_of_energy)

        self.reward[...] = 0.0
        self.reward[out_of_energy] = -1.0
        self.reward[self.success] = 1.0 + self.energy_reward_coeff * (self.energy_left[self.success] / self.MAX_ENERGY)

        for env_index, done in enumerate(self.done):
            if not done:
                continue
            self._reset_env(env_index)
            self.sheet_obs[env_index] = self.sheet[env_index]
            self.power_obs[env_index] = 1.0
            self._draw_cursur(env_index)

        return self.obs, self.reward, self.done, self.success


    def _update_obs(self):
        self.sheet_obs[...] = self.sheet
        self.power_obs[...] = (self.energy_left / self.MAX_ENERGY).to(self.cfg.device)

        for env_index in range(self.num_envs):
            self._draw_cursur(env_index)


    def _draw_cursur(self, env_index):
        cursor_dest_from = self._get_cursor_pos_int()[env_index] - self.cursor_hotspot
        cursor_dest_to = cursor_dest_from + self.cursor_shape

        cursor_src_from = -torch.minimum(cursor_dest_from, self.origin_int)
        cursor_src_to = self.cursor_shape - torch.maximum(cursor_dest_to - torch.tensor([64, 64]), self.origin_int)

        cursor_dest_from = torch.maximum(cursor_dest_from, self.origin_int)

        obs_slice = self.sheet_obs[env_index, 0, cursor_dest_from[0] : cursor_dest_to[0], cursor_dest_from[1] : cursor_dest_to[1]]
        mask_slice = self.cursor_mask[cursor_src_from[0] : cursor_src_to[0], cursor_src_from[1] : cursor_src_to[1]]
        obs_slice[mask_slice] = ((230.0 / 127.5) - 1.0)


    def _get_cursor_pos_int(self):
        return self.cursor_pos.to(torch.long)


    def _get_pixel(self):
        pos = self._get_cursor_pos_int()
        raveled_pos = (pos[:,0]*TEMPLATES[0][0].shape[0] + pos[:,1]).reshape(self.num_envs,1).to(self.cfg.device)
        return self.sheet_obs.reshape(self.num_envs, -1).gather(dim=1, index=raveled_pos).squeeze()


    def render(self, mode="human"):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        screen_width = SHEET_OBS_SPACE.shape[2] * 4
        screen_height = SHEET_OBS_SPACE.shape[1] * 4

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        image = self.obs[1,:-3].reshape(SHEET_OBS_SPACE.shape)
        image += 1.0
        image *= 127.5
        surf = pygame.surfarray.make_surface(image.cpu().numpy().astype(np.uint8).squeeze().swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (screen_width, screen_height))
        self.screen.blit(surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return torch.transpose(
                torch.tensor(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return True


    def _get_spawn_points(self, template):
        template = template.flatten()
        return torch.where(template == 1.0)[0]


ACTION_SPACE = None
SHEET_OBS_SPACE = None
POWER_OBS_SPACE = None
OBSERVATION_SPACE = None

#lesson_paths = ["l00", "l01", "l02", "l10", "l20", "l30"]
#lesson_lengths = [8, 8, 8, 10, 10, 10]
LESSON_PATHS = ["l00-I", "l01-C", "l02-V", "l03-T", "l04-Y", "l05-Z", "l06-X"]
LESSON_LENGTHS = [8, 8, 8, 8, 8, 8, 8]
SHEETS = None
TEMPLATES = None

def init(sheet_path):
    global SHEETS, TEMPLATES, ACTION_SPACE, SHEET_OBS_SPACE, POWER_OBS_SPACE, OBSERVATION_SPACE
    if SHEETS:
        return

    def _load_template(lesson_path, name):
        with Image.open(os.path.join(sheet_path, lesson_path,  name)) as image:
            template = torch.tensor(np.array(ImageOps.grayscale(image), dtype=np.float32))
            template /= 127.5
            template -= 1.0
        return template


    def _get_sheets(lesson):
            path = os.path.join(sheet_path, lesson)
            result = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            result.sort()
            return result

    SHEETS = [_get_sheets(lesson) for lesson in LESSON_PATHS]
    TEMPLATES = [[_load_template(lesson_path, sheet) for sheet in lesson] for lesson, lesson_path in zip(SHEETS, LESSON_PATHS)]


    ACTION_SPACE = Box(float("-1"), float("1"), (2, ))
    SHEET_OBS_SPACE = Box(float("-1"), float("1"), (1, TEMPLATES[0][0].shape[0], TEMPLATES[0][0].shape[1]))
    POWER_OBS_SPACE = Box(float("-1"), float("1"), (1, ))
    OBSERVATION_SPACE = Box(float("-1"), float("1"),
        tuple(np.prod(np.array(SHEET_OBS_SPACE.shape)) + np.array(POWER_OBS_SPACE.shape) + np.array(ACTION_SPACE.shape)))


def test():
    import time

    cfg = GpuConfig()

    env = WorbookVectorEnv(
        cfg = cfg,
        sheet_path = "sheets-64",
        num_envs=16
    )
    action = torch.tensor([[0.9, 0.1]] * 16)
    env.set_lesson(1)
    env.reset()
    for i in range(1000):
        time.sleep(0.8)
        _, rew, done, success = env.step(action)
        #print(rew)
        #print(done)
        #print(success)
        #print()
        env.render()



if __name__ == '__main__':
    test()
