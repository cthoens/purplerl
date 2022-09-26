import os


import torch
import numpy as np

from matplotlib import pyplot as plt

import PIL
from PIL import Image, ImageDraw

import purplerl.config
import purplerl.workbook_env as Env
from purplerl.workbook_env import WorkbookEnv
from purplerl.policy import PPO

color_map = plt.get_cmap('RdYlGn')

def do_eval(out_dir, epoch, lesson, policy_updater: PPO):
    if epoch % 50 != 0:
        return

    env = WorkbookEnv()
    env.set_lesson(lesson)

    states = [[
        WorkbookEnv.SheetState(
            lesson = lesson,
            template_idx = template_idx,
            flip = False,
            rot=rot
        ) for rot in range(4)] for template_idx in range(min(len(Env.TEMPLATES[lesson]), 4))
    ]

    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    _, axs = plt.subplots(len(states), len(states[0]))
    if len(states)==1:
        axs = [axs]
    for row, plt_row in enumerate(axs):
        plt_state = states[row]

        for ax, state in zip(plt_row, plt_state):
            sheet, spawn_points, values, action_means, action_stddevs = evaluate(env, state, policy_updater)
            act_image_np = np.array(visualize(env, sheet, spawn_points, values, action_means, action_stddevs))
            ax.imshow(act_image_np, cmap='RdYlGn', vmin=0.0, vmax=1.0)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{epoch}.png")
    return plt


def evaluate(env: WorkbookEnv, state, policy_updater: PPO):
    env.reset(sheet_state = state)

    spawn_points = env._get_spawn_points(env.sheet)

    values = np.zeros(spawn_points.shape)
    action_means = np.zeros(spawn_points.shape + env.action_space.shape)
    action_stddevs = np.zeros(spawn_points.shape + env.action_space.shape)

    with torch.no_grad():
        obs = torch.zeros(len(spawn_points), *Env.OBSERVATION_SPACE.shape, dtype=purplerl.config.dtype)
        for pt_idx, spawn_point in enumerate(spawn_points):
            idx = np.unravel_index(spawn_point, Env.SHEET_OBS_SPACE.shape)[1:]
            env.steps_left = 1
            env.cursor_pos = np.array(idx)
            obs[pt_idx] = torch.as_tensor(env._get_obs())

        obs = torch.as_tensor(obs, **policy_updater.cfg.tensor_args)
        encoded_obs = policy_updater.policy.obs_encoder(obs)
        values = policy_updater.value_net_tail(encoded_obs)
        actions_dists = policy_updater.policy.action_dist(encoded_obs=encoded_obs)
        action_means = actions_dists.mean.cpu().numpy()
        action_stddevs = actions_dists.stddev.cpu().numpy()

    return env.sheet, spawn_points, values, action_means, action_stddevs


def visualize(env, sheet, spawn_points, values, action_means, action_stddevs, scale = 22):
    mid = scale // 2

    crop_rect = find_crop(env.sheet)
    crop_rect[0] -= int(env.max_speed)
    crop_rect[1] += int(env.max_speed)

    sheet = (np.array(sheet) + 1.0) / 2.0
    values_scaled = (values + 1.0) / 2.0
    for pt_idx, spawn_point in enumerate(spawn_points):
        idx = np.unravel_index(spawn_point, Env.SHEET_OBS_SPACE.shape)[1:]
        sheet[idx] = values_scaled[pt_idx]

    colored_sheet = color_map(sheet)
    img = Image.fromarray((colored_sheet[:, :, :3] * 255).astype(np.uint8))
    act_image = img.resize( np.array(Env.SHEET_OBS_SPACE.shape[1:]) * scale, resample = PIL.Image.NEAREST )

    draw = ImageDraw.Draw(act_image)

    # draw grid
    for i in range(Env.SHEET_OBS_SPACE.shape[1]):
        pt_from = np.array([i,0])*scale
        pt_to =   np.array([i,128])*scale
        draw.line([tuple(pt_from), tuple(pt_to)], fill='black', width=1)
        pt_from = np.array([0,i])*scale
        pt_to =   np.array([128,i])*scale
        draw.line([tuple(pt_from), tuple(pt_to)], fill='black', width=1)

    # draw actions
    for pt_idx, spawn_point in enumerate(spawn_points):
        idx = np.flip(np.array(np.unravel_index(spawn_point, Env.SHEET_OBS_SPACE.shape)[1:]))

        direction = np.flip(action_means[pt_idx]) * scale
        center = idx*scale+mid
        dest = center+direction
        draw.line([tuple(center), tuple(dest)], fill=(0, 0, 196), width=3)
        draw.ellipse([tuple(center-1), tuple(center+1)], outline=(0, 0, 196), width=1)

        std = action_stddevs[pt_idx] * scale
        draw.ellipse([tuple(dest - std), tuple(dest + std)], outline=(0, 0, 196), width=2)

    crop = crop_rect * scale
    return act_image.crop(tuple(crop.flatten()))


def find_crop(img):
    for y in range(img.shape[0]):
        if np.any(img[y] > 0):
            upper = y
            break

    for y in range(img.shape[0]-1, upper, -1):
        if np.any(img[y] > 0):
            lower = y+1
            break

    img = img[upper:lower,:]

    for x in range(img.shape[1]):
        if np.any(img[:, x] > 0 ):
            left = x
            break

    for x in range(img.shape[1]-1, left+1, -1):
        if np.any(img[:, x] > 0 ):
            right = x+1
            break

    return np.array([[left, upper], [right, lower]])