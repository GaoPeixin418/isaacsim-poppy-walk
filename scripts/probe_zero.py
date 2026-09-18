#!/usr/bin/env python3
# 零动作探针：默认姿态 + 零动作 3 秒，测左右脚高度与基座横滚
# 判别"左脚差 7mm"是资产不对称还是策略右倾
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "source", "poppy_walk")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import poppy_walk  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Poppy-Walk-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=4)
env = gym.make(args.task, cfg=env_cfg)
unwrapped = env.unwrapped
robot = unwrapped.scene["robot"]
foot_ids, foot_names = robot.find_bodies(".*foot")

obs, _ = env.reset()
zeros = torch.zeros(env.action_space.shape, device="cuda:0")
# 2 秒零动作
zs, rolls = [], []
for i in range(int(2.0 / unwrapped.step_dt)):
    obs, rew, term, trunc, info = env.step(zeros)
    foot_z = robot.data.body_pos_w[:, foot_ids, 2]
    # 基座横滚: 从投影重力 x 分量读
    pg = robot.data.projected_gravity_b
    zs.append(foot_z.mean(0).cpu())
    rolls.append(pg[:, 0].mean().cpu())

zs = torch.stack(zs)
print("=" * 50)
print(f"脚部: {foot_names}")
for j, name in enumerate(foot_names):
    print(f"{name}: z mean = {zs[:, j].mean():.4f} m, "
          f"min = {zs[:, j].min():.4f}, max = {zs[:, j].max():.4f}")
print(f"左右脚高度差(l-r): {(zs[:, 0] - zs[:, 1]).mean() * 1000:+.2f} mm")
print(f"基座 projected_gravity_b.x (横滚指标): {torch.stack(rolls).mean():+.4f}")
print(f"基座 z mean: {robot.data.root_pos_w[:, 2].mean():.4f} (不严谨,仅参考)")
simulation_app.close()
