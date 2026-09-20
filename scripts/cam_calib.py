#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""相机约定标定：同一姿态下从体坐标前后左右各渲一帧，验证 set_camera_view 行为。"""
from __future__ import annotations

import argparse
import math
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
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

DEVICE = "cuda:0"


def quat_to_yaw(q):
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def b2w(offset, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * offset[0] - s * offset[1],
                     s * offset[0] + c * offset[1],
                     offset[2]])


def main() -> None:
    out_dir = os.path.join(_REPO, "out", "cam_calib")
    os.makedirs(out_dir, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    robot = env.unwrapped.scene["robot"]
    base = robot.data.root_pos_w[0].detach().cpu().numpy()
    yaw = quat_to_yaw(robot.data.root_quat_w[0].detach().cpu().numpy())
    print(f"[calib] base={base.round(3)} yaw={math.degrees(yaw):.1f}deg")

    views = {
        "front": (2.5, 0.0, 0.5),
        "back": (-2.5, 0.0, 0.5),
        "left": (0.0, 2.5, 0.5),
        "right": (0.0, -2.5, 0.5),
    }
    for name, off in views.items():
        eye = (base + b2w(off, yaw)).tolist()
        tgt = base.tolist()
        env.unwrapped.sim.set_camera_view(eye, tgt)
        img = env.render()
        imageio.imwrite(os.path.join(out_dir, f"cam_{name}.jpg"), img)
        print(f"[calib] {name}: eye={[round(v,2) for v in eye]}")

    env.close()
    print(f"[calib] done -> {out_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()
