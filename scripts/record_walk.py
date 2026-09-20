#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 行走视频录制（相机跟随版）。

为什么要单独写这个脚本而不用 play_poppy.py --video：
官方 play.py 的 RecordVideo 用的是**固定相机**（Isaac Lab 默认 ViewerCfg），
机器人以 ~0.3 m/s 走远后画面里只剩一个小点，没法当简历素材。
本脚本在每步前把视口相机"绑"在 0 号环境机器人身上（侧后方跟拍），
走再远机器人也始终在画面中央。

用法：
    bash scripts/run.sh scripts/record_walk.py --headless \
        --task Poppy-Walk-Play-v0 \
        --checkpoint logs/rsl_rl/poppy_walk/<run>/model_2400.pt \
        --seconds 12 --out out/videos
"""
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

parser = argparse.ArgumentParser(description="Record Poppy walking video with a follow camera.")
parser.add_argument("--task", type=str, default="Poppy-Walk-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--vx", type=float, default=0.2)
parser.add_argument("--seconds", type=float, default=12.0)
parser.add_argument("--out", type=str, default=os.path.join(_REPO, "out", "videos"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下 import 必须发生在 AppLauncher 之后（Omniverse 运行时限制）
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 正侧面跟拍偏置（**体坐标系**：y=解剖学正前方【相机标定实证】，x=身体右侧，z=上）
# 经典步态分析视角：双腿摆动/支撑相在画面里最分明
CAM_OFFSET = (2.0, -0.4, 0.6)    # eye 相对 base 的体坐标位置（右侧方、略靠后）
LOOK_OFFSET = (0.0, 0.2, -0.05)  # 注视点略低于基座（画面中心偏腿）


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def body_to_world(offset: tuple, yaw: float) -> np.ndarray:
    """体坐标 xy 偏置按 yaw 旋转到世界系（z 不变）。"""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * offset[0] - s * offset[1],
                     s * offset[0] + c * offset[1],
                     offset[2]])


def pin_command(env, vx: float) -> None:
    """钉死速度指令（同 d5_eval.py）。"""
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term._resampling_time_range = (1e9, 1e9)
    term.command[:, 0] = vx
    term.command[:, 1] = 0.0
    term.command[:, 2] = 0.0
    if hasattr(term, "cfg") and hasattr(term.cfg, "rel_standing_envs"):
        term.cfg.rel_standing_envs = 0.0


def main() -> None:
    os.makedirs(args.out, exist_ok=True)

    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.seconds + 5.0)
    # render_mode="rgb_array" 会强制 enable_cameras，headless 下也跑离屏渲染
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array")

    video_folder = os.path.abspath(args.out)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        episode_trigger=lambda e: True,
        disable_logger=True,
        name_prefix="poppy-walk-follow",
    )
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=DEVICE)

    obs, _ = env.reset()
    pin_command(env, args.vx)

    # 与训练一致：policy 频率 50 Hz（decimation 4 x sim dt 0.005）
    n_steps = int(args.seconds * 50)
    print(f"[record] 录制 {n_steps} 步（{args.seconds} s 仿真时间），相机跟随 env 0")

    robot = env.unwrapped.scene["robot"]
    for i in range(n_steps):
        # 先摆相机，再 step（step 内部触发本帧渲染）
        base = robot.data.root_pos_w[0].detach().cpu().numpy()
        yaw = quat_to_yaw(robot.data.root_quat_w[0].detach().cpu().numpy())
        eye = (base + body_to_world(CAM_OFFSET, yaw)).tolist()
        tgt = (base + body_to_world(LOOK_OFFSET, yaw)).tolist()
        env.unwrapped.sim.set_camera_view(eye, tgt)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        if i % 100 == 0:
            print(f"[record] step {i}/{n_steps}")

    env.close()
    print(f"[record] 完成，视频在 {video_folder}")


if __name__ == "__main__":
    main()
    simulation_app.close()
