#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""朝向探针：回答"机器人是不是横着走（螃蟹步）"。

判卷只测过速度大小和位移，从没测过**身体朝向 vs 运动方向**的夹角。
本探针在 12 s 回放中逐帧记录：
  - 基座 yaw（身体朝向，rad）
  - 世界系速度方向（运动朝向，rad）
  - 两者夹角 = "蟹行角"（0 = 朝前走，±90° = 纯横着走）
  - 左右脚逐帧接触力（正确的传感器索引口径！），顺带验证左脚是否真的落地
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

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Poppy-Walk-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--vx", type=float, default=0.2)
parser.add_argument("--seconds", type=float, default=12.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def pin_command(env, vx: float) -> None:
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term._resampling_time_range = (1e9, 1e9)
    term.command[:, 0] = vx
    term.command[:, 1] = 0.0
    term.command[:, 2] = 0.0
    if hasattr(term, "cfg") and hasattr(term.cfg, "rel_standing_envs"):
        term.cfg.rel_standing_envs = 0.0


def quat_to_yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.seconds + 5.0)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=DEVICE)

    obs, _ = env.reset()
    pin_command(env, args.vx)

    robot = env.unwrapped.scene["robot"]
    sensor = env.unwrapped.scene["contact_forces"]
    # 正确口径：力的索引必须走传感器自己的 body 顺序
    foot_ids, foot_names = sensor.find_bodies([".*l_foot", ".*r_foot"])
    print(f"[probe] sensor feet: {foot_names} ids={foot_ids}")

    n_steps = int(args.seconds * 50)
    yaws, crab_angles, speeds = [], [], []
    lf_forces, rf_forces = [], []

    for i in range(n_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        q = robot.data.root_quat_w[0].detach().cpu().numpy()
        yaw = quat_to_yaw(q)
        vw = robot.data.root_lin_vel_w[0].detach().cpu().numpy()
        speed = float(np.linalg.norm(vw[:2]))
        fz = sensor.data.net_forces_w[0, foot_ids, 2].detach().cpu().numpy()

        yaws.append(yaw)
        speeds.append(speed)
        lf_forces.append(float(fz[0]))
        rf_forces.append(float(fz[1]))
        if speed > 0.05:
            motion_dir = math.atan2(vw[1], vw[0])
            crab = (yaw - motion_dir + math.pi) % (2 * math.pi) - math.pi
            crab_angles.append(crab)

        if i % 50 == 0:
            print(f"[probe] t={i/50:.1f}s yaw={math.degrees(yaw):7.1f}deg "
                  f"speed={speed:.3f} fzL={fz[0]:5.1f}N fzR={fz[1]:5.1f}N")

    yaws = np.unwrap(np.array(yaws))
    crab = np.degrees(np.array(crab_angles))
    lf = np.array(lf_forces)
    rf = np.array(rf_forces)

    print("\n===== 朝向探针结果 =====")
    print(f"yaw 初值 {math.degrees(yaws[0]):.1f} deg, 末值 {math.degrees(yaws[-1]):.1f} deg, "
          f"12s 总漂移 {math.degrees(yaws[-1] - yaws[0]):.1f} deg")
    print(f"蟹行角(身体朝向 - 运动方向): mean={crab.mean():.1f} deg, "
          f"abs mean={np.abs(crab).mean():.1f} deg, p90 abs={np.percentile(np.abs(crab), 90):.1f} deg")
    print(f"左脚: 平均受力 {lf.mean():.1f} N, 承重占空 {(lf > 1.0).mean() * 100:.0f}%, 最小 {lf.min():.1f} N")
    print(f"右脚: 平均受力 {rf.mean():.1f} N, 承重占空 {(rf > 1.0).mean() * 100:.0f}%, 最小 {rf.min():.1f} N")
    print(f"左脚触地次数(受力上穿 1N): {int(((lf[1:] > 1.0) & (lf[:-1] <= 1.0)).sum())}")
    print(f"右脚触地次数(受力上穿 1N): {int(((rf[1:] > 1.0) & (rf[:-1] <= 1.0)).sum())}")


if __name__ == "__main__":
    main()
    simulation_app.close()
