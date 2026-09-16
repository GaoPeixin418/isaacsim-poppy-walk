#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 策略播放（可视化/验收）入口。

和 train_poppy.py 是完全一样的薄壳思路：注册任务 → 转交 Isaac Lab 官方 play.py。
区别只在默认任务（换成 *-Play-v0）和检查点的自动挑选。

================================================================================
为什么需要这个脚本：训练用的配置不能用来"看"
================================================================================
Isaac Lab 把每个任务注册成两个环境：

  Poppy-Stand-v0       训练用：2048 个环境、观测带噪声、每 5 s 随机推一把
  Poppy-Stand-Play-v0  演示用：环境数少、关噪声、关随机扰动

训练时那些随机化是**故意的**——它们强迫策略学会"在有干扰的情况下也能站住"，
这叫域随机化（domain randomization）。但你要"看看它站得怎么样"时，
噪声和随机推力只会让画面抖来抖去、看不清真实水平。
所以演示必须换成 Play 配置，这是官方约定的命名习惯（`-Play-v0` 后缀）。

================================================================================
服务器上没有显示器，怎么"看"
================================================================================
两条路，原理不同：

A. 录成视频（本脚本 --video 做的事）
   `--video` 会把环境包一层 gym 的标准 RecordVideo 包装器：
     · 把 render_mode 设成 "rgb_array"（每步返回一张 HxWx3 的 numpy 图片）
     · 同时强制 enable_cameras=True，让 Isaac Sim 在无窗口模式下也跑渲染管线
     · 每步存一帧，跑完用 ffmpeg 合成 mp4，落到 <log_dir>/videos/play/
   然后把 mp4 下载到本地看。优点是**不卡**，缺点是没法中途改参数。

B. 远程桌面 + GUI 模式（不加 --headless）
   服务器上其实有一套真 Xorg（DISPLAY=:1，跑着 GNOME，用 NVIDIA 驱动），
   用 ToDesk / xrdp 连上去，再把 DISPLAY 指到 :1，就能开出真的 3D 窗口，
   可以拖鼠标从任意角度看。缺点是远程桌面传画面的带宽有限，会有卡顿。

================================================================================
检查点怎么找
================================================================================
不传 --checkpoint 时，本脚本会在 logs/rsl_rl/<实验名>/ 下**按训练轮数找最大的**
（model_1199.pt 胜过 model_2.pt）。不用"最新的目录"，是因为调试时残留的
短跑（3 轮冒烟测试）目录更新时间可能更晚，会把真正训练好的模型盖掉。

================================================================================
用法
================================================================================
    # 看站立（默认）
    bash scripts/run.sh scripts/play_poppy.py --headless --video --num_envs 16

    # 看行走
    bash scripts/run.sh scripts/play_poppy.py --headless --video --num_envs 16 \
        --task Poppy-Walk-Play-v0

    # 指定检查点、录 500 步（=10 s 仿真时间）
    bash scripts/run.sh scripts/play_poppy.py --headless --video --video_length 500 \
        --checkpoint logs/rsl_rl/poppy_stand/2026-09-16_14-30-33/model_1199.pt

    # 开真实 GUI 窗口（需要 DISPLAY）
    DISPLAY=:1 XAUTHORITY=/run/user/1000/gdm/Xauthority \
        bash scripts/run.sh scripts/play_poppy.py --num_envs 8
"""
from __future__ import annotations

import glob
import os
import re
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# ---- 1. 让 poppy_walk 可被导入（仓库内开发模式） ----
_SRC = os.path.join(_REPO, "source", "poppy_walk")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import poppy_walk  # noqa: E402,F401  (gym.register 在这里发生)

# ---- 2. 定位 Isaac Lab 的官方播放脚本 ----
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/data/poppy/src/IsaacLab")
PLAY_PY = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rsl_rl", "play.py")
if not os.path.isfile(PLAY_PY):
    raise SystemExit(
        f"找不到 Isaac Lab 的播放脚本: {PLAY_PY}\n"
        f"请用环境变量指定 Isaac Lab 目录：ISAACLAB_ROOT=/path/to/IsaacLab"
    )

_SCRIPT_DIR = os.path.dirname(PLAY_PY)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---- 3. 默认任务：站立演示版 ----
if "--task" not in sys.argv:
    sys.argv = [sys.argv[0], "--task", "Poppy-Stand-Play-v0"] + sys.argv[1:]
_task = sys.argv[sys.argv.index("--task") + 1]

# 任务名 -> 实验名（对应 agent cfg 里的 experiment_name）
_exp = "poppy_walk" if "Walk" in _task else "poppy_stand"


def _best_checkpoint(experiment: str) -> tuple[int, str] | None:
    """在 logs/rsl_rl/<experiment>/ 的所有 run 里，找训练轮数最大的检查点。"""
    root = os.path.join(_REPO, "logs", "rsl_rl", experiment)
    best: tuple[int, str] | None = None
    for path in glob.glob(os.path.join(root, "*", "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        if not m:
            continue
        n = int(m.group(1))
        if best is None or n > best[0]:
            best = (n, path)
    return best


# ---- 4. 没给 --checkpoint 就自动挑一个 ----
if "--checkpoint" not in sys.argv:
    found = _best_checkpoint(_exp)
    if found is None:
        raise SystemExit(
            f"在 logs/rsl_rl/{_exp}/ 下没找到任何 model_*.pt —— 还没有训练结果，先训练再播放。"
        )
    n, path = found
    rel = os.path.relpath(path, _REPO)
    print(f"[play_poppy] 自动选择检查点: {rel}（第 {n} 轮）")
    sys.argv += ["--checkpoint", path]

print(f"[play_poppy] 已注册的任务: "
      f"{[k for k in __import__('gymnasium').registry.keys() if k.startswith('Poppy-')]}")
print(f"[play_poppy] 任务: {_task}")
print(f"[play_poppy] 转交 Isaac Lab 官方入口: {PLAY_PY}")
print(f"[play_poppy] 工作目录: {os.getcwd()}")

# ---- 5. 执行官方脚本 ----
runpy.run_path(PLAY_PY, run_name="__main__")
