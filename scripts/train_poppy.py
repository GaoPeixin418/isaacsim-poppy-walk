#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 训练入口：注册自定义任务 → 复用 Isaac Lab 官方 rsl_rl 训练脚本。

================================================================================
为什么写成这样一个"薄壳"，而不是直接改 Isaac Lab 的 train.py
================================================================================
Isaac Lab 的 train.py 里有一行注释：

    # PLACEHOLDER: Extension template (do not remove this comment)

它的意思是"扩展包应该在这里被 import"。官方推荐的做法就是去改这一行。
我们不这么做，理由有两条：
  1. **依赖方向**：改了 Isaac Lab 源码，我们的代码就和它的安装路径绑死了。
     换机器要再改一次，升级 Isaac Lab 要重打一次补丁。保持上游原样、
     由我们在外层 import，依赖方向才是干净的（我们依赖它，它不依赖我们）。
  2. **可说明性**：面试时能明确说出"哪些是我写的、哪些是框架提供的"。
     改动集中在自己的包里，diff 一目了然。

================================================================================
为什么 import poppy_walk 可以放在 AppLauncher 之前
================================================================================
Omniverse 要求所有 omni/pxr 模块都在 SimulationApp 启动【之后】才能导入
（否则出现 "Modules were loaded before SimulationApp was started" 警告，
严重时直接崩）。poppy_walk 的导入只做 gym.register，并且把 env_cfg / agent_cfg
写成字符串延迟导入，因此不会提前拉起 torch / omni —— 这是刻意的结构设计，
详见 poppy_walk/__init__.py 与 tasks/manager_based/standing/__init__.py。

================================================================================
用法
================================================================================
仓库根目录下：

    bash scripts/run.sh scripts/train_poppy.py --headless \\
        --num_envs 2048 --max_iterations 1500

不传 --task 时默认使用 Poppy-Stand-v0。
Isaac Lab 的安装位置用环境变量 ISAACLAB_ROOT 指定（默认 /data/poppy/src/IsaacLab）。
"""
from __future__ import annotations

import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# ---- 1. 让 poppy_walk 可被导入（仓库内开发模式） ----
_SRC = os.path.join(_REPO, "source", "poppy_walk")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import poppy_walk  # noqa: E402,F401  (gym.register 在这里发生)

# ---- 2. 定位 Isaac Lab 的官方训练脚本 ----
ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/data/poppy/src/IsaacLab")
TRAIN_PY = os.path.join(ISAACLAB_ROOT, "scripts", "reinforcement_learning", "rsl_rl", "train.py")
if not os.path.isfile(TRAIN_PY):
    raise SystemExit(
        f"找不到 Isaac Lab 的训练脚本: {TRAIN_PY}\n"
        f"请用环境变量指定 Isaac Lab 目录，例如：\n"
        f"    ISAACLAB_ROOT=/path/to/IsaacLab bash scripts/run.sh scripts/train_poppy.py --headless"
    )

# ---- 3. 补上 runpy 不会自动做的事 ----
# 直接 `python train.py` 时，解释器会把脚本所在目录放进 sys.path，
# 所以 train.py 里的 `import cli_args`（同目录文件）能找到。
# 而 runpy.run_path 不会修改 sys.path，必须手动补，否则 ImportError。
_SCRIPT_DIR = os.path.dirname(TRAIN_PY)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 默认任务，省掉每次都要写 --task
if "--task" not in sys.argv:
    sys.argv = [sys.argv[0], "--task", "Poppy-Stand-v0"] + sys.argv[1:]

print(f"[train_poppy] 已注册的任务: "
      f"{[k for k in __import__('gymnasium').registry.keys() if k.startswith('Poppy-')]}")
print(f"[train_poppy] 转交 Isaac Lab 官方入口: {TRAIN_PY}")
print(f"[train_poppy] 工作目录: {os.getcwd()}（日志会写到 <仓库>/logs/rsl_rl/ 下）")

# ---- 4. 执行官方脚本（等价于 `python train.py <args>`） ----
runpy.run_path(TRAIN_PY, run_name="__main__")
