# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""基于 Manager 的任务集合。

`common/` 不是任务，而是站/走共享的环境骨架（场景、动作、观测、事件、终止、
基础奖励）。两个任务目录只保留自己**真正不同**的部分。

★★ 为什么这里**不**import `common` ★★
    `common/poppy_env_cfg.py` 会 `import isaaclab.sim`，那条链最终会
    `import carb`。而 `carb` 只有等 Omniverse 的 SimulationApp 启动之后才存在。
    一旦在这里 import common，`import poppy_walk` 就会在 AppLauncher 之前
    崩在 `ModuleNotFoundError: No module named 'carb'`。

    所以 `common` 必须**延迟导入**：只有 gym 真正去构建环境、import
    `standing_env_cfg` / `walking_env_cfg` 的那一刻（此时 SimulationApp 已经起来了）
    才把 common 拉进来。这两个任务文件里都有 `from ..common import ...`，
    延迟链路自然成立。

    规律：**这个包里凡是 `import isaaclab.*` 的模块，都必须只在字符串入口点里出现。**
    （本任务目录下的 `*_env_cfg.py` 和 `agents/rsl_rl_ppo_cfg.py` 正好都是这样。）
"""

from . import standing, walking  # noqa: F401

__all__ = ["standing", "walking"]
