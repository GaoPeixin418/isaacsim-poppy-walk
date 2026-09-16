# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""poppy_walk —— 基于 Isaac Lab 的 Poppy 人形机器人行走/站立任务扩展包。

设计要点（为什么这样组织，而不是把配置塞进 isaaclab_tasks 里）：
  1. 不修改 Isaac Lab 源码 → 升级 Isaac Lab 不会冲突，装到别的机器上也不用重新打补丁。
  2. **导入本包必须是"轻量"的**：Omniverse 要求所有 omni/pxr 模块都在
     SimulationApp 启动之后才能导入，否则会出警告甚至崩溃。
     所以这里只做 gym.register，而把 env_cfg / agent_cfg 写成**字符串**延迟导入
     （gym 在真正构建环境时才去 import 那个模块）。
     这也是为什么各任务目录下的 __init__.py 里不写 `from . import agents`。
"""

from . import tasks  # noqa: F401

__all__ = ["tasks"]
