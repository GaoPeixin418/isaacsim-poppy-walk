# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""站/走两个任务共享的环境骨架。

为什么要有这一层（而不是让行走任务直接抄一份站立任务的配置）：
    Isaac Lab 官方的做法就是「一个通用基类 + 每个机器人写 override」——
    见 `isaaclab_tasks/.../locomotion/velocity/velocity_env_cfg.py` 里定义的
    `LocomotionVelocityRoughEnvCfg`，Cassie / H1 / G1 各自继承它、只改自己关心的项。
    好处很实际：
      · 稳定性项（姿态、高度、力矩正则、终止条件）只写一份 —— D5 调奖励时不会
        "改了站立忘了行走"，也不会出现两个任务的物理参数悄悄不一致；
      · 每个任务的配置文件只保留**真正不同**的部分，读起来就是一份"改动清单"。
"""

from .poppy_env_cfg import (  # noqa: F401
    BASE_BODY,
    FALLEN_BODIES,
    FOOT_BODIES,
    ActionsCfg,
    CommandsCfg,
    EventCfg,
    ObservationsCfg,
    PoppyBaseEnvCfg,
    PoppySceneCfg,
    RewardsCfg,
    TerminationsCfg,
)

__all__ = [
    "BASE_BODY",
    "FALLEN_BODIES",
    "FOOT_BODIES",
    "ActionsCfg",
    "CommandsCfg",
    "EventCfg",
    "ObservationsCfg",
    "PoppyBaseEnvCfg",
    "PoppySceneCfg",
    "RewardsCfg",
    "TerminationsCfg",
]
