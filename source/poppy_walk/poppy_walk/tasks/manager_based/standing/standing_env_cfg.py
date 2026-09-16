# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 站立任务的环境配置（D3 里程碑 1）。

================================================================================
这个任务在做什么
================================================================================
让 Poppy（85 cm / 2.607 kg 仿真质量 / 腿部 10 自由度）在平地上**保持直立静止**。

为什么"站立"本身是个值得单独立项的任务：
    直立是一个**不稳定平衡点**（倒立摆）。D3 实测：把机器人从站立高度放开，
    0.5 s 内倾角只有 0.74°，但 1.35 s 就倒了 —— 时间常数 sqrt(h/g) ≈ 0.22 s。
    也就是说误差以 e^(t/0.22) 增长。纯 PD 只能稳住**关节角**，稳不住**平衡**。
    平衡必须由策略学出来 —— 这正是本任务的内容，也是它作为"里程碑 1"的原因。

================================================================================
和官方 velocity locomotion 模板的关系（诚实说明）
================================================================================
结构完全沿用 Isaac Lab 官方的 velocity locomotion 配置（scene / observation /
action / command / reward / termination / event 七段式，以及
`decimation=4 + dt=0.005 → 50 Hz 控制频率`）。

不同的是：
  · 地形改成纯平面（`terrain_type="plane"`），去掉 height_scanner 与地形课程
    —— Poppy 是平地任务，抬进地形系统只会让配置更难排障
  · **速度指令范围全部设为 0** → 同一个环境从"速度跟踪"退化成"保持静止"。
    所以站立和行走本质是**同一个 MDP**，只是指令分布不同。
  · 奖励项按 Poppy 的物理量级重新设计并写明每项存在的理由
  · 观测拆成 policy / critic 两组 → 非对称 Actor-Critic

================================================================================
★ 这个文件现在只剩 20 行有内容的代码 —— 这是刻意的 ★
================================================================================
场景 / 动作 / 观测 / 事件 / 终止 / 基础奖励全部搬到了 `../common/poppy_env_cfg.py`，
由 `PoppyBaseEnvCfg` 提供。站立任务**不需要 override 任何一项**：
     · 机器人默认姿态 = 资产 cfg 里的零位姿态（直腿，脚底水平）
     · 指令 = 基类的"全 0 指令"
     · 奖励 = 基类的稳定性奖励
行走任务（`../walking/`）才会 override 默认姿态、指令范围和奖励。
这样 D5 调奖励时只改一处，不会出现"改了行走忘了站立"、
也不会出现两个任务的物理参数悄悄不一致。
"""
from __future__ import annotations

from isaaclab.utils import configclass

from ..common import PoppyBaseEnvCfg

__all__ = ["PoppyStandingEnvCfg", "PoppyStandingEnvCfg_PLAY"]


@configclass
class PoppyStandingEnvCfg(PoppyBaseEnvCfg):
    """站立环境主配置：基类的默认值就是为站立设计的，因此这里不 override 任何东西。"""


@configclass
class PoppyStandingEnvCfg_PLAY(PoppyStandingEnvCfg):
    """评估/演示用：环境少、关观测噪声、关随机推力、关质量随机化。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        # 评估时不要随机推力，否则测的是"恢复能力"而不是"保持能力"
        self.events.push_robot = None
