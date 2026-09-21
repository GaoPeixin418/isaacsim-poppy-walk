# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 行走任务的环境配置（D4）。

================================================================================
和站立任务（D3）的关系
================================================================================
**站立和行走是同一个 MDP，区别只在指令分布和奖励。** 这不是巧合，是刻意设计的：
D3 就用官方的 velocity locomotion 结构，把速度指令范围设成全 0。D4 要做的事因此
只有三件：

  1. 打开速度指令（`WalkCommandsCfg`）
  2. 把默认姿态从"直腿零位"换成"轻微屈膝"（`walk_default_joint_pos()`）
  3. 在稳定性奖励的"地基"之上加两项步态奖励（`WalkRewardsCfg`）

场景 / 动作 / 观测 / 事件 / 终止条件全部继承自 `../common/poppy_env_cfg.py`，
一行都不用改。这么做的好处很具体：D5 调奖励、D6 加域随机化时只改一个地方，
不会出现"行走改好了、站立悄悄失效"。

================================================================================
默认姿态为什么要屈膝（这是 D4 的第一件事）
================================================================================
屈膝量 k = 0.20 rad（11.5°），另外两个俯仰角由两个条件解出来：

  位置：hip_y = +k/2 = 0.100   →  脚留在骨盆正下方（残差 0.2 mm）
  姿态：ankle_y = +k/2 = 0.100 →  脚掌水平（实测倾角 0.0000°）

实测依据见 `assets/poppy.py` 里 `WALK_KNEE_K` 附近的注释，以及
`scripts/d4_foot_frame_solve.py`（实测雅可比 = [-1,-1,-1] / [+1,+1,+1]）。
注意 ankle 与 hip **同号**、与 knee 反号 —— 这一条差点被一个探针 bug 带错，
详细经过写在 assets/poppy.py 的注释和 docs/D4-walk-report.md 里。

================================================================================
奖励设计：每一项防的是什么作弊解
================================================================================
先把"主任务项"定下来（速度跟踪），然后每加一项都问：**不加会学出什么坏行为？**

| 奖励项 | 权重 | 不给会怎样（作弊解） |
|---|---|---|
| track_lin_vel_xy_exp   | +1.5 | 站着不动 / 原地小碎步 |
| track_ang_vel_z_exp    | +0.75 | 走偏、原地转圈 |
| base_height_l2 (−6.0)  | 软约束 | 蹲着走（质心低更好平衡，是行走最经典的作弊解）；行走要允许上下起伏，所以比站立的 −12 松 |
| flat_orientation_l2    | −1.5 | 前倾趴着走 / 用躯干配重；比站立的 −3 松，因为步态本身有俯仰摆动 |
| lin_vel_z_l2           | −2.0 | 一蹦一蹦地跳（也是"作弊"的一种：跳比走省事） |
| ang_vel_xy_l2          | −0.5 | 上身乱晃换取步幅 |
| joint_deviation_l1     | −0.3 | 关节停在极端角（如膝盖一直顶限位）凑出奇怪的姿态；权重比站立小很多，因为行走**本来就需要**大幅偏离默认姿态 |
| dof_pos_limits         | −2.0 | 关节长期顶限位（顶住时力矩最大最省事）—— 真机上舵机会烧 |
| dof_torques_l2         | −2e-4 | 用力矩硬顶；Poppy 舵机力矩本来就小，要鼓励省力的解 |
| dof_acc_l2             | −1e-7 | 关节加速度过大（真机上会让舵机齿轮受冲击） |
| action_rate_l2         | −0.02 | 控制量高频抖动（50 Hz 下的相邻动作差） |
| undesired_contacts     | −1.0 | 用躯干/大腿触地"爬"过去（同时有终止项兜底） |
| **feet_air_time**      | **+2.0** | **最重要的步态项**（落步事件版）：只在脚落地那一瞬给 `(腾空−0.2)`、封顶 0.15。不给会学"滑步"（两脚贴地蹭）；用"每步都给分"的版本（D4）会学"单腿跛行"（一只脚永久悬空白拿满分，D5 评估实测抓到） |
| **feet_hover**         | **−2.0** | 悬空超时惩罚：腾空 >0.5 s 后持续罚 —— 直接封死"永久悬空"（正常步态腾空 0.2~0.4 s 不触发） |
| **feet_slide**         | **−0.2** | 脚在地上滑（与 air_time 互补：一个管"抬起来"，一个管"落地后别蹭"） |

`feet_air_time` / `feet_hover` 是本包自写的（`walking/mdp.py`），各自的"为什么不用官方版"
写在那个文件里 —— 一句话：官方双足版缺"左右轮换"约束，官方落步版缺"腾空时长封顶"。

> ★ **奖励权重怎么看（D4 学到的）**：权重的绝对值没有意义，
> **只有"该项的理论最大值 与 主任务项理论最大值 的比例"才有意义**。
> 理论最大值 = 权重 ×（该项量纲上的上限），例如
> `feet_air_time` = 权重 × threshold、速度跟踪 = 权重 × 1.0。
> 第一次训练把 `feet_air_time` 给成 1.0×0.3 = 0.3 /s，只占速度跟踪 1.5 /s 的 20%，
> 结果策略 200 轮就学会"不迈步、靠前倾+蹭地"，第 300 轮就平台化。

================================================================================
指令范围为什么这么小
================================================================================
  lin_vel_y ∈ (0.0, 0.35) m/s   正常行走速度。Poppy 腿长只有 0.386 m，
                                步频/步幅的物理上限都低，"走得快"不是这个项目的目标。
  lin_vel_x = (0.0, 0.0)        **刻意锁死为 0**。Poppy 的踝关节只有俯仰、
                                髋侧摆只有 ±30°，**横向是完全欠驱动的**，
                                要求它侧移只会让策略在两个互相矛盾的目标间折中。
                                把 x 指令锁 0，这一项反而变成一个有用的"别横向漂移"惩罚
                                （track_lin_vel_xy_exp 用的是 xy 范数）。

  ⚠ v8 坐标系更正（2026-09-20，相机标定实验实证）：
  **基座坐标系比解剖学朝向歪了 90°——解剖学正前方 = 基座 +y，不是 +x。**
  v1–v7 一直把"前进"指令发在 x 通道上，等于命令机器人"向自己的右侧横移"，
  策略精确执行了这条歪指令，学出一套 ~3 Hz 横向交叉步（蟹行角实测 −0.6°，
  全部判卷指标对朝向盲，一路绿灯；肉眼看视频一帧就抓到）。
  v8 起把前进指令换到 y 通道：锁定 x=0（防横漂）、y∈(0, 0.35)（真·前进）。
  原先"锁 y 防侧移"的设计意图本身是对的，只是当时贴错了通道。
  ang_vel_z ∈ (−0.2, 0.2) rad/s 慢速转向。偏航靠 hip_z（±90° 行程）实现，做得到但不轻松。
                                给太大会把注意力从"走起来"引开，D5 再按需要放开。
"""
from __future__ import annotations

import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import feet_slide

from .mdp import (feet_air_time_landing, feet_hover_penalty,
    feet_loading_symmetry, base_roll_penalty, single_leg_lean_done)

from ....assets.poppy import POPPY_CFG, WALK_PELVIS_Z, walk_default_joint_pos
from ..common import FOOT_BODIES, CommandsCfg, PoppyBaseEnvCfg, RewardsCfg, TerminationsCfg as CommonTerminationsCfg

##
# 指令
##


@configclass
class WalkCommandsCfg(CommandsCfg):
    """把速度指令从"全 0"改成"采样"。这是行走与站立唯一的 MDP 层差别。

    `resampling_time_range=(10.0, 10.0)`：每 10 秒重采一次指令。
      为什么不是更短：一个指令至少要让策略走完几个完整步态周期再换，
      否则它学的是"如何应付变化"而不是"如何跟踪"。
      也不是更长：10 s 已经覆盖了一个回合（`episode_length_s=20`）的一半，
      足够多样。

    `rel_standing_envs=0.0`：不让一部分环境拿到 0 指令。
      原因见上面 air_time 的说明 —— 指令接近 0 时步态奖励被门控掉，
      混入 0 指令会让同一批数据里"该走"和"该站"的样本互相干扰。
      站立能力已经由 D3 单独训过，不需要在这里混着练。
    """

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,   # 直接采样 ang_vel_z，不用"航向角跟踪"那层间接目标
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(-0.35, 0.0),
            ang_vel_z=(-0.2, 0.2),
            heading=(0.0, 0.0),
        ),
    )


##
# 奖励：基类稳定性项 + 两项步态项
##


@configclass
class WalkRewardsCfg(RewardsCfg):
    """在站立那套"地基"上做三处放松 + 加两项步态奖励。

    三处放松（而不是重写）的理由：站立的那些权重是在"必须纹丝不动"的前提下定的，
    行走时躯干本来就会有周期性摆动、关节本来就会大幅运动，
    沿用站立的值会让策略不敢动 —— 表现为"能站稳但迈不开腿"。
    """

    # ---- 松弛版稳定项（重新声明而不是改基类的对象，避免共享可变默认值） ----
    # 高度：站立是 −12（近乎硬约束），行走给 −6，允许步态带来的上下起伏
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-6.0,
        params={"target_height": WALK_PELVIS_Z, "asset_cfg": SceneEntityCfg("robot")},
    )
    # 躯干水平：站立 −3（几乎不许倾），行走 −1.5（步态本身要前后俯仰）
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    # 默认姿态偏离：站立 −1.0，行走 −0.3（行走必须大幅偏离默认姿态才迈得开）
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ---- 步态项（D5 第二次修订：换掉可被"单腿跛行"攻破的奖励结构） ----
    #
    # ★ D4→D5 的教训：`feet_air_time_positive_biped` 的漏洞 ★
    #   它**每个控制步**只要"恰好一只脚在地上"就给分 —— 这意味着把一只脚
    #   永久抬着就能恒定拿满分。D5 首次量化评估抓到了实证：
    #       左脚离地时间占比 100%、足底接触力恒 0.00 N（全程悬空）、
    #       右脚独撑全部体重（21.3 N ≈ 2.6 kg × 9.8）—— 单腿跛行。
    #   速度跟踪 0.238 m/s、存活 100% 都是真的，唯独步态是坏的。
    #
    # 修复（一次改动，同一个目标"给奖励补上轮换约束"）：
    #   1. 换成【落步事件版】feet_air_time_landing（自写，官方 feet_air_time
    #      的封顶变体）：只在脚落地那一瞬给 (腾空 - 阈值)、封顶 max_bonus。
    #      跛行状态下没有"重新落地"事件 → 该项恒 0，作弊动力消失。
    #      封顶防的是下一个作弊解：不封顶时单次奖励随腾空时长线性涨，
    #      策略会转向"单脚跳 prolong 腾空"。
    #      ★ 短步给负分（官方语义）**不能剪掉**：v3 试过 clamp(min=0)，
    #      结果训出第三种步态"小碎步蹭走"（每步 <0.2 s、所有步态项恒 0，
    #      第 300 轮起 air_time 死在 0.0001、速度跟踪却 97%）。
    #      负分是唯一把步子推大的梯度。详见 mdp.py 的函数 docstring。
    #   2. 新增悬空超时惩罚 feet_hover_penalty：腾空 > 0.5 s 后每步都罚、
    #      越悬越重（超时量封顶 1 s，惩罚 -2.0/s 封顶）。
    #      正常步态腾空 0.2~0.4 s 完全不触发；"永久悬空"从满分变成重罚。
    #
    # 量级核算（沿用"理论最大值 ÷ 主任务理论最大值"原则）：
    #   健康步态（腾空 ~0.3 s、阈值 0.2、封顶 0.15）：每次落地拿 ~0.10，
    #   每秒落地 ~2/0.7 ≈ 2.9 次 → 未加权 ~0.29 /s；权重 2.0 → ~0.58 /s
    #   vs 速度跟踪 1.5 /s → 占 ~39%，与官方 Cassie（37.5%）同量级。
    feet_air_time = RewTerm(
        func=feet_air_time_landing,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "command_name": "base_velocity",
            "threshold": 0.2,
            "max_bonus": 0.15,
        },
    )
    # 悬空超时（防单腿跛行的直接封条；见上）
    feet_hover = RewTerm(
        func=feet_hover_penalty,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "max_air_time": 0.5,
        },
    )
    # 脚打滑：脚在接触状态下的水平速度范数。与 air_time 互补 ——
    # air_time 管"该抬的时候抬起来"，feet_slide 管"落地之后别蹭着走"。
    # 权重 −0.1 → −0.2：第一次训练的策略正是靠"蹭着走"拿到了 66% 的速度跟踪奖励，
    # 说明"蹭"这条路的性价比太高，需要加大它的代价。
    feet_slide = RewTerm(
        func=feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES),
        },
    )
    # ---- v6（2026-09-18）：承重对称 + 横滚，封堵"右倾踩空"失效模式 ----
    #
    # v5 失效：节律步态但整体右倾，左脚摆动最低点距地 ~7 mm 全程踩空
    # （接触力恒 0）。训练时随机推力让左脚偶发擦地，重置 air_time 计时
    # 还记"假落地"，使 feet_hover / feet_air_time 对其失明；探针已证明
    # 资产对称（脚高差 0.13 mm），右倾是策略学出来的。
    #   * feet_loading：2 s EMA 左右脚竖直力差 > 0.25 倍体重才罚，
    #     正常交替步态不触发，跛行满额 -2.0/s；
    #   * base_roll：横滚线性惩罚（0.15 rad 满额），把重心拉回双脚中间。
    feet_loading = RewTerm(
        func=feet_loading_symmetry,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "ema_alpha": 0.99,
            "asym_threshold": 0.25,
        },
    )
    base_roll = RewTerm(
        func=base_roll_penalty,
        weight=-1.0,
        params={"max_roll": 0.15},
    )


##
# 环境
##


@configclass
class WalkTerminationsCfg(CommonTerminationsCfg):
    """行走任务终止条件 = 通用终止 + 单腿倾斜判死 (v7)."""

    single_leg_lean = DoneTerm(
        func=single_leg_lean_done,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "min_loading": 0.18,
            "warmup_s": 2.5,
        },
    )


@configclass
class PoppyWalkEnvCfg(PoppyBaseEnvCfg):
    """行走环境主配置。"""

    commands: WalkCommandsCfg = WalkCommandsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    terminations: WalkTerminationsCfg = WalkTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # 把机器人换成"轻微屈膝"的默认姿态，并把基座放在对应的站立高度上。
        # 为什么连高度也要改：reset_root_state_uniform 是在"默认根状态"上加偏移，
        # 如果默认高度还是直腿的 0.42125 m，而脚因为屈膝只到 0.41919 m，
        # 每次 reset 都会先掉 2 mm 再落地 —— 这点冲击会被策略当成观测噪声。
        self.scene.robot = POPPY_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot",
            init_state=POPPY_CFG.init_state.replace(
                pos=(0.0, 0.0, WALK_PELVIS_Z),
                joint_pos=walk_default_joint_pos(),
            ),
        )


@configclass
class PoppyWalkEnvCfg_PLAY(PoppyWalkEnvCfg):
    """评估/演示用：环境少、关噪声、关随机推力、指令固定成定速前进。

    指令固定成 (0.25 m/s, 0, 0) 的意义：评估"能不能稳定跟踪一个速度"时，
    指令应该是常量。随机指令测的是鲁棒性（D6 的事），不是本次要验收的目标
    —— 变量控制住，看视频才知道"它走得不好"到底是策略问题还是指令太难。
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, -0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
