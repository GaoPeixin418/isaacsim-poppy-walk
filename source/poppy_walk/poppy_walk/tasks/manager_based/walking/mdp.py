# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 行走任务的自定义奖励项（D5）。

为什么不用官方现成的（都试过了）：
  · `feet_air_time_positive_biped`（D4 用的）：**每个控制步**只要"恰好一只脚
    在地上"就给分 —— 单腿跛行（左脚永久悬空）下这项恒定拿满分。
    D5 首次评估实测：左脚离地占比 100%、接触力恒 0 N、右脚独撑。
    它缺"左右脚轮换"这个约束，双足步态里这个约束恰恰是核心。
  · `feet_air_time`（官方落步事件版）：只在**脚落地那一瞬间**给
    `(腾空时长 - 阈值)` —— 跛行状态下没有"重新落地"事件，奖励恒 0，
    作弊动力消失。方向是对的，但不封顶：腾空越久单次奖励越大，
    会把它从"跛行"推向"单脚跳"（跳一下 prolong 腾空再落地）。

所以这里写两个小函数，各补一个缺口：
  · `feet_air_time_landing`  = 官方落步版 + 单次奖励封顶 max_bonus
  · `feet_hover_penalty`     = 单脚腾空超过 max_air_time 后持续罚
    （悬空越久罚越重 —— "永久悬空"从满分变成重罚）

轻量导入约束：本模块 import 了 isaaclab，只能被 walking_env_cfg.py
（即 gym 字符串入口点链上的模块）import，绝不能被包顶层 __init__ 碰到。
"""
from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time_landing(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
    max_bonus: float,
) -> torch.Tensor:
    """落步事件版抬脚奖励：只在脚落地的那一步给 `(腾空时长 - threshold)`，上限 max_bonus。

    与官方 `feet_air_time` 的唯一区别是 `clamp(max=max_bonus)`（上限）：
    不设上限时单次奖励随腾空时长线性增长，策略的最优反应是
    "跳起来延长腾空"而不是"迈更多步"。封顶之后，腾空超过
    threshold+max_bonus 不再有额外收益，落地步频才是收益来源。

    ★ 下限**刻意不设**（v3 的教训，2026-09-16）★
    第一版写了 `.clamp(min=0.0)` —— 想法是"没迈够 0.2 s 就不给分"。
    结果 v3 训练出第三种作弊步态【小碎步蹭走】：每次抬脚都只到
    0.19 s 以下，落地奖励恒 0、悬空惩罚恒 0、打滑惩罚也小 —— 从第
    300 轮起 feet_air_time 死在 0.0001，速度跟踪却有 97%。
    官方原版对短步给【负分】(air-threshold < 0)，正是这个负梯度在
    推着策略把步子迈大。**塑形奖励在当前行为附近必须有梯度，
    "达不到标准就一分不给"等于在局部最优旁边挖了一道护城河。**
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold).clamp(max=max_bonus) * first_contact, dim=1)
    # 指令接近 0（不该迈步）时置 0，防"站着抬脚白拿分"
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_hover_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_air_time: float,
) -> torch.Tensor:
    """悬空超时惩罚：任一脚腾空超过 max_air_time 后，超出部分每步都在罚（封顶 1 s 的超出量）。

    正常步态里单脚腾空 0.2~0.4 s，max_air_time=0.5 时完全不触发；
    而"单腿跛行"里悬空脚的 current_air_time 一路涨到回合结束（20 s），
    惩罚强度远超任何奖励项 —— 这条路被直接封死。

    ★ 返回【正】的超出量，配【负】权重（与 feet_slide 同约定）★
    v4 的教训（2026-09-17）：第一版返回负值 -sum(excess) 又配了负权重 -2.0，
    乘出来是 +2.0/s —— "悬空惩罚"实际成了"悬空奖励"，v4 训出的单腿跛行
    就是我亲手奖励出来的（训练日志铁证：feet_hover ≈ +0.001，恒正）。
    Isaac Lab 奖励 = 权重 × 函数值 × dt，**符号约定必须在写第一行前想清楚**。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = (air_time - max_air_time).clamp(min=0.0, max=1.0)
    return torch.sum(excess, dim=1)


# ================= v6 新增（2026-09-18）=================
# v5 失效模式复盘：节律步态存在，但整体右倾，左脚摆动最低点悬在地面以上
# ~7 mm，整个评估过程接触力恒 0（"踩空"）。训练时的随机推力会让左脚
# 偶尔擦地——擦地瞬间 current_air_time 被清零、还会记一次"假落地"——
# 所以 feet_hover_penalty 和 feet_air_time_landing 都对它失明。
# 零动作探针（scripts/probe_zero.py）已证明资产左右对称（静止脚高差
# 0.13 mm），右倾是策略自己学出来的。v6 从两个方向封堵：


def feet_loading_symmetry(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    ema_alpha: float = 0.99,
    asym_threshold: float = 0.25,
) -> torch.Tensor:
    """双脚承重对称性惩罚（左右脚竖直接触力的慢 EMA 差）。

    用时间常数约 1/(1-alpha)*dt ~= 2 s 的 EMA 分别平滑左右脚竖直接触力，
    惩罚 |EMA_L - EMA_R| / 总体重 超过 asym_threshold 的部分：
      * 正常交替步态：左右 EMA 都围绕 mg/2 波动，短时交替分量被滤掉，
        对称差 << 0.25 -> 不惩罚；
      * 跛行：EMA_L -> 0、EMA_R -> mg，对称差 ~= 1.0 -> 满额惩罚；
      * 偶发假擦地（力度小、时长短）对 2 s EMA 影响可忽略 -> 不失效。
    阈值偏移（asym_threshold）保证惩罚只打"系统性跛行"，不打正常步态
    的瞬时不对称。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fz = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]  # (num_envs, 2)

    # EMA 状态挂在 env 对象上（跨 step 存活；形状变化时自动重建）
    if getattr(env, "_v6_load_ema", None) is None or env._v6_load_ema.shape != fz.shape:
        env._v6_load_ema = fz.clone()
    else:
        # 回合刚开始的 env 重置 EMA，避免上一回合的跛行残差污染新回合
        fresh = env.episode_length_buf <= 1
        if fresh.any():
            env._v6_load_ema[fresh] = fz[fresh]
        env._v6_load_ema.mul_(ema_alpha).add_(fz, alpha=1.0 - ema_alpha)

    # 总体重（kg * 9.81）做无量纲化，缓存一次
    if getattr(env, "_v6_body_weight", None) is None or env._v6_body_weight.device != fz.device:
        masses = env.scene["robot"].root_physx_view.get_masses()
        env._v6_body_weight = masses.sum(dim=1).to(fz.device) * 9.81

    asym = (env._v6_load_ema[:, 0] - env._v6_load_ema[:, 1]).abs() / env._v6_body_weight
    return (asym - asym_threshold).clamp(min=0.0, max=1.0)


def base_roll_penalty(
    env: ManagerBasedRLEnv,
    max_roll: float = 0.15,
) -> torch.Tensor:
    """基座横滚（侧倾）惩罚。

    v5 的"左脚踩空 7 mm"根因是策略学出的右倾。本项给侧倾角一个线性
    代价，把重心往双脚中间拉。横滚角从 projected_gravity_b 的 y 分量
    反解：直立时 g_b=(0,0,-1)；纯横滚 phi 时 g_b=(0,-sin phi,-cos phi)，
    故 phi = asin(-g_y)。返回值 clamp 到 [0,1]，配负权重使用。
    """
    g_b = env.scene["robot"].data.projected_gravity_b
    roll = torch.asin((-g_b[:, 1]).clamp(-1.0, 1.0)).abs()
    return (roll / max_roll).clamp(max=1.0)
