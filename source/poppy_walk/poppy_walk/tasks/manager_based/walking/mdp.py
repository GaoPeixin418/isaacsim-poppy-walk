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
    """落步事件版抬脚奖励：只在脚落地的那一步给 `(腾空时长 - threshold)`，封顶 max_bonus。

    与官方 `feet_air_time` 的唯一区别是 `clamp(max=max_bonus)`：
    没有封顶时，单次奖励随腾空时长线性增长，策略的最优反应是
    "跳起来延长腾空"而不是"迈更多步"（速率 (air-t0)/air 随 air 单调升）。
    封顶之后，腾空超过 threshold+max_bonus 就不再有额外收益，
    多落地一次 = 多拿一次满分，迈步的步频才成为唯一收益来源。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum(((last_air_time - threshold).clamp(min=0.0, max=max_bonus)) * first_contact, dim=1)
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

    返回负值（惩罚），使用时配负权重。
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = (air_time - max_air_time).clamp(min=0.0, max=1.0)
    return -torch.sum(excess, dim=1)
