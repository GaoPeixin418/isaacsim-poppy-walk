# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""PPO 训练超参数（RSL-RL 实现）。

超参数的取值理由都写在参数旁边。初值参照 Isaac Lab 官方的 Cassie/Anymal
velocity 配置（那套是经过大量验证的），只按 Poppy 的规模做了两处调整：
网络变小（观测 39 维、动作 10 维，不需要 512 宽的 MLP）
迭代数减少（单环境并行度 2048，站立任务收敛快）

关于 PPO 本身（面试常问，先说清）：
  PPO 是在"策略梯度"上加一个**裁剪**约束：新策略与旧策略的概率比被限制在
  [1-ε, 1+ε] 内，避免一次更新把策略推得太远导致崩溃。
  本项目用 rsl_rl 的实现，关键旋钮有四个：
    · clip_param (ε)      —— 裁剪范围，默认 0.2
    · num_learning_epochs —— 同一批数据复用几遍
    · desired_kl          —— 自适应学习率的目标 KL；schedule="adaptive" 时
                             每轮根据实际 KL 调 lr，是让训练"不用手动调 lr"的关键
    · entropy_coef        —— 鼓励探索；对站立任务，探索不足会卡在"僵硬直立"的局部解
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PoppyStandingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    #: 每次更新前每个环境采多少步。
    #: batch_size = num_envs × num_steps_per_env = 2048 × 24 = 49152 步/轮。
    #: 24 是官方 locomotion 的常用值：足够覆盖一个动作周期，又不至于让 PPO 的
    #: "同批数据复用" 过期（PPO 是 on-policy，数据越旧越有害）。
    num_steps_per_env = 24

    #: 总迭代数。站立任务比行走简单，1500 轮足够看到收敛或明确失败。
    max_iterations = 1500
    save_interval = 50

    experiment_name = "poppy_stand"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # 观测 39 维、动作 10 维，比人形大机器人小得多。
        # 512 宽的 MLP 在这规模上属于过参数化，既慢又更容易过拟合到随机种子。
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 站立任务的探索需求比行走低；0.005 略低于官方 0.01，
        # 避免策略一直抖（噪声标准差降不下来）。
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        # adaptive + desired_kl：学习率按实际 KL 自动缩放，省掉手工调 lr 这一步。
        # 对新手最容易踩的坑之一就是 lr 固定且偏大 → 前几百轮看着不错然后突然崩。
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
