# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""行走任务的 PPO 训练超参数。

和站立任务（`standing/agents/rsl_rl_ppo_cfg.py`）的区别只有三处，
每一处都对应一个具体的"行走比站立难在哪"：

1. `max_iterations` 1500 → 3000
   行走要同时学两件事：**周期性的步态**（什么时候抬哪只脚）和**平衡**。
   步态是一个需要"时序"的技能，站立不需要。经验上行走的收敛轮数是站立的 2 倍以上。

2. `entropy_coef` 0.005 → 0.01
   行走存在大量局部最优解（原地小碎步、单腿蹦、拖地滑行），
   探索不足会卡在其中一个。熵系数是唯一直接鼓励"保持随机性"的旋钮。
   注意不能一味加大：太大会让策略一直抖，学不出干净的周期。

3. `save_interval` 50 → 100
   行走的检查点更大更多，落盘太频会拖慢训练（实测 3090 上 2048 环境能跑到
   4 万步/秒，磁盘 IO 是少数几个能把吞吐打下来的东西）。

其余超参（clip_param 0.2 / num_learning_epochs 5 / num_mini_batches 4 /
adaptive lr with desired_kl 0.01）沿用官方 locomotion 的经过验证的值。
"""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PoppyWalkPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    #: batch_size = 2048 × 24 = 49152 步/轮
    num_steps_per_env = 24

    #: 行走比站立难，给一倍余量
    max_iterations = 3000
    save_interval = 100

    experiment_name = "poppy_walk"
    empirical_normalization = False

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # 观测 39 维（policy）/ 43 维（critic）、动作 10 维 —— 小网络足够
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 比站立高：行走的局部最优更多，需要更多探索
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
