# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""注册站立任务。

注意这里的两个细节（都是刻意的）：

1. 用 `f"{__name__}...."` 拼字符串，而不是先 `from . import agents` 再
   `f"{agents.__name__}...."`。
   官方模板用的是后者，但那会让"导入本包"时立刻把 agents 模块（进而
   isaaclab_rl / torch / omni）拉进来。而我们要保证 import poppy_walk
   发生在 SimulationApp 启动【之前】，所以必须延迟。

2. 每个任务注册两份：普通版（训练）和 PLAY 版（评估/演示）。
   PLAY 版环境数少、关观测噪声、关随机推力 —— 评估时测的应该是
   "策略保持得多好"，而不是"策略能不能应付随机推力"。
"""

import gymnasium as gym

##
# 站立（D3 里程碑 1）
##

gym.register(
    id="Poppy-Stand-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_env_cfg:PoppyStandingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PoppyStandingPPORunnerCfg",
    },
)

gym.register(
    id="Poppy-Stand-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standing_env_cfg:PoppyStandingEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PoppyStandingPPORunnerCfg",
    },
)
