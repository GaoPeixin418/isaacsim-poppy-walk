# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""注册行走任务（D4）。

和站立任务（`../standing/__init__.py`）完全同构，两个细节仍然刻意保留：

1. `env_cfg_entry_point` / `rsl_rl_cfg_entry_point` 用 `f"{__name__}...."` 拼字符串，
   而不是先 `from . import agents` —— 保证 `import poppy_walk` 是轻量导入，
   不把 torch / omni 拉进 SimulationApp 启动之前。

2. 注册训练版 + PLAY 版两份。PLAY 版把速度指令固定成定速前进，
   用于评估/录屏时的变量控制。
"""

import gymnasium as gym

##
# 行走（D4，里程碑 2 的基础）
##

gym.register(
    id="Poppy-Walk-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:PoppyWalkEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PoppyWalkPPORunnerCfg",
    },
)

gym.register(
    id="Poppy-Walk-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.walking_env_cfg:PoppyWalkEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PoppyWalkPPORunnerCfg",
    },
)
