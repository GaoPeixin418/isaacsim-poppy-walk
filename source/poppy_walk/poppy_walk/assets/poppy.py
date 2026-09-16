# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 人形机器人资产配置（Isaac Lab 扩展包）。

这个文件只负责一件事：把仓库里已经修正好的 USD 资产包装成 Isaac Lab 的
:class:`ArticulationCfg`，并显式声明执行器参数。

设计取舍说明（这些都是有依据的，不是抄来的）：

1. **USD 路径用"相对本文件推导 + 环境变量可覆盖"**
   相对推导保证 `pip install -e` 之后不管从哪个目录跑都能找到资产；
   环境变量覆盖让同一个包也能用别的资产（比如以后做 sim2sim 用 MuJoCo 版）。
   参考做法是抄别人成功过的框架——这里沿用 Isaac Lab 的 anymal/cassie 资产 cfg
   写法，只换资产和参数。

2. **执行器力矩上限必须显式收紧**
   URDF 里写的 effort 是 3.1 N·m（MX-28），实测堵转力矩只有约 2.5 N·m。
   如果放任 URDF 的值，策略会以为关节比实际更有力，学到"用力顶"的策略。
   髋部 MX-64 用 6.0 N·m。这两个数是 D2 资产审计的产物。

3. **刚度/阻尼的单位是 N·m/rad**（不是 N·m/deg）
   这是 D3 实测结论：`ImplicitActuatorCfg.stiffness` 与 URDF/Isaac Lab 一致用弧度，
   而 USD 里的 `drive:angular:physics:stiffness` 是"度"。
   同一组数在两者之间会差 57.296 倍 —— 抄错一次就会让 PD 软 57 倍。
   （实测证据：配置里写 8.0 N·m/rad，USD 里读到 0.1396 = 8/57.296。）

4. **kp/kd 的取值依据**（见 kp_design_note）
"""
from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# 路径
##

_DEFAULT_USD = Path(__file__).resolve().parents[4] / "assets" / "poppy" / "poppy.usd"
POPPY_USD_PATH = os.environ.get("POPPY_USD_PATH", str(_DEFAULT_USD))

##
# 物理参数（都带出处）
##

#: 执行器力矩上限（N·m）
EFFORT_MX28 = 2.5      # 腿部 MX-28AT 实测堵转力矩（URDF 里写 3.1，偏大）
EFFORT_MX64 = 6.0      # 髋部 MX-64AT

#: kp 的设计思路：kp = 力矩上限 / 设计偏角
#: 设计偏角取 0.2 rad（11.5°）——即"关节承受满力矩时允许偏 11.5°"。
#: 这是个有物理含义的尺度：太小 → 关节像刚性连接、策略失去柔顺；
#: 太大 → 关节软到指令做不出来。
#:   MX-28 组: 2.5 / 0.2 = 12.5
#:   MX-64 组: 6.0 / 0.2 = 30.0   （比值为 2.4，正好等于力矩比，不是拍脑袋的 2.5 倍）
KP_MX28 = 12.5
KP_MX64 = 30.0

#: kd 的设计思路：临界阻尼近似 kd = 2·ζ·sqrt(kp·J_eff)
#: 为什么不能简单地 kd ∝ kp：
#:   单关节 PD 闭环固有频率 ω = sqrt(kp/J)，临界阻尼要求 kd = 2Jω = 2·sqrt(kp·J)。
#:   即 kd 应随 sqrt(kp) 增长。按线性给，kp 变大时会过阻尼（响应迟钝）；
#:   给固定值则欠阻尼（高频抖动，PhysX 里表现为关节嗡嗡响）。
DAMP_ZETA = 1.0
J_EFF_EST = 0.02       # kg·m^2，髋-膝-踝串联的等效惯量量级估计


def kd_for(kp: float, zeta: float = DAMP_ZETA, j_eff: float = J_EFF_EST) -> float:
    """由 kp 反推临界阻尼附近的 kd。"""
    return 2.0 * zeta * (max(kp, 1e-6) * j_eff) ** 0.5


#: 站立姿态的默认关节角：**零位**
#: 依据（D3 静运动学实测）：
#:   · 关节零位 = 腿完全伸直，脚底水平（脚掌倾角 0°）
#:   · 此时站立骨盆高度 0.4210 m（解析）/ 0.42125 m（仿真收敛标定），两者差 0.25 mm
#:   · 质心水平投影落在双脚脚底凸包内，到最近倾覆边 28.7 mm
#:   · 该姿态下各关节承担的重力力矩几乎为 0（实测 < 5e-5 N·m）
#:     —— 也就是说"站不住"不是因为关节顶不住，而是因为平衡本身是倒立摆问题
#: 注意：抬头看 D4 时这里会改成轻微屈膝姿态（提高膝关节可控性），
#:      但屈膝姿态需要重新解算踝补偿（D3 Part2 实测：髋 k/2 + 踝 k/2 并不能让脚掌水平，
#:      脚掌倾角恰好等于 k），所以 D3 先用零位。
STANDING_PELVIS_Z = 0.42125
DEFAULT_JOINT_POS = {
    ".*hip_x": 0.0,
    ".*hip_z": 0.0,
    ".*hip_y": 0.0,
    ".*knee_y": 0.0,
    ".*ankle_y": 0.0,
}

##
# 资产配置
##

POPPY_CFG = ArticulationCfg(
    # 场景里会被替换成 {ENV_REGEX_NS}/Robot；单独跑探针脚本时用这个默认路径
    prim_path="/World/Poppy",
    spawn=sim_utils.UsdFileCfg(
        usd_path=POPPY_USD_PATH,
        activate_contact_sensors=True,     # 训练需要接触传感器（摔倒检测 + 步态奖励）
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
            fix_root_link=False,           # 自由浮动基座 —— 行走任务必须如此
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # 落到地面附近：0.42125 m 是"脚底刚好贴地"的骨盆高度，不带初始下落
        pos=(0.0, 0.0, STANDING_PELVIS_Z),
        # 朝向：把 Poppy 的身体坐标系旋转对齐到 Isaac Lab 约定。
        # (w,x,y,z) = (0.7071, 0, 0, 0.7071) 表示绕 **Z 轴 +90°**：
        #   身体前方（URDF 里是 -Y）→ 世界 +X
        #   身体左侧（URDF 里是 +X）→ 世界 +Y
        # 为什么必须做：Isaac Lab 的世界系里 +X 是"前"、+Y 是"左"，
        # 速度指令 lin_vel_x 的语义就依赖这个。不转的话指令说"往前走"，
        # 机器人实际是往侧面走 —— 这类错位在训练曲线上看不出来，只能靠几何核对。
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
        joint_pos=DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 膝关节/踝关节/髋部侧摆与旋转：MX-28
        "mx28": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
            effort_limit_sim=EFFORT_MX28,
            velocity_limit_sim=7.0,
            stiffness=KP_MX28,
            damping=kd_for(KP_MX28),
        ),
        # 髋部俯仰：MX-64，承重最大
        "mx64_hip_pitch": ImplicitActuatorCfg(
            joint_names_expr=[".*hip_y"],
            effort_limit_sim=EFFORT_MX64,
            velocity_limit_sim=8.2,
            stiffness=KP_MX64,
            damping=kd_for(KP_MX64),
        ),
    },
)
