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
STANDING_PELVIS_Z = 0.42125
DEFAULT_JOINT_POS = {
    ".*hip_x": 0.0,
    ".*hip_z": 0.0,
    ".*hip_y": 0.0,
    ".*knee_y": 0.0,
    ".*ankle_y": 0.0,
}

##
# 行走任务的默认姿态（D4）：轻微屈膝
##

#: 为什么行走要屈膝（不能用 D3 那套直腿零位）：
#:   1. 直腿时膝关节处于**奇异位形** —— 腿的雅可比退化，膝盖几乎无法改变腿长，
#:      策略失去了"用膝调高度"这个最主要的控制手段；
#:   2. 真机上直立靠"顶死膝"承重，舵机会长期堵转发热；
#:   3. 行走的摆动腿需要主动抬起，从一个已屈曲的膝开始更自然。
#:
#: 屈膝 k 之后，另外两个俯仰角由两个条件确定（两个条件，两个未知数）：
#:
#:   条件 1 · 位置：脚要留在骨盆正下方
#:       hip_y = +k/2
#:       依据：D3 实测 ∂y_foot/∂hip_y ≈ -0.362 m/rad、∂y_foot/∂knee_y ≈ -0.180 m/rad，
#:             比值 2.01 → 髋补一半正好抵消膝带来的前后位移。
#:             D4 复测残差：k=0.2 时脚相对骨盆 y = -0.0052 m（零位是 -0.0050 m），差 0.2 mm。
#:
#:   条件 2 · 姿态：脚掌要水平
#:       hip_y + knee_y + ankle_y = 0   →   ankle_y = +k/2
#:       依据：D4 实测雅可比（`scripts/d4_foot_frame_solve.py`）
#:             右腿 θ_r = -(hip_y + knee_y + ankle_y)，系数实测 [-1.0000, -1.0000, -1.0000]
#:             左腿 θ_l = +(hip_y + knee_y + ankle_y)，系数实测 [+1.0000, +1.0000, +1.0000]
#:             即三根俯仰轴**同号**相加（髋与踝同向、膝反向）。
#:       实测验证：k=0.05~0.30 全程脚掌倾角 0.0000°（双脚）。
#:
#: ★ 一个被修正的错误结论（记在这里避免以后再犯）★
#:   D3 的报告里曾写「髋 +k/2、膝 -k、踝 +k/2 不能让脚掌水平，倾角恰好等于 k」。
#:   这是**探针脚本自己写错了符号**：`d3_pd_calib.py` 的 build_pose 把镜像符号 s
#:   也乘到了踝上，实际摆出来的姿态是 (k/2, -k, **-k/2**)，角度和 = -k。
#:   按 θ = -Σ 算，倾角当然等于 k —— 解析从来没算错，是测量工具错了。
#:   教训：解析与实测矛盾时，两边都可能错；要回头分别验证各自的前提，
#:         而不是直接采信"实测"那一侧（实测也是一个程序）。
#:
#: 取值依据：k = 0.20 rad ≈ 11.5°，是有腿式机器人默认姿态的常见量级
#:   （太小等于没屈膝，太大则腿长明显缩短、髋力矩需求上升）。
WALK_KNEE_K = 0.20
WALK_HIP_Y = 0.10          # = +k/2
WALK_ANKLE_Y = 0.10        # = +k/2（实测解算值，不是 +k/2 的猜测量）
#: 屈膝 k=0.2 时"脚底刚好贴地"的骨盆高度（m）。
#: 实测方式：悬空摆好姿态 → 量脚底相对骨盆的位置 → 落地自检（基座锁在该高度 +
#: 重力 + 真实接触跑 1 s，脚底最低点落在 ±0.1 mm 内）。
#: 注意它比直腿的 0.42125 m **低** 2.1 mm —— 屈膝确实让腿变短了。
#: （D3 报告里那个"屈膝后站立高度反而升高到 0.43210 m"的疑点，根源还是上面那个
#:   符号 bug：脚掌翘了 11.5°，脚尖/脚跟扎进地里，骨盆自然被顶高。）
WALK_PELVIS_Z = 0.41919


def walk_default_joint_pos() -> dict[str, float]:
    """行走默认姿态的关节角，遵守镜像约定 **右腿 = -左腿**。

    返回 {关节名: 弧度}。Isaac Lab 会把 key 当正则做全名匹配，
    所以这里写全名即可（不会误匹配到别的关节）。
    """
    return {
        ".*hip_x": 0.0,
        ".*hip_z": 0.0,
        "r_hip_y": +WALK_HIP_Y,
        "l_hip_y": -WALK_HIP_Y,
        "r_knee_y": -WALK_KNEE_K,
        "l_knee_y": +WALK_KNEE_K,
        "r_ankle_y": +WALK_ANKLE_Y,
        "l_ankle_y": -WALK_ANKLE_Y,
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
