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
  · **速度指令范围全部设为 0** → 同一个环境从"速度跟踪"退化成"保持静止"，
    D4 行走只需把 ranges 打开，不需要另写一套环境
  · 奖励项按 Poppy 的物理量级重新设计并写明每项存在的理由（见 RewardsCfg）
  · 观测拆成 policy / critic 两组 → 非对称 Actor-Critic（见 ObservationsCfg）

================================================================================
单位与坐标系（踩过的坑写在这里，避免重新踩）
================================================================================
  · ImplicitActuatorCfg 的 stiffness 单位 = **N·m/rad**；
    USD 里 `drive:angular:physics:stiffness` = **N·m/deg**。两者差 57.296 倍。
  · 世界系：+X 前、+Y 左、+Z 上（Poppy 资产已通过 init_state.rot 对齐）
  · 关节限位（rad，D3 实测，已逐项核对镜像关系）：
        hip_x  [-0.524, +0.497] / 镜像
        hip_z  [-0.436, +1.571] / 镜像
        hip_y  [-1.815, +1.466] / 镜像
        knee_y [-2.339, +0.061] / 镜像   ← 膝盖【单向】弯曲，这是防"反关节"的物理约束
        ankle_y [-0.785, +0.785]
"""
from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from ....assets.poppy import POPPY_CFG, STANDING_PELVIS_Z

##
# 一些从 D2/D3 实验里量出来的常量
##

#: 站立时骨盆高度（m）。由"基座锁定 → 量脚底离地 → 下移 → 迭代收敛"实测得到。
#: 用途：reset 时把机器人放在正好贴地的高度（否则每次 reset 都要掉一下，
#: 落地冲击会被策略当成噪声，早期学习明显变慢）。
STANDING_HEIGHT = STANDING_PELVIS_Z

#: 关节零位就是站立姿态（见 assets/poppy.py 的说明）。默认关节角在资产 cfg 里定义，
#: 动作层用 use_default_offset=True 直接引用它，所以这里不再重复一份。

#: 各刚体的语义名字。注意 Poppy 的根刚体叫 pelvis，不叫 base —— 
#: 官方配置里到处是 "base"，直接抄会静默失效（SceneEntityCfg 找不到 body 会报错，
#: 但如果写成 ".*base" 这种正则就会匹配到空集合，奖励项直接恒为 0，很难发现）。
BASE_BODY = "pelvis"
FOOT_BODIES = ["l_foot", "r_foot"]
#: 除了脚以外，任何触地的刚体都算"摔倒"
FALLEN_BODIES = ["pelvis", "l_hip", "r_hip", "l_thigh", "r_thigh", "l_shin", "r_shin"]


##
# 场景
##


@configclass
class PoppySceneCfg(InteractiveSceneCfg):
    """平地 + 一个机器人 + 接触传感器 + 灯光。"""

    # 平地。用 plane 而不是 terrain generator：
    # 平地任务的物理没有地形变量，去掉生成器既省显存也少一个出错源。
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    # 接触传感器：
    #   history_length=3  —— 用于"最近 3 个物理步内是否接触"，比瞬时值抗噪
    #   track_air_time=True —— 记录抬脚/落脚的空中时间，D4 的步态奖励要用
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0, color=(0.85, 0.85, 0.9)),
    )


##
# 动作
##


@configclass
class ActionsCfg:
    """关节位置控制（位置目标），PD 在仿真内部隐式完成。

    为什么用关节位置而不是力矩：
      MX-28/MX-64 是带内部位置控制的舵机，真机接口本来就是"给角度目标"。
      用位置动作 → 策略的输出量与真机接口一致，后面要做 sim2real 时不用改动作层。

    use_default_offset=True 的含义：策略输出的是**相对默认姿态的偏移**，
      即 q_target = q_default + scale * a。
      好处：策略输出 0 就已经是"站在默认姿态"了，探索从合理位置开始；
      坏处：默认姿态一变，同一个策略的输出含义就变了（配置耦合，必须写进文档）。

    scale 的取值：0.5 rad 是官方常用值。Poppy 的膝盖单向行程 2.34 rad、
      脚踝 ±0.785 rad，±0.5 rad 的输出范围对站立和慢走都够用。
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


##
# 速度指令
##


@configclass
class CommandsCfg:
    """速度指令。D3 全部为 0 → 任务退化成"保持静止且不出位移"。"""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,      # D3 不需要转向；D4 打开（会和 ang_vel_z 联动）
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(0.0, 0.0),
        ),
    )


##
# 观测
##


@configclass
class ObservationsCfg:
    """两组观测：policy（策略可见）/ critic（特权，只有价值网络能看到）。

    为什么要有 critic 组（非对称 Actor-Critic）：
      · 基座线速度 base_lin_vel 在真机上需要状态估计（腿式里程计）才能得到，
        而且噪声很大。策略不该依赖它。但价值网络只是在训练时估回报，
        给它更准的状态能让 advantage 估计方差更小，学得更快。
      · 这是一个"训练时可以用、部署时不能用"的信息 → 正好放进 critic。
    实现上不需要额外配置：Isaac Lab 的 rsl_rl 包装器只要看到名为 "critic" 的
    观测组，就会自动设置 num_privileged_obs 并把该组喂给 critic。
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """策略观测（对应真机可得量）。顺序即拼接顺序，改动会破坏已训好的模型。"""

        # 基座角速度：IMU 直接可测
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        # 投影重力：把重力方向投到机身系，等价于"姿态角"的无奇异表述；IMU 可测
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 速度指令：D3 恒为 0，但保留它 → D4 打开 ranges 就能直接用
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        # 关节相对默认姿态的角度：舵机自带位置反馈
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # 关节速度：差分或舵机反馈，噪声大，所以噪声区间给得宽（±1.5 rad/s）
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 上一步动作：让策略知道自己刚做了什么，显著改善平滑性
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """特权观测：策略看不到，只喂给价值网络。"""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)          # ← 特权：真机需状态估计
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        base_height = ObsTerm(func=mdp.base_pos_z)              # ← 特权：真机需估计

        def __post_init__(self):
            self.enable_corruption = False      # 价值网络要的是准状态，不加噪声
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


##
# 奖励
##


@configclass
class RewardsCfg:
    """站立任务的奖励。每一项都写清"为什么存在"和"去掉会怎样"。

    权重不是拍脑袋的：先按物理量级给初值，跑 30 次迭代后打印分项量级，
    再把每项调到"与主任务项同量级或更小"（见 docs/D3-standing-report.md）。
    """

    # ---------------- 任务项 ----------------
    # 指令速度恒为 0 → 这一项等价于"不许平移"。
    # 用 exp(-误差²/std²) 而不是 -误差²：指数形式的梯度在远离目标时会饱和，
    # 避免机器人刚摔倒时被巨大的速度误差惩罚淹没其他信号。
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # 高度保持：站立的定义里包含"站得高"。
    # 没有这一项，策略会发现"蹲下去"（甚至趴下）也很稳 —— 而且蹲姿的质心更低、
    # 反而更好平衡，于是奖励更高 → 学会蹲着不动。这是站立任务最经典的作弊解。
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-12.0,
        params={"target_height": STANDING_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )

    # ---------------- 稳定项 ----------------
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-3.0)
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.5)

    # ---------------- 正则项 ----------------
    # 偏离默认姿态惩罚：把"贴近自然站立姿态"变成一个有价值的行为。
    # 权重不大（关节误差 0.1 rad 才扣 0.1 分），因为平衡本身需要脚踝偏离默认姿态，
    # 权重过大会和 track 项打架。
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    # 关节限位惩罚：这是"防作弊"项。不用这一项，策略会让关节长期顶在限位上
    # （顶住时力矩最大，是最"省事"的平衡方式），但真机上舵机长期堵转会烧。
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
    # 力矩正则：Poppy 的舵机力矩小，鼓励"省力的解"
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    # 动作变化率：抑制抖动。注意这是**控制频率 50 Hz 下的相邻动作差**，
    # 权重过大是"机器人反应迟钝"的常见原因。
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.02)

    # ---------------- 防作弊 ----------------
    # 躯干/大腿触地惩罚（同时有终止项兜底）
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FALLEN_BODIES),
            "threshold": 1.0,
        },
    )


##
# 终止条件
##


@configclass
class TerminationsCfg:
    """终止条件。

    没有终止条件的后果：摔倒后状态继续产生奖励 → 策略学会"躺着蹭分"。
    但终止也不能滥用：过于敏感会让策略因为怕摔倒而不敢动。
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 躯干或大腿触地 = 已经摔倒，立刻重开
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FALLEN_BODIES),
            "threshold": 1.0,
        },
    )
    # 姿态终止：倾角超过 50° 就重开。
    # 为什么是 50° 而不是 30° 或 60°（这是实测定的，不是拍的）：
    #   · 绕脚边翻倒时，骨盆高度 = 0.42·cos θ。要让高度跌破 0.25 m 需要 θ > 53.5°。
    #     所以阈值取 60° 的话，这条终止【永远不会触发】—— 高度判据总是先到。
    #     取 50° 才能让它成为真正生效的、语义正确的判据（"倾了"而不是"矮了"）。
    #   · 也不能取 30°：Poppy 横向欠驱动（踝关节没有侧摆自由度），
    #     恢复过程中出现 30°~45° 的摆动是正常的，卡太紧会打断"挣扎恢复"的学习，
    #     而 D3 的验收标准之一恰恰是"能抗住随机推力"。
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)}
    )
    # 高度过低 → 一定是倒了（正常站立骨盆 0.42 m，取 0.25 m 留足余量）
    low_base_height = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.25}
    )

    # 关于"关节越限终止"（刻意不加，理由写清楚）：
    #   1) PhysX 在求解器层面就是硬约束，关节物理上不可能越过 URDF 限位，
    #      所以这一项本身抓不到任何东西；
    #   2) "策略顶在限位上偷懒"这个真实风险由奖励项 dof_pos_limits 处理；
    #   3) Isaac Lab v2.1.0 的 mdp.joint_pos_out_of_limit 有索引 bug：
    #         out_of_upper_limits = torch.any(..., dim=1)   # 先降成 1 维
    #         return ... out_of_upper_limits[:, asset_cfg.joint_ids]   # 再按 2 维索引 → IndexError
    #      官方配置里没有任何任务用它（只有 cartpole 用了带显式边界的
    #      joint_pos_out_of_manual_limit），所以这个分支没有被上游测到。
    #      直接用会在第一次 termination_manager.compute() 时崩掉。
    #   4) 摔倒已经由 illegal_contact / bad_orientation / low_base_height 三重兜住。


##
# 事件（domain randomization 的入口，D6 会大幅扩展）
##


@configclass
class EventCfg:
    """开局与重置的随机化。D3 只用"轻度"版本，目的是让策略不要依赖精确初始条件。"""

    # 地面摩擦随机化：D3 用固定在 1.0 的窄区间，D6 再打开
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.0),
            "dynamic_friction_range": (0.6, 0.9),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # 质量随机化：URDF 给出的总质量 2.607 kg 明显小于真机的 3.5 kg
    # （真机含电池/树莓派等）。这里不加固定补偿，而是把它作为已知建模差异记录下来，
    # 用域随机化覆盖。±0.1 kg 约相当于 ±4%。
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=BASE_BODY),
            "mass_distribution_params": (-0.1, 0.1),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    # ---- reset ----
    # 注意：reset_root_state_uniform 是在【默认根状态】上加偏移，
    # 所以 init_state.pos.z 必须已经等于站立高度，否则每次 reset 都会下落一下。
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.3, 0.3)},
            "velocity_range": {
                "x": (-0.15, 0.15),
                "y": (-0.15, 0.15),
                "z": (-0.05, 0.05),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    # 关节角度偏移：从默认姿态附近随机出发（±0.05 rad ≈ 3°）
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )

    # ---- 随机推力 ----
    # D3 的验收标准之一是"能抗住随机推力"，所以推力从一开始就要有。
    # 推力直接改基座速度（等价于瞬间冲量），对 2.6 kg 的小机器人，
    # 0.3 m/s 的横向速度突增已经相当可观。
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
    )


##
# 环境
##


@configclass
class PoppyStandingEnvCfg(ManagerBasedRLEnvCfg):
    """站立环境主配置。"""

    scene: PoppySceneCfg = PoppySceneCfg(num_envs=2048, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """后处理：把机器人放进场景，并设定仿真/控制频率。"""
        self.scene.robot = POPPY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # ---- 控制频率 ----
        # 物理步 5 ms（200 Hz），每 4 个物理步做一次控制决策 → 控制频率 50 Hz。
        # 为什么是 50 Hz：
        #   · 舵机总线（Dynamixel TTL）实际可达 ~50~100 Hz，50 Hz 与真机匹配
        #   · 行走任务的典型控制频率就在 30~100 Hz，太高会让动作变化率惩罚失效
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        # Poppy 只有 2.6 kg、11 个刚体，接触对很少，碰撞堆可以开小一点省显存
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 2**15

        # 接触传感器按物理步更新：抬脚/落脚的空中时间需要物理步精度
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt


@configclass
class PoppyStandingEnvCfg_PLAY(PoppyStandingEnvCfg):
    """评估/演示用：环境少、关观测噪声、关随机推力。"""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 32
        self.scene.env_spacing = 2.0
        self.observations.policy.enable_corruption = False
        # 评估时不要随机推力，否则测的是"恢复能力"而不是"保持能力"
        self.events.push_robot = None
