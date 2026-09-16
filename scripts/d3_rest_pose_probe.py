#!/usr/bin/env python3
"""D3 前置实验 ②：rest pose 标定 —— 让物理当裁判，不靠纸上推算

为什么要做这件事
----------------
D2 修正后的资产在 Isaac Sim 里能不能站住，取决于三件事：
  1. **关节零位是否对应"直立姿态"**。Dynamixel 舵机的零位是机械装配决定的，
     跟"人体自然直立"没有任何约定关系。URDF 里全部关节角 = 0 时，机器人可能是
     直腿、也可能是某个歪斜姿态。
  2. **PD 增益量级**。D3 探针实验①已定量证明 ImplicitActuatorCfg.stiffness 的
     单位就是 N·m/rad（换算系数 c ≈ 0.95~1.01），所以 D2 按重力力矩估的
     hip_y kp≈20 / 其余 kp≈8 可以直接填。
  3. **姿态本身是否在支撑多边形内**。Poppy 脚掌小、无踝侧摆，对姿态很敏感。

方法（本脚本的核心思想）
------------------------
不去纸上推算零位，而是**把候选姿态用 PD 硬性保持住、放开基座、加重力**，
让物理来当裁判：谁能站住，谁就是物理上自洽的直立姿态。

  · 姿态参数化：屈膝量 k + 髋俯仰补偿 c + 踝俯仰补偿 d
  · 左右镜像：r_joint = -l_joint（由关节限位表推断，见下）
  · 候选网格：k ∈ {0.0, 0.15, 0.30, 0.45} × c ∈ {-0.5, 0, +0.5} × d ∈ {-0.5, 0, +0.5} = 36 个
  · 判据：测量窗口内的存活率 / 基座倾角 / 基座高度 / 水平漂移

关于"左右镜像"这个关键推理
--------------------------
关节限位表（assets/poppy/urdf/Poppy_Humanoid_locked_joints.tsv）：

    r_knee_y  [-2.339, +0.061]     l_knee_y  [-0.061, +2.339]
    r_hip_y   [-1.484, +1.833]     l_hip_y   [-1.815, +1.466]
    r_hip_z   [-1.571, +0.436]     l_hip_z   [-0.436, +1.571]

限位区间左右**近似反对称**，说明左脚和右脚处于镜像坐标系里：
同一个物理动作（比如"屈膝"）在左右腿的**数值符号相反**。
所以「屈膝 k」应当写成  r_knee_y = -k,  l_knee_y = +k。

这也是当年"反关节"最容易被误判的地方 —— 如果两份代码一份用镜像约定、
一份用同一符号，看起来就是"一条腿正常、一条腿反着弯"。

膝盖弯曲方向由限位区间直接读出：右膝只有 -2.339 的行程、反向仅剩 0.061，
所以右膝"能弯"的方向是**负**方向（人类膝盖只能向后弯，符合）。

补偿量的物理依据（为什么是 k/2）
--------------------------------
记大腿俯仰角 t、膝盖角 = -k（右腿）。要让脚掌保持水平且落在髋正下方：

    脚掌相对地面角 = t + 膝角 + 踝角 = 0
    脚踝水平位置   = f(t, 膝角)  需要为 0

小角度下这一组约束的解正好是 t ≈ +k/2、踝 ≈ +k/2（髋前倾一半、踝再补回来）。
但**这依赖髋/膝/踝三根轴的"正方向"是否一致**，URDF 里没有任何保证 ——
所以 c、d 各扫 {-0.5, 0, +0.5} 三个值，让物理选。这就是"能推导的推导、不能推导的扫描"。

输出
----
    out/d3_rest_pose/sweep_result.json   36 个候选的量化排名 + 推荐姿态
    out/d3_rest_pose/render_*.png        最优姿态多视角渲染（--mode render）

用法
----
    python scripts/d3_rest_pose_probe.py --mode sweep
    python scripts/d3_rest_pose_probe.py --mode both      # 扫描 + 渲染
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import zlib

# ---------------------------------------------------------------------------
# 1. 路径与常量
# ---------------------------------------------------------------------------

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

#: 腿部 10 个训练关节（上身 15 个关节已在 D2 的 URDF 里锁成固定关节）
LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]

#: 关节限位（弧度），来自 D3 探针实验的逐项核对，用于软限位与结果校验
JOINT_LIMITS = {
    "r_hip_x": (-0.49741884, 0.52359878),
    "r_hip_z": (-1.57079633, 0.43633231),
    "r_hip_y": (-1.48352986, 1.83259571),
    "r_knee_y": (-2.33874120, 0.06108652),
    "r_ankle_y": (-0.78539816, 0.78539816),
    "l_hip_x": (-0.52359878, 0.49741884),
    "l_hip_z": (-0.43633231, 1.57079633),
    "l_hip_y": (-1.81514242, 1.46607657),
    "l_knee_y": (-0.06108652, 2.33874120),
    "l_ankle_y": (-0.78539816, 0.78539816),
}

#: PD 增益起始值（单位 N·m/rad、N·m·s/rad），依据 D2 审计的重力力矩估算
#: hip_y 驱动整条腿的摆动 → 负载最大，配 MX-64（6 N·m），增益高
#: 其余关节配 MX-28（2.5 N·m），增益低
KP_HIP_Y, KD_HIP_Y = 20.0, 0.6
KP_OTHER, KD_OTHER = 8.0, 0.3

#: 力矩上限（N·m）：有意低于 URDF 的 3.1 / 7.3，因为那是舵机的峰值指标
EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0

#: 控制频率 50 Hz（项目约定：以后迁移真机时 MX-28 的通信周期也是这个量级）
CTRL_DT = 0.02
PHYS_DT = 0.005
DECIMATION = int(round(CTRL_DT / PHYS_DT))  # 4

#: 单次试验时长（秒）
CALIB_SECONDS = 2.5    # 标定参考高度用
SETTLE_SECONDS = 1.5   # 每个候选的落地+稳定时间
MEASURE_SECONDS = 1.0  # 只测量窗口内的数据

#: 存活判据
ALIVE_TILT_DEG = 45.0
ALIVE_HEIGHT_RATIO = 0.55  # 基座高度低于参考高度的 55% 视为倒了

#: 候选网格
K_GRID = [0.0, 0.15, 0.30, 0.45]
C_GRID = [-1.0, 0.0, 1.0]   # 髋俯仰补偿系数（乘以 k/2）；理论值 +1
D_GRID = [-1.0, 0.0, 1.0]   # 踝俯仰补偿系数（乘以 k/2）；理论值 -1


# ---------------------------------------------------------------------------
# 2. 启动 Isaac Sim（必须在导入 isaaclab 之前完成 AppLauncher）
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Poppy rest pose 标定（物理当裁判）")
parser.add_argument("--mode", type=str, default="sweep", choices=["sweep", "render", "both"],
                    help="sweep = 数值扫描；render = 渲染最优姿态；both = 先扫描再渲染")
parser.add_argument("--pose_json", type=str, default=None,
                    help="render 模式下指定要渲染的关节角 JSON（默认用扫描结果里的推荐姿态）")
# 注意：--headless / --device / --enable_cameras 由 AppLauncher.add_app_launcher_args 注册，
# 这里绝对不能再手写一遍，否则 argparse 会因为重复选项直接报错。

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.mode in ("render", "both"):
    args_cli.enable_cameras = True  # 渲染必须有相机管线

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 以下导入必须在 AppLauncher 之后 ----------------------------------------

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402  (v2.1.0 里叫这个名字，不是 quat_apply_inverse)

DEVICE = args_cli.device or "cuda:0"


# ---------------------------------------------------------------------------
# 3. 机器人配置
# ---------------------------------------------------------------------------

def build_poppy_cfg() -> ArticulationCfg:
    """构造 Poppy 的 ArticulationCfg。

    两组执行器分开配置的原因：Poppy 髋俯仰（hip_y）用 MX-64（6 N·m），
    其余腿部关节用 MX-28（2.5 N·m）。力矩上限差了 2.4 倍，
    如果统一按 MX-28 配，髋部会被人为削弱；统一按 MX-64 配，
    膝盖就有了不存在的能力 —— 两者都会让学到的策略在真机上失效。
    """
    return ArticulationCfg(
        prim_path="/World/Poppy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=POPPY_USD,
            activate_contact_sensors=False,
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
                enabled_self_collisions=False,          # Poppy 网格互穿多，自碰撞会炸求解器
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=False,                    # 浮动基座 —— 这才是行走任务
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.4),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"],
                effort_limit=EFFORT_MX64,
                effort_limit_sim=EFFORT_MX64,
                velocity_limit_sim=8.2,
                stiffness=KP_HIP_Y,
                damping=KD_HIP_Y,
            ),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
                effort_limit=EFFORT_MX28,
                effort_limit_sim=EFFORT_MX28,
                velocity_limit_sim=7.0,
                stiffness=KP_OTHER,
                damping=KD_OTHER,
            ),
        },
    )


# ---------------------------------------------------------------------------
# 4. 姿态参数化与候选生成
# ---------------------------------------------------------------------------

def build_pose(k: float, c: float, d: float) -> dict[str, float]:
    """由 (屈膝量 k, 髋补偿系数 c, 踝补偿系数 d) 生成一组关节角。

    【镜像约定：r_joint = -l_joint】
    这条约定现在是**经过实测验证**的（D3 运动学识别 + 轴向修正之后，
    5 个镜像关节对在 r=-l 下都给出完全镜像的形变）。
    官方 URDF 原本有个例外：l_knee_y 的 axis 漏了符号翻转，导致左膝
    只能朝"反着弯"的方向运动，对称屈膝在物理上不可能实现。
    已在 d2_patch_urdf.py 第 5 步修正（只改这一处）。

    k > 0 表示屈膝。
    c 是髋俯仰补偿系数：实测 r_hip_y=+0.3 → 右脚 -Y 0.106，r_knee_y=+0.3 → 右脚 -Y 0.052，
      两者比值 ≈ 2，所以抵消膝的位移需要髋补偿 ≈ k/2，即 c = +1。
    d 是踝俯仰补偿系数：要让脚掌保持水平（三根轴都是绕 X 的俯仰），
      需 髋 + 膝 + 踝 = 0 → 踝 = -(髋 + 膝) = -(k/2 - k) = +k/2，即 d = -1。
    这两个是理论推导值，网格里用 {0, +1, -1} 扫一遍验证。
    """
    pose = {name: 0.0 for name in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k                # 屈膝（右膝取负 = 正常弯曲方向）
        pose[f"{side}_hip_y"] = -s * c * k / 2.0      # 髋反向补偿（保持脚在髋下方）
        pose[f"{side}_ankle_y"] = s * d * k / 2.0     # 踝补偿（保持脚掌水平）
    return pose


def pose_to_tensor(robot: Articulation, pose: dict[str, float]) -> torch.Tensor:
    """按 robot.joint_names 的顺序把姿态字典转成张量（形状 (num_instances, num_joints)）。

    必须按名字映射而不是按顺序填 —— USD 里的关节顺序是导入器的输出顺序，
    跟 URDF 的书写顺序不一定一致，这是资产类 bug 的常见来源。
    """
    vals = torch.zeros(len(robot.joint_names), device=DEVICE)
    for i, name in enumerate(robot.joint_names):
        if name in pose:
            vals[i] = float(pose[name])
    return vals.unsqueeze(0).repeat(robot.num_instances, 1)


def check_limits(pose: dict[str, float]) -> list[str]:
    """检查姿态是否越限，返回越限说明列表（空列表 = 全部合规）。"""
    bad = []
    for name, val in pose.items():
        lo, hi = JOINT_LIMITS[name]
        if val < lo - 1e-6 or val > hi + 1e-6:
            bad.append(f"{name}={val:+.3f} 超出 [{lo:+.3f}, {hi:+.3f}]")
    return bad


# ---------------------------------------------------------------------------
# 5. 仿真工具函数
# ---------------------------------------------------------------------------

def reset_robot(robot: Articulation, pose_t: torch.Tensor, base_z: float) -> None:
    """把机器人复位到指定姿态 + 指定高度，速度清零。"""
    n = robot.num_instances
    root_pose = torch.zeros(n, 7, device=DEVICE)
    root_pose[:, 3] = 1.0          # 单位四元数 (w, x, y, z)
    root_pose[:, 2] = base_z
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))
    robot.write_joint_state_to_sim(pose_t, torch.zeros_like(pose_t))


def compute_tilt_deg(quat_w: torch.Tensor) -> torch.Tensor:
    """基座相对竖直方向的倾角（度）。

    做法：把世界重力方向 (0,0,-1) 旋转到基座坐标系，得到"投影重力"。
    基座竖直时投影重力 = (0,0,-1)，倾角 0；倾倒 90° 时投影重力水平。
    这与 Isaac Lab 观测里的 projected_gravity 是同一个量，训练时直接复用。
    """
    grav = torch.tensor([0.0, 0.0, -1.0], device=quat_w.device).repeat(quat_w.shape[0], 1)
    projection: torch.Tensor = quat_rotate_inverse(quat_w, grav)
    return torch.rad2deg(torch.acos((-projection[:, 2]).clamp(-1.0, 1.0)))


def run_episode(robot: Articulation, sim: SimulationContext, pose_t: torch.Tensor,
                base_z: float, total_seconds: float, measure_seconds: float) -> dict:
    """复位后放开基座跑一段，返回测量窗口内的统计量。"""
    reset_robot(robot, pose_t, base_z)
    robot.set_joint_position_target(pose_t)

    total_steps = int(total_seconds / PHYS_DT)
    measure_steps = int(measure_seconds / PHYS_DT)
    start_measure = total_steps - measure_steps

    tilts, heights = [], []
    xy_start = None
    for step in range(total_steps):
        if step % DECIMATION == 0:
            robot.set_joint_position_target(pose_t)   # 50 Hz 更新目标
        sim.step()
        robot.update(PHYS_DT)
        if step >= start_measure:
            tilt = compute_tilt_deg(robot.data.root_quat_w)
            tilts.append(float(tilt[0]))
            heights.append(float(robot.data.root_pos_w[0, 2]))
            if xy_start is None:
                xy_start = robot.data.root_pos_w[0, :2].clone()

    xy_end = robot.data.root_pos_w[0, :2]
    drift = float(torch.norm(xy_end - xy_start))

    tilts_t = torch.tensor(tilts)
    heights_t = torch.tensor(heights)
    return {
        "tilt_mean_deg": float(tilts_t.mean()),
        "tilt_max_deg": float(tilts_t.max()),
        "height_mean": float(heights_t.mean()),
        "height_min": float(heights_t.min()),
        "drift_xy": drift,
        "_tilts": tilts,
        "_heights": heights,
    }


# ---------------------------------------------------------------------------
# 6. 阶段一：标定参考高度
# ---------------------------------------------------------------------------

def calibrate_reference_height(robot: Articulation, sim: SimulationContext) -> float:
    """用"直腿站立"标定参考高度。

    直腿姿态下整条腿几乎是一根刚性柱体 —— 落地后极稳定，
    因此它给出的基座高度是可靠的几何参考值。
    后面每个候选姿态都从这个高度 + 2cm 处释放，
    避免"从 1 米高砸下来"这种会把好姿态也判死的测试方式。
    """
    pose = build_pose(0.0, 0.0, 0.0)
    pose_t = pose_to_tensor(robot, pose)
    # 【重要修正】从 1.0 m 放开 → 落地速度 3.5 m/s，好姿态也会被砸倒。
    # 运动学识别实测：零位时脚在 pelvis 下方 0.386 m，所以直腿站立的 pelvis
    # 高度应该是 ~0.40 m。这里从 0.45 m 轻微落下（落差 5 cm，落地速度 1 m/s）。
    res = run_episode(robot, sim, pose_t, base_z=0.45,
                      total_seconds=CALIB_SECONDS, measure_seconds=0.5)
    h = res["height_mean"]
    print(f"[标定] 直腿姿态落地后基座高度 = {h:.4f} m，倾角 = {res['tilt_mean_deg']:.2f}°")
    if h < 0.15:
        raise RuntimeError(f"直腿姿态都站不住（基座高度仅 {h:.3f} m），资产或地面有问题，先回 D2 排查")
    return h


# ---------------------------------------------------------------------------
# 7. 阶段二：候选姿态扫描
# ---------------------------------------------------------------------------

def sweep(robot: Articulation, sim: SimulationContext, ref_height: float) -> list[dict]:
    candidates = []
    for k in K_GRID:
        for c in C_GRID:
            for d in D_GRID:
                candidates.append({"k": k, "c": c, "d": d})

    print(f"\n[扫描] 共 {len(candidates)} 个候选姿态，释放高度 = {ref_height + 0.02:.4f} m")
    print(f"{'#':>3} {'k':>5} {'c':>5} {'d':>5} | {'存活率':>7} {'倾角均值':>8} "
          f"{'倾角峰值':>8} {'高度比':>7} {'漂移m':>7}  判定")
    print("-" * 92)

    results = []
    for idx, cand in enumerate(candidates):
        pose = build_pose(cand["k"], cand["c"], cand["d"])
        violations = check_limits(pose)
        if violations:
            cand["limits_violated"] = violations
        pose_t = pose_to_tensor(robot, pose)

        res = run_episode(robot, sim, pose_t, base_z=ref_height + 0.02,
                          total_seconds=SETTLE_SECONDS + MEASURE_SECONDS,
                          measure_seconds=MEASURE_SECONDS)

        h_ratio = res["height_mean"] / ref_height
        alive_frac = float(np.mean([
            1.0 if (t < ALIVE_TILT_DEG and h > ALIVE_HEIGHT_RATIO * ref_height) else 0.0
            for t, h in zip(res["_tilts"], res["_heights"])
        ]))

        record = {
            **cand,
            "alive_frac": alive_frac,
            "tilt_mean_deg": res["tilt_mean_deg"],
            "tilt_max_deg": res["tilt_max_deg"],
            "height_mean": res["height_mean"],
            "height_ratio": h_ratio,
            "drift_xy": res["drift_xy"],
            "limits_violated": violations,
            "pose": pose,
        }
        results.append(record)

        verdict = "站住" if alive_frac > 0.95 else ("摇晃" if alive_frac > 0.3 else "倒了")
        print(f"{idx:>3} {cand['k']:>5.2f} {cand['c']:>+5.1f} {cand['d']:>+5.1f} | "
              f"{alive_frac * 100:>6.1f}% {res['tilt_mean_deg']:>7.2f}° "
              f"{res['tilt_max_deg']:>7.2f}° {h_ratio:>7.3f} {res['drift_xy']:>7.4f}  {verdict}")

    # 排序：先看存活率，再看倾角均值，再看漂移（越小越好）
    results.sort(key=lambda r: (-r["alive_frac"], r["tilt_mean_deg"], r["drift_xy"]))
    return results


# ---------------------------------------------------------------------------
# 8. 阶段三：渲染（可选，用于目视确认）
# ---------------------------------------------------------------------------

def _normalize(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-9 else list(v)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """求"从 pos 看向 target"的相机四元数（wxyz）。

    采用 convention="world"：相机前向 = +X，上向 = +Z。
    所以旋转矩阵的三列分别是相机在世界的 x/y/z 轴。
    """
    f = _normalize(_sub(target, pos))              # 前向 → 镜头 +X
    u = _normalize(up)
    z = _normalize(_sub(u, [f[0] * _dot(u, f), f[1] * _dot(u, f), f[2] * _dot(u, f)]))
    y = _cross(z, f)                                # 右手系
    m = np.array([[f[0], y[0], z[0]],
                  [f[1], y[1], z[1]],
                  [f[2], y[2], z[2]]], dtype=np.float64)

    t = m[0, 0] + m[1, 1] + m[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    return tuple(float(x) for x in q)


def write_png(path: str, arr: np.ndarray) -> None:
    """零依赖写 PNG（只用 zlib + struct）—— 避免为了存图再装 Pillow。"""
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 6))
    blob += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(blob)


def render_pose(robot: Articulation, sim: SimulationContext, pose: dict[str, float],
                base_z: float, tag: str) -> list[str]:
    """把某个姿态渲染成多视角 PNG，供人目视确认。"""
    from isaaclab.sensors import Camera, CameraCfg

    pose_t = pose_to_tensor(robot, pose)
    reset_robot(robot, pose_t, base_z)
    robot.set_joint_position_target(pose_t)
    for _ in range(int(1.2 / PHYS_DT)):      # 先跑 1.2 s 让它落到地面并稳定
        robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)

    target = (0.0, 0.0, base_z * 0.55)
    dist = 1.25
    views = {
        "front": (0.0, -dist, base_z * 0.75),      # 从 -Y 方向看（判断左右对称性）
        "side": (-dist, 0.0, base_z * 0.75),       # 从 -X 方向看（判断膝盖弯曲方向）
        "q34": (-dist * 0.72, -dist * 0.72, base_z * 1.15),
    }

    written = []
    for name, pos in views.items():
        cam_cfg = CameraCfg(
            prim_path=f"/World/Cam_{name}",
            update_period=0.0,
            height=720,
            width=960,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(pos=pos, rot=look_at_quat(pos, target), convention="world"),
        )
        cam = Camera(cam_cfg)
        cam.update(PHYS_DT)
        rgb = cam.data.output["rgb"]
        img = rgb[0, :, :, :3].to(torch.uint8).cpu().numpy()
        path = os.path.join(OUT_DIR, f"render_{tag}_{name}.png")
        write_png(path, img)
        written.append(path)
        print(f"[渲染] {path}  ({img.shape[1]}x{img.shape[0]})")
    return written


# ---------------------------------------------------------------------------
# 9. 主流程

def finish() -> None:
    """硬退出。

    【为什么不用 simulation_app.close()】
    Isaac Sim 4.5 在 headless 下 close() 经常**永远不返回** —— 实测多次：
    进程停在 R 状态，占着约 1.7 GB 内存 + 数 GB 显存不放，还阻塞了调用它的
    脚本链（"跑完 A 再跑 B" 的写法会因为 A 不退出而永远等下去）。
    对批处理实验脚本来说，结果已经写盘，**"直接硬退出"比"优雅关闭但可能不返回"可靠得多**。
    os._exit 会跳过程序级清理，但内核会回收进程的显存/内存，所以是安全的。
    """
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    sim = SimulationContext(sim_utils.SimulationCfg(
        dt=PHYS_DT,
        device=DEVICE,
        gravity=(0.0, 0.0, -9.81),
    ))

    # 地面 + 光照
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(build_poppy_cfg())
    sim.reset()

    print("=" * 92)
    print("Poppy rest pose 标定 —— 物理当裁判")
    print("=" * 92)
    print(f"关节数        : {robot.num_joints}")
    print(f"刚体数        : {robot.num_bodies}")
    print(f"关节名        : {robot.joint_names}")
    print(f"刚体名        : {robot.body_names}")
    print(f"控制频率      : {1.0 / CTRL_DT:.0f} Hz  (物理 dt = {PHYS_DT} s, decimation = {DECIMATION})")
    print(f"PD 增益       : hip_y kp={KP_HIP_Y}/kd={KD_HIP_Y}  其余 kp={KP_OTHER}/kd={KD_OTHER}  (N·m/rad)")
    print(f"力矩上限      : MX-28 {EFFORT_MX28} N·m / MX-64 {EFFORT_MX64} N·m")
    print(f"总质量        : {float(robot.data.default_mass.sum()):.3f} kg")
    print(f"资产          : {POPPY_USD}")

    written_images: list[str] = []

    if args_cli.mode in ("sweep", "both"):
        ref_height = calibrate_reference_height(robot, sim)
        results = sweep(robot, sim, ref_height)

        best = results[0]
        summary = {
            "reference_height": ref_height,
            "pd_gains": {"kp_hip_y": KP_HIP_Y, "kd_hip_y": KD_HIP_Y,
                         "kp_other": KP_OTHER, "kd_other": KD_OTHER},
            "effort_limits": {"mx28": EFFORT_MX28, "mx64": EFFORT_MX64},
            "best_pose": best,
            "ranking": results,
        }
        out_json = os.path.join(OUT_DIR, "sweep_result.json")
        with open(out_json, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        print("\n" + "=" * 92)
        print("扫描结论")
        print("=" * 92)
        print(f"参考高度          : {ref_height:.4f} m")
        print(f"最优姿态 (k,c,d)  : k={best['k']:.2f} rad  c={best['c']:+.1f}  d={best['d']:+.1f}")
        print(f"  · 存活率        : {best['alive_frac'] * 100:.1f}%")
        print(f"  · 倾角均值/峰值 : {best['tilt_mean_deg']:.2f}° / {best['tilt_max_deg']:.2f}°")
        print(f"  · 基座高度      : {best['height_mean']:.4f} m  (参考的 {best['height_ratio'] * 100:.1f}%)")
        print(f"  · 水平漂移      : {best['drift_xy'] * 100:.2f} cm")
        print(f"  · 关节角        : {json.dumps(best['pose'], indent=6)}")
        print(f"\n结果已写入        : {out_json}")

        standable = [r for r in results if r["alive_frac"] > 0.95 and r["tilt_mean_deg"] < 5.0]
        print(f"\n能稳定站住（存活率>95% 且倾角<5°）的候选共 {len(standable)} 个：")
        for r in standable[:8]:
            print(f"  k={r['k']:.2f} c={r['c']:+.1f} d={r['d']:+.1f} → "
                  f"倾角 {r['tilt_mean_deg']:.2f}° 漂移 {r['drift_xy'] * 100:.2f} cm")

        if args_cli.mode == "both":
            # 顺带渲染一个"全零姿态"作为对照 —— 这样能直接看出
            # "URDF 零位"到底是不是直立姿态，这是 D2 遗留的最后一个未知量
            zero_pose = build_pose(0.0, 0.0, 0.0)
            written_images += render_pose(robot, sim, zero_pose, ref_height + 0.01, "zeropose")
            written_images += render_pose(robot, sim, best["pose"], ref_height + 0.01, "best")

    if args_cli.mode == "render":
        ref_height = calibrate_reference_height(robot, sim)
        if args_cli.pose_json:
            with open(args_cli.pose_json) as fh:
                pose = json.load(fh)
        else:
            with open(os.path.join(OUT_DIR, "sweep_result.json")) as fh:
                pose = json.load(fh)["best_pose"]["pose"]
        written_images += render_pose(robot, sim, pose, ref_height + 0.01, "custom")

    if written_images:
        print("\n渲染图片：")
        for p in written_images:
            print(f"  {p}")

    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
