#!/usr/bin/env python3
"""D3 · rest pose 静力学判据（修正版）+ PD 增益标定。

================================================================================
这个脚本在回答什么问题
================================================================================
D3 要把「Poppy 站在地上」这件事做成 RL 任务，那么在写环境之前必须先定死三件事：

  Q1  站立时各关节的 rest pose（默认关节角）取多少？
  Q2  站立时 pelvis 高度取多少？（决定 env 里 reset_root_state 的采样中心）
  Q3  关节 PD 增益 (kp, kd) 取多少？

Q2 和 Q3 都极易出错，而且错了以后表现为「训练不收敛」，很难反查。
所以先用一个**纯静力学、不含控制**的实验把它们定下来。

================================================================================
上一版脚本错在哪（这是我重写它的原因，错误本身比结论更值得记）
================================================================================
上一版把脚底几何读错了轴序：
  · 实测 STL 的 AABB 是 x=[-0.0464,+0.0070] y=[-0.0350,+0.0098] z=[-0.0441,+0.1025]
  · 脚本凭「z 跨度最小」就假设 z 竖直，于是把 z=-0.0441 当成脚底
    —— 其实那 0.0441 是脚在【前后方向】的一半长度，脚底是 y=-0.0350
  · 后果 1：支撑多边形在前后方向只有 0.9 mm 厚（这个数字一眼就该报警）
  · 后果 2：站立高度算成 0.386+0.0441=0.430 m，而正确值是 0.386+0.0350=0.4210 m
            —— 基座被抬高 9 mm，脚其实悬空，导致「有接触」的假设整体失效

教训（写进方法论）：
  STL 没有轴标签。「哪一维竖直」不能靠猜，要靠**物理量级**判断：
  脚厚只有几厘米，而脚长十几厘米；再拿 link 的世界姿态把候选面变到世界系，
  看哪个面水平 —— 水平且最低的那个才是脚底。

================================================================================
本脚本的三层结构
================================================================================
  Part 1  站立高度自动标定：基座锁住、脚悬空起步、迭代下移直到脚底刚好触地
  Part 2  静力学判据：质心水平投影是否落在支撑多边形（双脚脚底凸包）内
  Part 3  PD 增益标定：逐关节看「误差 / 需要多少力矩 / 有没有撞力限」
          → 反推出 kp = 所需力矩 / 允许误差，而不是拍脑袋

帧约定（实测，全脚本统一）：
  base/pelvis 系 : +x 左, -y 前, +z 上
  foot link 系   : +x 左, +y 上, +z 前   （link 世界姿态 = 绕 X 转 +90°）
  脚底平面       : foot link 局部 y = -0.0350
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Isaac Sim 启动必须在其它 omni 导入之前
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="D3 rest pose 静力学 + PD 标定")
parser.add_argument("--drop-seconds", type=float, default=2.5, help="落体观察时长")
parser.add_argument("--out", type=str, default="static_pd_probe.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

PHYS_DT = 0.005
DECIMATION = 4
DEVICE = args_cli.device or "cuda:0"

EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0
#: kp 应当与执行器能力成正比 —— 所以 hip（MX-64）与其它（MX-28）的 kp 比例
#: 直接取力矩比 6.0/2.5 = 2.4。这比「审计报告说 2.5 倍」更有物理依据。
KG_RATIO = EFFORT_MX64 / EFFORT_MX28            # = 2.4

LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]

#: 脚底接触多边形，foot link 局部系。由 scripts/d3_foot_geometry.py 从 STL 实测。
#: sole_y 是脚底平面高度；poly_xz 是脚底平面在 (x=左右, z=前后) 上的凸包顶点。
FOOT_SOLE = {
    "l_foot": {
        "sole_y": -0.035000,
        "poly_xz": [
            (-0.042199, 0.068111),
            (-0.041815, -0.025743),
            (-0.003618, -0.019865),
            (0.003822, 0.058904),
            (-0.002621, 0.088877),
        ],
    },
    "r_foot": {
        "sole_y": -0.035000,
        "poly_xz": [
            (-0.003630, 0.060648),
            (0.003274, -0.018990),
            (0.041815, -0.025743),
            (0.042554, 0.065428),
        ],
    },
}

#: 直立时脚 link 相对 pelvis 的高度（d3_pose_ident.py 运动学实测）
FOOT_LINK_Z_REL_PELVIS = -0.3860
SOLE_OFFSET = 0.0350
STANDING_PELVIS_Z_ANALYTIC = -FOOT_LINK_Z_REL_PELVIS + SOLE_OFFSET      # ≈ 0.4210 m

#: 阻尼系数设计：kd = 2·ζ·sqrt(kp·J_eff)
DAMP_ZETA = 1.0
J_EFF_EST = 0.02          # kg·m^2，腿部反射惯量量级估计（髋-膝-踝串联的等效值）


def kd_for(kp: float) -> float:
    """临界阻尼近似：kd = 2·ζ·sqrt(kp·J_eff)。

    为什么不能简单地 kd ∝ kp：
      PD 闭环的固有频率 ω = sqrt(kp/J)，临界阻尼要求 kd = 2·J·ω = 2·sqrt(kp·J)。
      即 kd 应随 sqrt(kp) 增长，而不是线性。按线性给 kp 增大时的 kd 会过阻尼
      （响应变迟钝），按固定值给则会欠阻尼（高频抖动，PhysX 里会炸）。
    """
    return 2.0 * DAMP_ZETA * math.sqrt(max(kp, 1e-6) * J_EFF_EST)


def build_pose(k: float) -> dict[str, float]:
    """由屈膝量 k 生成 rest pose。

    几何依据（d3_pose_ident.py 实测）：
      · 三根俯仰轴 hip_y / knee_y / ankle_y 都绕 base 的 X 轴 → 脚掌保持水平需
        髋 + 膝 + 踝 = 0
      · 膝盖只能单方向弯（限位 [-2.339, +0.061]），所以屈膝取负值
      · 脚的落点由髋的角度主导（实测髋:膝 ≈ 2:1 的水平位移比）→ 髋补 k/2
      · 镜像约定 r = -l（5 对关节都验证过对称）
    """
    pose = {name: 0.0 for name in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * (k / 2.0)
        pose[f"{side}_ankle_y"] = s * (k / 2.0)
    return pose


def build_poppy_cfg(kp28: float, kp_hip: float, kd28: float, kd_hip: float) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="/World/Poppy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=POPPY_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_linear_velocity=100.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                fix_root_link=False,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, STANDING_PELVIS_Z_ANALYTIC),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"],
                effort_limit=EFFORT_MX64, effort_limit_sim=EFFORT_MX64,
                velocity_limit_sim=8.2, stiffness=kp_hip, damping=kd_hip,
            ),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
                effort_limit=EFFORT_MX28, effort_limit_sim=EFFORT_MX28,
                velocity_limit_sim=7.0, stiffness=kp28, damping=kd28,
            ),
        },
    )


# ---------------------------------------------------------------------------
# 几何工具
# ---------------------------------------------------------------------------

def quat_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def foot_sole_points_world(body_name: str, pos, quat):
    """把脚底凸包顶点从 foot link 局部系变到世界系。"""
    d = FOOT_SOLE[body_name]
    R = quat_to_matrix(quat)
    out = []
    for xl, zl in d["poly_xz"]:
        v = (xl, d["sole_y"], zl)
        out.append(tuple(float(pos[k]) + sum(R[k][j] * v[j] for j in range(3)) for k in range(3)))
    return out


def convex_hull(points):
    """Andrew monotone chain，返回逆时针凸包。"""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def signed_distance_to_hull(poly, p):
    """点到凸包（逆时针）的带符号最小距离。>0 在内部。

    支撑多边形的判据用「凸包」而不是「并集」：
      刚体在若干个单边接触点上，倾覆只可能绕接触点凸包的边发生；
      两脚之间的空隙不是支撑面，但它也不构成倾覆边（凸包把它盖住了）。
      所以静平衡条件 = 质心水平投影落在接触点凸包内。
    """
    n = len(poly)
    if n < 3:
        return -float("inf")
    best = float("inf")
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ex, ey)
        if ln < 1e-12:
            continue
        cr = (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / ln
        best = min(best, cr)
    return best


def hull_edge_distances(poly, p):
    """返回 [(边序号, 到该边的带符号距离)]，用于找出「最接近倾覆的那条边」。"""
    n = len(poly)
    out = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ex, ey = b[0] - a[0], b[1] - a[1]
        ln = math.hypot(ex, ey)
        if ln < 1e-12:
            continue
        out.append((i, (ex * (p[1] - a[1]) - ey * (p[0] - a[0])) / ln))
    return out


# ---------------------------------------------------------------------------
# 仿真原语
# ---------------------------------------------------------------------------

def pose_to_tensor(robot, pose):
    vals = torch.zeros(len(robot.joint_names), device=DEVICE)
    for i, name in enumerate(robot.joint_names):
        vals[i] = float(pose.get(name, 0.0))
    return vals.unsqueeze(0).repeat(robot.num_instances, 1)


def reset_robot(robot, pose_t, base_z):
    n = robot.num_instances
    rp = torch.zeros(n, 7, device=DEVICE)
    rp[:, 3] = 1.0
    rp[:, 2] = base_z
    robot.write_root_pose_to_sim(rp)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))
    robot.write_joint_state_to_sim(pose_t, torch.zeros_like(pose_t))


def set_gains(robot, kp28, kp_hip):
    stiff = torch.zeros(robot.num_instances, len(robot.joint_names), device=DEVICE)
    damp = torch.zeros_like(stiff)
    for i, jn in enumerate(robot.joint_names):
        if jn.endswith("hip_y"):
            stiff[:, i], damp[:, i] = kp_hip, kd_for(kp_hip)
        else:
            stiff[:, i], damp[:, i] = kp28, kd_for(kp28)
    robot.write_joint_stiffness_to_sim(stiff)
    robot.write_joint_damping_to_sim(damp)
    return stiff, damp


def lowest_sole_z(robot):
    """两只脚所有脚底顶点的最低世界 z。"""
    body_names = list(robot.body_names)
    zs = []
    for name in ("l_foot", "r_foot"):
        i = body_names.index(name)
        for p in foot_sole_points_world(name, robot.data.body_pos_w[0][i],
                                        robot.data.body_quat_w[0][i]):
            zs.append(p[2])
    return min(zs)


def hold_pose(robot, sim, pose, base_z, steps, reset_base=True):
    """锁住基座、放开关节，跑 steps 步。reset_base=True 时每步重置基座位姿。"""
    pose_t = pose_to_tensor(robot, pose)
    for _ in range(steps):
        if reset_base:
            reset_robot(robot, pose_t, base_z=base_z)
        robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)
    return pose_t


def calibrate_standing_height(robot, sim, pose, iters=5, steps=90, verbose=False):
    """迭代求「脚底刚好触地」的 pelvis 高度。

    为什么用迭代而不是解析式：
      解析式需要知道完整的运动学链长，而屈膝姿态会让腿"变短"，手算容易错。
      迭代法只需要一个可测的量（脚底最低点离地多少），从高处往下逼近，
      保证脚永远是悬空（不穿地），因此不会触发接触力干扰测量。收敛性很好：
      每步误差 ∝ 当前偏差，实测 3~4 次就到 1e-5 m。
    """
    set_gains(robot, 200.0, 200.0 * KG_RATIO)     # 标定阶段用很硬的 PD，减小关节误差影响
    z = STANDING_PELVIS_Z_ANALYTIC + 0.030        # 从高处起步 → 脚一定悬空
    trace = []
    for _ in range(iters):
        hold_pose(robot, sim, pose, z, steps)
        dz = lowest_sole_z(robot)
        trace.append((z, dz))
        z -= dz
        if abs(dz) < 1e-5:
            break
    if verbose:
        print("      标定迭代: " + "  ".join(f"z={a:.5f}(Δ{b:+.5f})" for a, b in trace))
    return z, trace


def static_state(robot, sim, pose, base_z, settle_steps=250):
    """锁基座 + 重力加载 → 测静态量。返回逐关节诊断 + 质心 + 支撑多边形。"""
    pose_t = hold_pose(robot, sim, pose, base_z, settle_steps)

    body_names = list(robot.body_names)
    masses = robot.data.default_mass[0].to(DEVICE)
    com_pos = getattr(robot.data, "body_com_pos_w", None)
    if com_pos is None:
        com_pos = robot.data.body_pos_w
    com_w = (com_pos[0] * masses.unsqueeze(1)).sum(dim=0) / masses.sum()
    root = robot.data.root_pos_w[0]

    pts = []
    for name in ("l_foot", "r_foot"):
        i = body_names.index(name)
        pts += foot_sole_points_world(name, robot.data.body_pos_w[0][i],
                                      robot.data.body_quat_w[0][i])
    poly = convex_hull([(p[0], p[1]) for p in pts])
    com_xy = (float(com_w[0]), float(com_w[1]))
    margin = signed_distance_to_hull(poly, com_xy)
    edges = hull_edge_distances(poly, com_xy)
    worst_edge = min(edges, key=lambda t: t[1])[0] if edges else -1

    q = robot.data.joint_pos[0]
    qd = robot.data.joint_vel[0]
    err = q - pose_t[0]
    applied = getattr(robot.data, "applied_torque", None)
    computed = getattr(robot.data, "computed_torque", None)

    per_joint = {}
    for i, jn in enumerate(robot.joint_names):
        lim = EFFORT_MX64 if jn.endswith("hip_y") else EFFORT_MX28
        a_tq = float(applied[0][i]) if applied is not None else float("nan")
        c_tq = float(computed[0][i]) if computed is not None else float("nan")
        per_joint[jn] = {
            "target": float(pose_t[0][i]),
            "actual": float(q[i]),
            "err": float(err[i]),
            "err_deg": math.degrees(float(err[i])),
            "vel": float(qd[i]),
            "applied_torque": a_tq,
            "computed_torque": c_tq,
            "effort_limit": lim,
            "saturation_pct": (abs(a_tq) / lim * 100.0) if lim > 0 else float("nan"),
        }

    sole_min_z = min(p[2] for p in pts)
    h_com = float(com_w[2]) - sole_min_z
    return {
        "pelvis_z": float(root[2]),
        "com_world": [float(v) for v in com_w],
        "com_xy": list(com_xy),
        "support_polygon_xy": [[float(a), float(b)] for a, b in poly],
        "support_x_range": [min(p[0] for p in poly), max(p[0] for p in poly)],
        "support_y_range": [min(p[1] for p in poly), max(p[1] for p in poly)],
        "com_inside_support": bool(margin > 0),
        "com_margin_m": float(margin),
        "worst_edge_index": int(worst_edge),
        "h_com_above_ground": h_com,
        "tau_inverted_pendulum_s": math.sqrt(h_com / 9.81) if h_com > 0 else 0.0,
        "sole_min_z": sole_min_z,
        "total_mass": float(masses.sum()),
        "joint_err_max": float(err.abs().max()),
        "joint_err_rms": float((err ** 2).mean().sqrt()),
        "joints": per_joint,
    }


def dynamic_analysis(robot, sim, pose, base_z, seconds):
    """放开基座，看倒立摆失稳要多长时间（只作对照，不作判据）。"""
    pose_t = pose_to_tensor(robot, pose)
    reset_robot(robot, pose_t, base_z)
    robot.set_joint_position_target(pose_t)
    steps = int(seconds / PHYS_DT)
    series = []
    for step in range(steps):
        if step % DECIMATION == 0:
            robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)
        if step % 10 == 0:
            rq = robot.data.root_quat_w[0]
            # 机身 z 轴与重力反方向的夹角 = 倾角
            w, x, y, z = [float(v) for v in rq]
            up_z = 1 - 2 * (x * x + y * y)
            tilt = math.degrees(math.acos(min(1.0, max(-1.0, up_z))))
            series.append({
                "t": step * PHYS_DT,
                "tilt": tilt,
                "pelvis_z": float(robot.data.root_pos_w[0, 2]),
                "joint_err_max": float((robot.data.joint_pos[0] - pose_t[0]).abs().max()),
            })
    topple = next((r["t"] for r in series if r["tilt"] > 15.0), None)
    return {"series": series, "topple_time": topple}


def finish() -> None:
    """硬退出 —— simulation_app.close() 在本环境下会挂住，累积僵尸进程。"""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE,
                                                    gravity=(0.0, 0.0, -9.81)))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    kp0, kp0_hip = 12.5, 12.5 * KG_RATIO
    robot = Articulation(build_poppy_cfg(kp0, kp0_hip, kd_for(kp0), kd_for(kp0_hip)))
    sim.reset()

    total_mass = float(robot.data.default_mass.sum())
    print("=" * 104)
    print("D3 · rest pose 静力学判据（修正版）+ PD 增益标定")
    print("=" * 104)
    print(f"关节数 {len(robot.joint_names)}，刚体数 {len(robot.body_names)}，总质量 {total_mass:.3f} kg")
    print(f"脚底平面在 foot link 局部 y = {FOOT_SOLE['l_foot']['sole_y']:+.4f} m")
    print(f"解析站立高度 = 0.3860 + 0.0350 = {STANDING_PELVIS_Z_ANALYTIC:.4f} m")
    print(f"kp 初始值：MX-28 组 {kp0:.1f}，MX-64(hip_y) {kp0_hip:.1f}"
          f"（比例 {KG_RATIO:.2f} = 力矩比 6.0/2.5）")
    print(f"kd 设计：kd = 2·ζ·sqrt(kp·J_eff)，ζ={DAMP_ZETA}, J_eff={J_EFF_EST} kg·m^2")
    print(f"  → kd(12.5) = {kd_for(12.5):.3f}，kd(60) = {kd_for(60):.3f}（注意是 sqrt 而不是线性）")
    print()
    print("判据说明：直立是【不稳定】的倒立摆平衡，PD 只能稳关节角、不能稳平衡。")
    print("          所以「这个姿态能不能站」必须用静力学判据（质心投影落在支撑多边形内），")
    print("          而不是「放开后能不能自己站住」—— 后者任何姿态都会倒（那正是 RL 要学的）。")

    results = []

    # ================= Part 1 · 站立高度标定 =================
    print("\n" + "-" * 104)
    print("Part 1 · 站立高度自动标定（锁基座 → 量脚底离地 → 下移 → 迭代）")
    print("-" * 104)
    print(f"{'屈膝 k':>7}{'标定次数':>10}{'pelvis z (m)':>15}{'解析预测':>12}{'差值 (mm)':>12}"
          f"{'脚底 z (m)':>13}  说明")
    print("-" * 104)
    heights = {}
    for k in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45, 0.60]:
        pose = build_pose(k)
        z, trace = calibrate_standing_height(robot, sim, pose)
        heights[k] = z
        note = "零位=腿完全伸直" if k == 0.0 else f"腿缩短 {(0.4210 - z) * 1000:+.1f} mm"
        print(f"{k:>7.2f}{len(trace):>10}{z:>15.5f}{STANDING_PELVIS_Z_ANALYTIC:>12.5f}"
              f"{(z - STANDING_PELVIS_Z_ANALYTIC) * 1000:>+12.2f}{trace[-1][1]:>13.6f}  {note}")
        results.append({"k": k, "standing_pelvis_z": z, "calib_trace": trace})

    # ================= Part 2 · 静力学判据 =================
    print("\n" + "-" * 104)
    print("Part 2 · 静力学判据：质心水平投影 vs 支撑多边形（双脚脚底凸包）")
    print("-" * 104)
    print(f"{'屈膝 k':>7}{'CoM x':>10}{'CoM y':>10}{'CoM z':>10}{'h_com':>9}"
          f"{'τ 倒立摆':>10}{'支撑内?':>9}{'边界余量(m)':>13}{'x 范围':>18}{'y 范围':>18}")
    print("-" * 104)
    for rec in results:
        k = rec["k"]
        set_gains(robot, kp0, kp0_hip)
        st = static_state(robot, sim, build_pose(k), rec["standing_pelvis_z"], settle_steps=300)
        rec["static"] = st
        cx, cy = st["com_xy"]
        xr, yr = st["support_x_range"], st["support_y_range"]
        print(f"{k:>7.2f}{cx:>10.4f}{cy:>10.4f}{st['com_world'][2]:>10.4f}"
              f"{st['h_com_above_ground']:>9.4f}{st['tau_inverted_pendulum_s']:>9.3f}s"
              f"{('是' if st['com_inside_support'] else '★否★'):>9}"
              f"{st['com_margin_m'] * 1000:>12.1f}mm"
              f"{'[%+.4f,%+.4f]' % (xr[0], xr[1]):>18}"
              f"{'[%+.4f,%+.4f]' % (yr[0], yr[1]):>18}")

    r0 = next(r for r in results if r["k"] == 0.0)
    poly0 = r0["static"]["support_polygon_xy"]
    print("\n  零位姿态的支撑多边形（世界 xy，单位 m，顶序为逆时针）：")
    print("    " + " → ".join(f"({x:+.4f},{y:+.4f})" for x, y in poly0))
    print(f"    支撑面 x 跨度 {r0['static']['support_x_range'][1] - r0['static']['support_x_range'][0]:.4f} m，"
          f"y（前后）跨度 {r0['static']['support_y_range'][1] - r0['static']['support_y_range'][0]:.4f} m")
    print("    上一版脚本把这一维算成 0.9 mm，是因为把 STL 的 z 当成了竖直方向。")
    print(f"    质心投影 ({r0['static']['com_xy'][0]:+.4f},{r0['static']['com_xy'][1]:+.4f})，"
          f"到最近倾覆边的距离 {r0['static']['com_margin_m'] * 1000:+.1f} mm（>0 即在支撑内）")

    # ================= Part 2b · 到倒计时（对照） =================
    print("\n" + "-" * 104)
    print("Part 2b · 动力学对照：从标定站立高度放开，看到倒用多久（不作为判据）")
    print("-" * 104)
    print(f"{'屈膝 k':>7}{'释放高度':>11}{'Tilt@0.1s':>12}{'Tilt@0.3s':>12}{'Tilt@0.5s':>12}"
          f"{'z@0.3s':>10}{'到倒用时':>11}{'前0.5s关节误差':>16}")
    print("-" * 104)
    for rec in [r for r in results if r["k"] in (0.0, 0.10, 0.20, 0.30)]:
        k = rec["k"]
        set_gains(robot, kp0, kp0_hip)
        h = rec["standing_pelvis_z"] + 0.004
        dy = dynamic_analysis(robot, sim, build_pose(k), h, args_cli.drop_seconds)

        def at(t, key):
            return next((r[key] for r in dy["series"] if r["t"] >= t), dy["series"][-1][key])

        early = [r["joint_err_max"] for r in dy["series"] if r["t"] <= 0.5]
        topple = f"{dy['topple_time']:.2f}s" if dy["topple_time"] is not None else "未倒(>2.5s)"
        print(f"{k:>7.2f}{h:>11.4f}{at(0.1, 'tilt'):>11.2f}°{at(0.3, 'tilt'):>11.2f}°"
              f"{at(0.5, 'tilt'):>11.2f}°{at(0.3, 'pelvis_z'):>10.4f}{topple:>11}"
              f"{max(early) if early else 0:>15.4f} rad")
        rec["dynamic"] = {"topple_time": dy["topple_time"],
                          "tilt_at": {str(t): at(t, "tilt") for t in (0.1, 0.3, 0.5, 1.0)},
                          "series": dy["series"]}

    # ================= Part 3 · PD 增益标定 =================
    print("\n" + "-" * 104)
    print("Part 3 · PD 增益标定（基座锁住，重力通过 PD 加载到关节 → 稳态误差 = 刚度够不够）")
    print("-" * 104)
    k_test = 0.20
    z_test = heights[k_test]
    print(f"测试姿态：屈膝 k = {k_test:.2f} rad（{math.degrees(k_test):.1f}°），"
          f"pelvis z = {z_test:.5f} m")
    print(f"kp 比例固定为 {KG_RATIO:.2f}（= 力矩比，hip_y 用 MX-64）")
    print()
    print(f"{'kp(MX-28)':>10}{'kp(hip)':>9}{'kd(28)':>8}{'kd(hip)':>9}"
          f"{'误差max(rad)':>13}{'误差(°)':>9}{'误差rms':>9}{'最大饱和%':>11}"
          f"{'CoM余量(mm)':>13}  判定")
    print("-" * 104)
    pd_rows = []
    for kp28 in [6.0, 9.0, 12.5, 18.0, 26.0, 38.0, 55.0]:
        kp_hip = kp28 * KG_RATIO
        set_gains(robot, kp28, kp_hip)
        st = static_state(robot, sim, build_pose(k_test), z_test, settle_steps=300)
        sat_max = max(j["saturation_pct"] for j in st["joints"].values()
                      if j["saturation_pct"] == j["saturation_pct"])
        err = st["joint_err_max"]
        if err > 0.05:
            verdict = "刚度过软"
        elif err > 0.02:
            verdict = "偏软"
        elif err > 0.005:
            verdict = "合适"
        else:
            verdict = "可能过刚（注意高频抖动）"
        print(f"{kp28:>10.1f}{kp_hip:>9.1f}{kd_for(kp28):>8.3f}{kd_for(kp_hip):>9.3f}"
              f"{err:>13.5f}{math.degrees(err):>9.3f}{st['joint_err_rms']:>9.5f}"
              f"{sat_max:>10.1f}%{st['com_margin_m'] * 1000:>12.1f}   {verdict}")
        pd_rows.append({"kp28": kp28, "kp_hip_y": kp_hip, "kd28": kd_for(kp28),
                        "kd_hip_y": kd_for(kp_hip), "joint_err_max": err,
                        "joint_err_rms": st["joint_err_rms"], "max_saturation_pct": sat_max,
                        "com_margin_m": st["com_margin_m"], "joints": st["joints"]})

    # 逐关节明细（用最硬的那一次，此时力矩分布最接近"纯重力"）
    print("\n  逐关节明细（kp(MX-28)=55 那一组，看谁在承担重力、谁撞了力限）：")
    print(f"    {'关节':>10}{'目标(rad)':>12}{'实际(rad)':>12}{'误差(°)':>10}"
          f"{'实际力矩':>11}{'力限':>8}{'饱和%':>9}{'速度':>10}")
    for jn, j in pd_rows[-1]["joints"].items():
        print(f"    {jn:>10}{j['target']:>12.4f}{j['actual']:>12.4f}{j['err_deg']:>10.3f}"
              f"{j['applied_torque']:>11.4f}{j['effort_limit']:>8.2f}"
              f"{j['saturation_pct']:>8.1f}%{j['vel']:>10.4f}")

    # 由「所需力矩 / 允许误差」反推 kp
    print("\n  由实测力矩反推 kp（kp = 该关节所需力矩 / 允许误差）：")
    print(f"    {'关节':>10}{'所需力矩':>11}{'kp@0.02rad':>13}{'kp@0.05rad':>13}{'kp@0.10rad':>13}")
    rec_kp = {}
    for jn, j in pd_rows[-1]["joints"].items():
        tq = abs(j["applied_torque"])
        rec_kp[jn] = {"torque": tq, "kp_at_0.02": tq / 0.02, "kp_at_0.05": tq / 0.05,
                      "kp_at_0.10": tq / 0.10}
        print(f"    {jn:>10}{tq:>11.4f}{tq / 0.02:>13.1f}{tq / 0.05:>13.1f}{tq / 0.10:>13.1f}")
    print("\n  选取原则：kp 让「最常见的负载下误差 ≈ 0.02~0.05 rad」即可，")
    print("  同时不能让 kp 大到 kp×力限对应角度远小于指令分辨率（否则策略的小动作被吃掉）。")

    set_gains(robot, kp0, kp0_hip)

    out = os.path.join(OUT_DIR, args_cli.out)
    with open(out, "w") as fh:
        json.dump({
            "total_mass": total_mass,
            "foot_sole_local": FOOT_SOLE,
            "standing_pelvis_z_analytic": STANDING_PELVIS_Z_ANALYTIC,
            "effort_limits": {"MX28": EFFORT_MX28, "MX64": EFFORT_MX64},
            "kd_design": {"zeta": DAMP_ZETA, "J_eff": J_EFF_EST},
            "results": results,
            "pd_sweep": pd_rows,
            "kp_from_torque": rec_kp,
        }, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {out}")
    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
