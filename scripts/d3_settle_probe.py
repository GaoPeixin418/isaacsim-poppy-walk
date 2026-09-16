#!/usr/bin/env python3
"""D3 前置实验 ⑥：rest pose 的静力学判据 + 动力学到倒计时

前面几版为什么都没得出结论
--------------------------
· 第 1 版从 z=1.0 m 放开 → 落地速度 3.5 m/s，好姿态也被砸倒（判据错）
· 第 2 版从 z=0.45 m 放开（已接近真实站立高度 0.430 m）→ 还是倒（86.7°）
· **真正的原因不是资产**：PD 只能稳住关节角，而"直立"本身是**不稳定的倒立摆平衡**。
  靠固定关节角被动站立，时间常数 τ = √(h_com/g) ≈ 0.17 s，
  任何 1e-4 rad 的数值扰动在 2.5 s 内会放大 e^(2.5/0.17) ≈ 2×10⁶ 倍 → 必倒。
  **站立本来就只能靠策略学，不能靠"姿态对"自动站住。**

所以正确做法是**把静力学与动力学分开**：

Part 1 · 静力学判据（决定性）
    准静态地把机器人摆成候选姿态，直接算：
      · 全身质心 CoM 投影到地面，落在**支撑多边形**内吗？
      · 支撑多边形由**脚掌碰撞网格**给出（不是脚 link 原点 —— 那只是脚踝位置）
      · 质心高度 h_com → 估算倒立摆时间常数 τ = √(h_com/g)
    这是"这个姿态能不能站"的**充分必要**的静态条件，与控制器无关。

Part 2 · 动力学到倒计时
    从真实站立高度释放，记录 tilt / 高度的时间序列，报"到倒下用时"。
    好的 rest pose 应该：早期（0.2 s 内）tilt 接近 0、高度等于站立高度，
    然后按 τ 的量级缓慢倾倒 —— 这个倾倒**正是 RL 要去解决的问题**。

脚掌几何（从 STL 实测，记录在案）
----------------------------------
    l_foot_respondable.stl   足迹 0.0534(x) × 0.0448(y)，脚底面在 link 原点下方 0.0441 m
    r_foot_respondable.STL   足迹 0.0534(x) × 0.0448(y)，脚底面在 link 原点下方 0.0450 m
    → 直立时 pelvis 高度 = 0.386（脚 link 相对 pelvis）+ 0.044 = **0.430 m**
    → 脚掌只有 5×4.5 cm —— 这就是 Poppy 横向不稳的物理根源

用法
----
    python scripts/d3_settle_probe.py --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Poppy rest pose 静力学判据 + 到倒计时")
parser.add_argument("--drop_seconds", type=float, default=2.0, help="动力学测试时长（s）")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

PHYS_DT = 0.005
DECIMATION = 4
DEVICE = args_cli.device or "cuda:0"

KP_HIP_Y, KD_HIP_Y = 20.0, 0.6
KP_OTHER, KD_OTHER = 8.0, 0.3
EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0

LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]

#: 脚掌碰撞网格在【foot link 局部系】里的包围盒（从 STL 实测，见文件头）
#: 坐标约定：局部 x 是左右方向、y 是前后方向、z 竖直；脚底面在 z = SOLE_Z
FOOT_AABB_LOCAL = {
    "l_foot": {"x": (-0.0464, 0.0070), "y": (-0.0350, 0.0098), "sole_z": -0.0441},
    "r_foot": {"x": (-0.0071, 0.0464), "y": (-0.0350, 0.0098), "sole_z": -0.0450},
}

#: 直立时脚 link 相对 pelvis 的高度（运动学识别实测）
FOOT_LINK_Z_REL_PELVIS = -0.3860
STANDING_PELVIS_Z = FOOT_LINK_Z_REL_PELVIS * -1.0 + 0.0441   # ≈ 0.430 m


def build_pose(k: float) -> dict[str, float]:
    """由屈膝量 k 生成 rest pose。

    镜像约定 r = -l（已由 5 对关节的镜像测试验证）。
    髋补偿 = +k/2、踝补偿 = +k/2，依据：
      · 实测 r_hip_y=+0.3 → 脚 -Y 0.106，r_knee_y=+0.3 → 脚 -Y 0.052，比值 ≈ 2 → 髋补 k/2
      · 三根俯仰轴都绕 X（脚掌旋转轴实测）→ 脚掌水平需 髋+膝+踝 = 0 → 踝 = -(髋+膝) = +k/2
    """
    pose = {name: 0.0 for name in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * (k / 2.0)
        pose[f"{side}_ankle_y"] = s * (k / 2.0)
    return pose


def build_poppy_cfg() -> ArticulationCfg:
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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, STANDING_PELVIS_Z),
                                                   joint_pos={".*": 0.0}),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"],
                effort_limit=EFFORT_MX64, effort_limit_sim=EFFORT_MX64,
                velocity_limit_sim=8.2, stiffness=KP_HIP_Y, damping=KD_HIP_Y,
            ),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
                effort_limit=EFFORT_MX28, effort_limit_sim=EFFORT_MX28,
                velocity_limit_sim=7.0, stiffness=KP_OTHER, damping=KD_OTHER,
            ),
        },
    )


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


# ---------------------------------------------------------------------------
# 支撑多边形与几何工具
# ---------------------------------------------------------------------------

def quat_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def foot_sole_corners_world(body_name: str, pos, quat):
    """把脚掌脚底面的 4 个角点变换到世界系（取 xy 投影）。"""
    aabb = FOOT_AABB_LOCAL[body_name]
    R = quat_to_matrix(quat)
    corners = []
    for x in aabb["x"]:
        for y in aabb["y"]:
            v = [x, y, aabb["sole_z"]]
            wx = float(pos[0]) + sum(R[0][k] * v[k] for k in range(3))
            wy = float(pos[1]) + sum(R[1][k] * v[k] for k in range(3))
            wz = float(pos[2]) + sum(R[2][k] * v[k] for k in range(3))
            corners.append((wx, wy, wz))
    return corners


def convex_hull(points):
    """单调链求凸包（Andrew's monotone chain）。points: [(x,y), ...]"""
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


def point_in_polygon(pt, poly) -> bool:
    """射线法判断点是否在多边形内（含边界由容差处理）。"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside


def distance_to_polygon_border(pt, poly) -> float:
    """点到多边形边界的最短距离（带符号：在内为正）。"""
    x, y = pt
    best = 1e9
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        px, py = x1 + t * dx, y1 + t * dy
        best = min(best, math.hypot(x - px, y - py))
    return best if point_in_polygon(pt, poly) else -best


# ---------------------------------------------------------------------------
# Part 1 · 静力学判据
# ---------------------------------------------------------------------------

def quat_to_euler_deg(q):
    """四元数 → (roll, pitch, yaw) 度，用于打印脚掌 link 的实际朝向。"""
    w, x, y, z = [float(v) for v in q]
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = math.degrees(math.atan2(sinr, cosr))
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.degrees(math.asin(sinp))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.degrees(math.atan2(siny, cosy))
    return roll, pitch, yaw


def static_analysis(robot: Articulation, sim: SimulationContext, pose: dict,
                    settle_steps: int = 200, debug: bool = False) -> dict:
    """准静态测量：每步只重置【基座位姿】，关节状态不重置。

    为什么不重置关节状态：这样重力会通过 PD 加载到关节上，稳态的
    |q_actual - q_target| 就是**静态跟踪误差** —— 直接反映 PD 刚度够不够。
    基座每步重置则保证机器人不会因倒立摆失稳而倒下，把"几何/刚度"问题
    和"平衡控制"问题彻底分开。
    """
    pose_t = pose_to_tensor(robot, pose)
    for _ in range(settle_steps):
        reset_robot(robot, pose_t, base_z=STANDING_PELVIS_Z)
        robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)
    reset_robot(robot, pose_t, base_z=STANDING_PELVIS_Z)
    robot.set_joint_position_target(pose_t)
    sim.step()
    robot.update(PHYS_DT)

    body_names = list(robot.body_names)
    masses = robot.data.default_mass[0].to(DEVICE)   # default_mass 在 CPU 上，必须搬到 GPU
    com_pos = getattr(robot.data, "body_com_pos_w", None)
    if com_pos is None:
        com_pos = robot.data.body_pos_w
    com = com_pos[0]
    total_m = masses.sum()
    com_w = (com * masses.unsqueeze(1)).sum(dim=0) / total_m
    root = robot.data.root_pos_w[0]

    com_rel = [float(com_w[i] - root[i]) for i in range(3)]

    # 支撑多边形 = 两只脚脚底面 8 个角点的凸包
    pts = []
    for name in ["l_foot", "r_foot"]:
        i = body_names.index(name)
        pts += foot_sole_corners_world(name, robot.data.body_pos_w[0][i],
                                       robot.data.body_quat_w[0][i])
    if debug:
        print("\n  [调试] 零位姿态下脚 link 的实际朝向与脚底角点")
        for name in ["l_foot", "r_foot"]:
            i = body_names.index(name)
            pos = robot.data.body_pos_w[0][i]
            q = robot.data.body_quat_w[0][i]
            r, p, yw = quat_to_euler_deg(q)
            print(f"    {name}: link 世界位置 ({float(pos[0]):+.4f}, {float(pos[1]):+.4f}, "
                  f"{float(pos[2]):+.4f})  朝向 rpy = ({r:+.2f}°, {p:+.2f}°, {yw:+.2f}°)")
            print(f"      四元数 (w,x,y,z) = ({float(q[0]):+.4f}, {float(q[1]):+.4f}, "
                  f"{float(q[2]):+.4f}, {float(q[3]):+.4f})")
            for c in foot_sole_corners_world(name, pos, q):
                print(f"      角点 ({c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f})")
    poly = convex_hull([(p[0], p[1]) for p in pts])
    com_xy = (float(com_w[0]), float(com_w[1]))
    inside = point_in_polygon(com_xy, poly)
    margin = distance_to_polygon_border(com_xy, poly)

    h_com = float(com_w[2]) - min(p[2] for p in pts)      # 质心相对地面高度
    tau = math.sqrt(h_com / 9.81) if h_com > 0 else 0.0

    # 静态跟踪误差：重力通过 PD 加载到关节上后的稳态偏差 —— 直接反映刚度够不够
    jerr = (robot.data.joint_pos[0] - pose_t[0]).abs()

    return {
        "com_world": [float(v) for v in com_w],
        "com_rel_pelvis": com_rel,
        "support_polygon_xy": [[float(x), float(y)] for x, y in poly],
        "com_inside_support": bool(inside),
        "com_margin_m": float(margin),
        "h_com_above_ground": h_com,
        "tau_inverted_pendulum_s": tau,
        "pelvis_z": float(root[2]),
        "sole_min_z": float(min(p[2] for p in pts)),
        "total_mass": float(total_m),
        "joint_err_max": float(jerr.max()),
        "joint_err": {n: float(jerr[i]) for i, n in enumerate(robot.joint_names)},
    }


# ---------------------------------------------------------------------------
# Part 2 · 动力学：到倒计时
# ---------------------------------------------------------------------------

def dynamic_analysis(robot: Articulation, sim: SimulationContext, pose: dict,
                     base_z: float, seconds: float) -> dict:
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
        if step % 10 == 0:                     # 每 0.05 s 记一次
            grav = torch.tensor([0.0, 0.0, -1.0], device=DEVICE).repeat(robot.num_instances, 1)
            pg = quat_rotate_inverse(robot.data.root_quat_w, grav)
            tilt = float(torch.rad2deg(torch.acos((-pg[0, 2]).clamp(-1.0, 1.0))))
            series.append({
                "t": step * PHYS_DT,
                "tilt": tilt,
                "pelvis_z": float(robot.data.root_pos_w[0, 2]),
                "joint_err_max": float((robot.data.joint_pos[0] - pose_t[0]).abs().max()),
            })

    topple_t = None
    for rec in series:
        if rec["tilt"] > 15.0:
            topple_t = rec["t"]
            break
    return {"series": series, "topple_time": topple_t}


# ---------------------------------------------------------------------------

def finish() -> None:
    """硬退出 —— 原因见 d3_pose_ident.py 里 finish() 的说明（close() 会挂住）。"""
    import sys as _sys
    _sys.stdout.flush()
    _sys.stderr.flush()
    os._exit(0)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE,
                                                    gravity=(0.0, 0.0, -9.81)))
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    robot = Articulation(build_poppy_cfg())
    sim.reset()

    print("=" * 100)
    print("D3 · rest pose 静力学判据 + 到倒计时")
    print("=" * 100)
    print(f"脚底相对 foot link 原点: l_foot {FOOT_AABB_LOCAL['l_foot']['sole_z']:.4f} m, "
          f"r_foot {FOOT_AABB_LOCAL['r_foot']['sole_z']:.4f} m")
    print(f"直立时 pelvis 高度（推算）: {STANDING_PELVIS_Z:.4f} m")
    print(f"总质量: {float(robot.data.default_mass.sum()):.3f} kg")
    print("\n注意：PD 只能稳关节角，直立是【不稳定】的倒立摆平衡。")
    print("     所以「姿态对不对」必须用静力学判据（CoM 是否落在支撑多边形内），")
    print("     而不是「放开后能不能自己站住」—— 后者任何姿态都会倒（这正是 RL 要学的）。\n")

    results = []

    # ---------- 静态：多个屈膝量 ----------
    print("-" * 100)
    print("Part 1 · 静力学：质心投影 vs 支撑多边形（决定性判据）")
    print("-" * 100)
    print(f"{'屈膝 k':>7} {'CoM x':>9} {'CoM y':>9} {'CoM z':>9} "
          f"{'支撑内?':>8} {'边界余量':>10} {'h_com':>8} {'τ 倒立摆':>10}  判定")
    print("-" * 100)
    for k in [0.0, 0.10, 0.20, 0.30, 0.45]:
        pose = build_pose(k)
        st = static_analysis(robot, sim, pose, settle_steps=250, debug=(k == 0.0))
        com = st["com_rel_pelvis"]
        verdict = "静态稳定" if st["com_inside_support"] else "★质心在支撑外★"
        print(f"{k:>7.2f} {com[0]:>9.4f} {com[1]:>9.4f} {com[2]:>9.4f} "
              f"{('是' if st['com_inside_support'] else '否'):>8} {st['com_margin_m']:>10.4f} "
              f"{st['h_com_above_ground']:>8.4f} {st['tau_inverted_pendulum_s']:>9.4f}s  {verdict}")
        results.append({"k": k, "static": st})

    print("\n  支撑多边形（xy，单位 m）：")
    for r in results:
        if r["k"] == 0.0:
            poly = r["static"]["support_polygon_xy"]
            print("    " + " → ".join(f"({x:+.4f},{y:+.4f})" for x, y in poly))
            print(f"    x 范围 [{min(p[0] for p in poly):+.4f}, {max(p[0] for p in poly):+.4f}]，"
                  f"y 范围 [{min(p[1] for p in poly):+.4f}, {max(p[1] for p in poly):+.4f}]")
            print(f"    → 脚掌前后只有 {max(p[1] for p in poly) - min(p[1] for p in poly):.4f} m 深，"
                  f"所以质心在 y 方向的容错极小 —— 这正是 Poppy 难站的原因")

    # ---------- 动态：到倒计时 ----------
    print("\n" + "-" * 100)
    print("Part 2 · 动力学：从真实站立高度释放，看到倒用时（用于对照，不作为判据）")
    print("-" * 100)
    print(f"释放高度 = {STANDING_PELVIS_Z + 0.004:.4f} m（站立高度 + 4 mm，落差可忽略）")
    print(f"{'屈膝 k':>7} {'Tilt@0.1s':>11} {'Tilt@0.3s':>11} {'Tilt@0.5s':>11} "
          f"{'z@0.3s':>9} {'到倒用时':>10} {'前0.5s关节误差':>15}")
    print("-" * 100)
    for k in [0.0, 0.10, 0.20, 0.30]:
        pose = build_pose(k)
        dy = dynamic_analysis(robot, sim, pose, STANDING_PELVIS_Z + 0.004, args_cli.drop_seconds)

        def tilt_at(t):
            for rec in dy["series"]:
                if rec["t"] >= t:
                    return rec["tilt"]
            return dy["series"][-1]["tilt"]

        def z_at(t):
            for rec in dy["series"]:
                if rec["t"] >= t:
                    return rec["pelvis_z"]
            return dy["series"][-1]["pelvis_z"]

        early = [r["joint_err_max"] for r in dy["series"] if r["t"] <= 0.5]
        topple = f"{dy['topple_time']:.2f}s" if dy["topple_time"] is not None else "未倒"
        print(f"{k:>7.2f} {tilt_at(0.1):>10.2f}° {tilt_at(0.3):>10.2f}° {tilt_at(0.5):>10.2f}° "
              f"{z_at(0.3):>9.4f} {topple:>10} {max(early) if early else 0:>14.4f} rad")
        results[[i for i, r in enumerate(results) if r["k"] == k][0]]["dynamic"] = {
            "topple_time": dy["topple_time"],
            "tilt_at": {str(t): tilt_at(t) for t in (0.1, 0.3, 0.5, 1.0)},
            "z_at_0.3": z_at(0.3),
            "series": dy["series"],
        }

    # ---------- Part 3：PD 增益扫描（静态跟踪误差） ----------
    print("\n" + "-" * 100)
    print("Part 3 · PD 增益扫描：静态跟踪误差（基座锁住，只让重力通过 PD 加载到关节）")
    print("-" * 100)
    print("为什么这一步必须有：Part 2 显示 k=0.30 时跟踪误差达 0.31 rad（18°），")
    print("说明 kp=8 对屈膝姿态太软。刚度不够 → 策略想做的动作做不出来 → 学不会站立。")
    print("（运行中改刚度用 write_joint_stiffness_to_sim，不用重新 spawn 机器人）\n")

    k_test = 0.20
    pose = build_pose(k_test)
    print(f"测试姿态：屈膝 k = {k_test:.2f} rad（{math.degrees(k_test):.1f}°）")
    print(f"{'kp_other':>9}{'kp_hip_y':>10}{'kd_other':>10}{'最大跟踪误差':>14}{'误差角度':>10}"
          f"{'CoM 边界余量':>14}{'姿态倾角退化':>14}  判定")
    print("-" * 100)

    pd_rows = []
    for kp_other in [8.0, 15.0, 25.0, 40.0, 60.0]:
        kp_hip = kp_other * 2.5                     # D2 审计的比例：hip_y 负载最大
        kd_other = 0.03 * kp_other
        kd_hip = 0.03 * kp_hip

        stiff = torch.zeros(robot.num_instances, len(robot.joint_names), device=DEVICE)
        damp = torch.zeros_like(stiff)
        for i, jn in enumerate(robot.joint_names):
            if jn.endswith("hip_y"):
                stiff[:, i] = kp_hip
                damp[:, i] = kd_hip
            else:
                stiff[:, i] = kp_other
                damp[:, i] = kd_other
        robot.write_joint_stiffness_to_sim(stiff)
        robot.write_joint_damping_to_sim(damp)

        st = static_analysis(robot, sim, pose, settle_steps=250)
        err = st["joint_err_max"]
        verdict = ("刚度过软" if err > 0.05 else ("合适" if err > 0.005 else "可能过刚/求解器风险"))
        print(f"{kp_other:>9.1f}{kp_hip:>10.1f}{kd_other:>10.3f}{err:>14.4f}"
              f"{math.degrees(err):>9.2f}°{st['com_margin_m']:>14.4f}"
              f"{'—':>14}  {verdict}")
        pd_rows.append({"kp_other": kp_other, "kp_hip_y": kp_hip, "kd_other": kd_other,
                        "joint_err_max": err, "joint_err": st["joint_err"],
                        "com_margin_m": st["com_margin_m"]})

    print("\n  说明：静态跟踪误差 < 0.02 rad（1.1°）算合格；")
    print("       显式/implicit 刚度太大时 PhysX 求解器会不稳（表现为高频抖动），")
    print("       所以不能一味加大 kp —— D3 训练时若看到动作高频振荡，先怀疑 kp 过刚。")

    # 恢复默认刚度，避免影响后续
    stiff = torch.zeros(robot.num_instances, len(robot.joint_names), device=DEVICE)
    damp = torch.zeros_like(stiff)
    for i, jn in enumerate(robot.joint_names):
        if jn.endswith("hip_y"):
            stiff[:, i], damp[:, i] = KP_HIP_Y, KD_HIP_Y
        else:
            stiff[:, i], damp[:, i] = KP_OTHER, KD_OTHER
    robot.write_joint_stiffness_to_sim(stiff)
    robot.write_joint_damping_to_sim(damp)

    out = os.path.join(OUT_DIR, "settle_probe.json")
    with open(out, "w") as fh:
        json.dump({"standing_pelvis_z": STANDING_PELVIS_Z,
                   "foot_aabb_local": FOOT_AABB_LOCAL,
                   "results": results,
                   "pd_sweep": pd_rows}, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {out}")
    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
