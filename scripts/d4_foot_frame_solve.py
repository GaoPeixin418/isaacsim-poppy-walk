#!/usr/bin/env python3
"""D4 · 脚掌姿态解算：把「屈膝时踝关节该补多少」从猜测变成解算。

================================================================================
要解决的问题
================================================================================
行走用的默认姿态需要轻微屈膝（直腿的膝关节是奇异位形，腿长不可调；
真机直立时靠"顶死膝"承重也会让舵机长期堵转）。
屈膝 k 之后要同时满足两个条件：

  条件 1（位置）：脚要留在骨盆正下方 —— 否则质心投影跑到支撑多边形外
  条件 2（姿态）：脚掌要水平 —— 否则默认姿态就站在脚尖/脚跟上

条件 1 有干净的解析解：hip_y = +k/2。
（依据：D3 实测 ∂y_foot/∂hip_y ≈ -0.362 m/rad、∂y_foot/∂knee_y ≈ -0.180 m/rad，
 比值 2.01 → 髋补一半正好抵消膝带来的前后位移。）

条件 2 就是本脚本要解的东西。

================================================================================
★ 这里发现了一个"测量工具本身的 bug"，值得单独记下来 ★
================================================================================
D3 的标定脚本（d3_pd_calib.py）里 build_pose 是这样写的：

    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"]  = s * k
        pose[f"{side}_hip_y"]   = -s * (k / 2.0)
        pose[f"{side}_ankle_y"] = s * (k / 2.0)     # ← 用了 s，和 knee 同号

于是右腿实际被摆成了 (hip, knee, ankle) = (+k/2, -k, **-k/2**)，角度和 = -k。
而 D3 报告里写的是 "(+k/2, -k, +k/2)"，和 = 0。
两者差了整整一个 k —— 实测倾角"恰好等于 k"就是这么来的。

**结论反转**：不是"三根俯仰轴的方向不像假设那样简单相加"（那是错的），
而是**探针把镜像符号乘错了关节**。解析推导 θ = -(hip+knee+ankle) 一直是对的，
本脚本 Part 3 会用实测同时验证这两种摆法，把这件事钉死。

教训：当"实测"和"解析"矛盾时，两者都可能错。**先用一个更小的实验把
      双方各自的前提分别验证一遍**，而不是直接采信"实测"这一侧
      —— 实测也只是一个程序，程序也会写错符号。

================================================================================
方法：有符号倾角 + 中心差分求雅可比
================================================================================
1. 有符号倾角（不用 acos —— 它只有 0~180°，丢符号）
       n = 脚底法向（脚 link 局部 -y）在【基座坐标系】下的表达
       θ = atan2( n[1], -n[2] )
   零位姿态 θ=0；正负号就是脚尖上翘/下压的方向。
   它对三根俯仰轴是**精确线性**的，系数理论值为 ±1：

       右腿 θ_r = -(hip_y + knee_y + ankle_y)
       左腿 θ_l = +(hip_y + knee_y + ankle_y)      （镜像后符号翻转）

2. 中心差分验证上面的系数（不是猜的，是测的）：
       c_j = ( θ(q_j + δ) - θ(q_j - δ) ) / (2δ)

3. 解算（要求两只脚同时水平，即 Σ=0，镜像约定 r = -l 自动满足）：
       ankle = -( hip + knee )
   取 knee = -k、hip = +k/2 → ankle = +k/2。

4. 回代实测（不信任解析，只信任回代），并给出 D4 训练要用的
   默认关节角与基座初始高度。

运行：
    bash scripts/run.sh scripts/d4_foot_frame_solve.py --headless
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="D4 脚掌姿态解算")
parser.add_argument("--out", type=str, default="d4_foot_frame.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d4_foot_frame")

PHYS_DT = 0.005
DEVICE = args_cli.device or "cuda:0"
EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0
KG_RATIO = EFFORT_MX64 / EFFORT_MX28

SKY_Z = 1.0                     # 悬空高度：纯几何测量，不受接触影响
SOLE_OFFSET = 0.0350            # foot link 原点到脚底平面（D3 实测）
FOOT_LINK_Z_REL_PELVIS = -0.3860
KP_PROBE = 400.0                # 纯几何测量，刚度拉高把跟踪误差压掉
DELTA = 0.05                    # 中心差分步长。膝上限 +0.061 rad，δ=0.05 不顶限位

SIDES = ("r", "l")
LEG_JOINTS = [f"{s}_{j}" for s in SIDES
              for j in ("hip_x", "hip_z", "hip_y", "knee_y", "ankle_y")]
PITCH = ("hip_y", "knee_y", "ankle_y")

KS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


# ---------------------------------------------------------------------------
def kd_for(kp: float) -> float:
    return 2.0 * 1.0 * math.sqrt(max(kp, 1e-6) * 0.02)


def build_poppy_cfg(kp28: float, kp_hip: float) -> ArticulationCfg:
    """悬空标定用的最简配置：平地 + 高刚度驱动。"""
    return ArticulationCfg(
        prim_path="/World/Poppy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=POPPY_USD,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_linear_velocity=100.0,
                max_angular_velocity=100.0, max_depenetration_velocity=1.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8,
                solver_velocity_iteration_count=1, fix_root_link=False),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, SKY_Z), joint_pos={".*": 0.0}),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"], effort_limit=EFFORT_MX64,
                effort_limit_sim=EFFORT_MX64, velocity_limit_sim=8.2,
                stiffness=kp_hip, damping=kd_for(kp_hip)),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*(hip_x|hip_z|knee_y|ankle_y)"],
                effort_limit=EFFORT_MX28, effort_limit_sim=EFFORT_MX28,
                velocity_limit_sim=5.8, stiffness=kp28, damping=kd_for(kp28)),
        },
    )


def pose_to_tensor(robot, pose: dict[str, float]):
    v = torch.zeros(len(robot.joint_names), device=DEVICE)
    for i, n in enumerate(robot.joint_names):
        v[i] = float(pose.get(n, 0.0))
    return v.unsqueeze(0).repeat(robot.num_instances, 1)


def write_root(robot, z: float):
    n = robot.num_instances
    rp = torch.zeros(n, 7, device=DEVICE)
    rp[:, 3] = 1.0
    rp[:, 2] = z
    robot.write_root_pose_to_sim(rp)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))


def step(robot, sim, pose_t, n: int, base_z: float = SKY_Z, teleport: bool = True):
    """跑 n 步。**必须调 write_data_to_sim**（D3 踩过的坑：不调则目标不进 PhysX）。"""
    if teleport:
        robot.write_joint_state_to_sim(pose_t, torch.zeros_like(pose_t))
    for _ in range(n):
        write_root(robot, base_z)
        robot.set_joint_position_target(pose_t)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYS_DT)


def _quat_apply(quat, vec):
    """w-first 四元数旋转向量（本地实现，避免不同版本 API 差异）。"""
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    t = 2.0 * torch.stack([
        y * vec[:, 2] - z * vec[:, 1],
        z * vec[:, 0] - x * vec[:, 2],
        x * vec[:, 1] - y * vec[:, 0],
    ], dim=1)
    return vec + w.unsqueeze(1) * t + torch.stack([
        y * t[:, 2] - z * t[:, 1],
        z * t[:, 0] - x * t[:, 2],
        x * t[:, 1] - y * t[:, 0],
    ], dim=1)


def measure(robot, sim, pose, settle: int = 30, base_z: float = SKY_Z):
    """摆好姿态 → 稳态平均 → 返回每只脚的有符号倾角 / 法向 / 位置。"""
    pt = pose_to_tensor(robot, pose)
    step(robot, sim, pt, settle, base_z)

    q_sum, n_acc = None, 0
    for _ in range(10):
        write_root(robot, base_z)
        robot.set_joint_position_target(pt)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYS_DT)
        q_sum = robot.data.joint_pos.clone() if q_sum is None else q_sum + robot.data.joint_pos
        n_acc += 1
    q_mean = q_sum / n_acc

    out = {"joints_actual": {n: float(q_mean[0][i])
                             for i, n in enumerate(robot.joint_names)}}
    root_q = robot.data.root_quat_w[0]
    pz = float(robot.data.root_pos_w[0][2])
    for foot in ("l_foot", "r_foot"):
        fi = list(robot.body_names).index(foot)
        fq = robot.data.body_quat_w[0][fi]
        fp = robot.data.body_pos_w[0][fi]
        n_local = torch.tensor([0.0, -1.0, 0.0], device=DEVICE).unsqueeze(0)
        n_world = _quat_apply(fq.unsqueeze(0), n_local)[0]
        n_base = quat_rotate_inverse(root_q.unsqueeze(0), n_world.unsqueeze(0))[0]
        theta = math.atan2(float(n_base[1]), -float(n_base[2]))
        out[foot] = {
            "theta_deg": math.degrees(theta),
            "theta_rad": theta,
            "n_base": [float(x) for x in n_base],
            "pos_rel_pelvis": [float(fp[0]), float(fp[1]), float(fp[2]) - pz],
            # 脚水平（θ=0）时，脚底平面 = link 原点竖直下移 SOLE_OFFSET
            "sole_z_rel_pelvis": float(fp[2]) - pz - SOLE_OFFSET * math.cos(theta),
        }
    return out


def build_pose(k: float, ankle: float | None = None, hip: float | None = None):
    """屈膝 k 的对称姿态，遵守镜像约定 r = -l。

    hip 与 ankle 用同一个镜像符号（因为 θ = -(hip+knee+ankle)，两者要同向才能抵消膝）；
    knee 用相反的符号（屈膝）。
    """
    hip = k / 2.0 if hip is None else hip
    ac = k / 2.0 if ankle is None else ankle
    pose = {n: 0.0 for n in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * hip
        pose[f"{side}_ankle_y"] = -s * ac
    return pose


def build_pose_d3_bug(k: float):
    """复现 D3 探针的口径：踝用与 knee 同号的 s → 右腿实际是 (k/2, -k, -k/2)。"""
    pose = {n: 0.0 for n in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * (k / 2.0)
        pose[f"{side}_ankle_y"] = s * (k / 2.0)
    return pose


def finish(code=0):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def hr(n=100):
    print("-" * n)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE,
                                                   gravity=(0.0, 0.0, -9.81)))
    g = sim_utils.GroundPlaneCfg()
    g.func("/World/ground", g)

    robot = Articulation(build_poppy_cfg(KP_PROBE, KP_PROBE * KG_RATIO))
    sim.reset()

    print("=" * 100)
    print("D4 · 脚掌姿态解算（实测雅可比 + 回代验证）")
    print("=" * 100)
    print(f"  关节顺序: {list(robot.joint_names)}")
    print(f"  差分步长 δ = {DELTA} rad；悬空高度 {SKY_Z} m；kp = {KP_PROBE}")
    result = {}

    # ============ Part 1 · 零位自检 ============
    print()
    print("Part 1 · 零位自检（FK 与仿真是否一致）")
    hr()
    m0 = measure(robot, sim, {n: 0.0 for n in LEG_JOINTS})
    for foot in ("l_foot", "r_foot"):
        d = m0[foot]
        print(f"  {foot}: 倾角 {d['theta_deg']:+9.4f}°  "
              f"法向(基座系) [{d['n_base'][0]:+.3f} {d['n_base'][1]:+.3f} {d['n_base'][2]:+.3f}]  "
              f"脚相对骨盆 [{d['pos_rel_pelvis'][0]:+.5f} {d['pos_rel_pelvis'][1]:+.5f} "
              f"{d['pos_rel_pelvis'][2]:+.5f}]")
    print(f"  判读：倾角 ≈ 0、法向 = (0,0,-1)、脚相对骨盆 z ≈ {FOOT_LINK_Z_REL_PELVIS}")
    result["zero_pose"] = m0

    # ============ Part 2 · 实测雅可比 ============
    print()
    print("Part 2 · 实测雅可比  c_j = ∂θ_foot/∂q_j  （理论值：右腿 [-1,-1,-1]，左腿 [+1,+1,+1]）")
    hr()
    print(f"  {'工作点':>10}{'腿':>4}{'c(hip_y)':>12}{'c(knee_y)':>12}{'c(ankle_y)':>12}"
          f"{'θ 在该点(°)':>14}   与理论一致?")
    jac = {}
    for k_op, tag in ((0.0, "零位"), (0.20, "k=0.20")):
        base = build_pose(k_op)
        base_m = measure(robot, sim, base)
        for side in SIDES:
            cs = {}
            for j in PITCH:
                name = f"{side}_{j}"
                p_plus, p_minus = dict(base), dict(base)
                p_plus[name] += DELTA
                p_minus[name] -= DELTA
                tp = measure(robot, sim, p_plus)[f"{side}_foot"]["theta_rad"]
                tm = measure(robot, sim, p_minus)[f"{side}_foot"]["theta_rad"]
                cs[j] = (tp - tm) / (2.0 * DELTA)
            sgn = -1.0 if side == "r" else +1.0
            ok = all(abs(cs[j] - sgn) < 1e-6 for j in PITCH)
            jac[f"{tag}/{side}"] = cs
            print(f"  {tag:>10}{side:>4}{cs['hip_y']:>12.4f}{cs['knee_y']:>12.4f}"
                  f"{cs['ankle_y']:>12.4f}{base_m[f'{side}_foot']['theta_deg']:>14.4f}"
                  f"   {'✔' if ok else '✘'}")
    result["jacobian"] = jac

    # ============ Part 3 · 两种摆法对比（复现 D3 的 bug） ============
    print()
    print("Part 3 · 钉死这个 bug：两种踝符号摆法的实측倾角对比")
    hr()
    print(f"  {'k':>7}{'D3 口径 右腿(hip,knee,ankle)':>32}{'Σ':>8}{'倾角(°)':>11}"
          f"{'':>4}{'正确口径 右腿':>22}{'Σ':>8}{'倾角(°)':>11}")
    result["bug_compare"] = []
    for k in KS:
        d3_pose = build_pose_d3_bug(k)
        ok_pose = build_pose(k, ankle=k / 2.0)
        d3_m = measure(robot, sim, d3_pose)
        ok_m = measure(robot, sim, ok_pose)
        d3_leg = (d3_pose["r_hip_y"], d3_pose["r_knee_y"], d3_pose["r_ankle_y"])
        ok_leg = (ok_pose["r_hip_y"], ok_pose["r_knee_y"], ok_pose["r_ankle_y"])
        row = {
            "k": k, "d3_leg": d3_leg, "sum_d3": sum(d3_leg),
            "theta_d3_deg": d3_m["r_foot"]["theta_deg"],
            "ok_leg": ok_leg, "sum_ok": sum(ok_leg),
            "theta_ok_deg": ok_m["r_foot"]["theta_deg"],
            "theta_ok_left_deg": ok_m["l_foot"]["theta_deg"],
        }
        result["bug_compare"].append(row)
        print(f"  {k:>7.2f}   ({d3_leg[0]:+.2f}, {d3_leg[1]:+.2f}, {d3_leg[2]:+.2f})"
              f"{sum(d3_leg):>8.2f}{d3_m['r_foot']['theta_deg']:>11.4f}"
              f"      ({ok_leg[0]:+.2f}, {ok_leg[1]:+.2f}, {ok_leg[2]:+.2f})"
              f"{sum(ok_leg):>8.2f}{ok_m['r_foot']['theta_deg']:>11.4f}")
    print()
    print("  判读：D3 口径的 Σ = -k → 倾角恰好 = k（【倾角等于 k】的全部原因就是这个）")
    print("        正确口径 Σ = 0 → 倾角 = 0 ✔（左右脚都是）")

    # ============ Part 4 · 落地高度 ============
    print()
    print("Part 4 · 各屈膝量的几何量（用正确姿态）")
    hr()
    print(f"  {'k':>7}{'hip_y':>10}{'knee_y':>10}{'ankle_y':>10}"
          f"{'左倾角(°)':>12}{'右倾角(°)':>12}{'脚相对骨盆 y':>14}{'脚底相对骨盆 z':>16}"
          f"{'→ 站立骨盆高(m)':>18}")
    rows = []
    for k in KS:
        pose = build_pose(k, ankle=k / 2.0)
        m = measure(robot, sim, pose)
        h = -m["r_foot"]["sole_z_rel_pelvis"]
        r = {
            "k": k, "hip_y": k / 2.0, "knee_y": -k, "ankle_y": k / 2.0,
            "theta_left_deg": m["l_foot"]["theta_deg"],
            "theta_right_deg": m["r_foot"]["theta_deg"],
            "foot_y_rel_pelvis": m["r_foot"]["pos_rel_pelvis"][1],
            "sole_z_rel_pelvis": m["r_foot"]["sole_z_rel_pelvis"],
            "standing_pelvis_z": h,
        }
        rows.append(r)
        print(f"  {k:>7.2f}{k/2:>10.4f}{-k:>10.4f}{k/2:>10.4f}"
              f"{m['l_foot']['theta_deg']:>12.4f}{m['r_foot']['theta_deg']:>12.4f}"
              f"{m['r_foot']['pos_rel_pelvis'][1]:>14.5f}"
              f"{m['r_foot']['sole_z_rel_pelvis']:>16.5f}{h:>18.5f}")
    print("  注：脚相对骨盆 y 越接近 0，说明【脚在骨盆正下方】这个位置条件满足得越好")
    print(f"      （D3 实测比值 2.01 → hip 补 k/2 后残差应该在 1e-4 m 量级）")
    result["geometry"] = rows

    # ============ Part 5 · 落地自检（带重力 + 真实接触） ============
    print()
    print("Part 5 · 落地自检：把基座锁在解算高度，带重力跑 1 s，看脚底是否正好贴地")
    hr()
    print(f"  {'k':>7}{'锁定高度(m)':>14}{'关节最大跟踪误差(rad)':>24}"
          f"{'左倾角(°)':>12}{'右倾角(°)':>12}{'脚底最低点(m)':>16}  判定")
    land = []
    for r in rows:
        k = r["k"]
        pose = build_pose(k, ankle=k / 2.0)
        pt = pose_to_tensor(robot, pose)
        base_z = r["standing_pelvis_z"]
        step(robot, sim, pt, 200, base_z)          # 0.5 s 收敛
        q_err_max = float((robot.data.joint_pos - pt).abs().max())
        m = measure(robot, sim, pose, settle=20, base_z=base_z)
        # measure 里会再摆一次姿态 + 锁基座，这里只取几何量
        lowest = min(m["l_foot"]["pos_rel_pelvis"][2] - SOLE_OFFSET * math.cos(m["l_foot"]["theta_rad"]),
                     m["r_foot"]["pos_rel_pelvis"][2] - SOLE_OFFSET * math.cos(m["r_foot"]["theta_rad"]))
        # 基座 z + 相对量 = 脚底世界高度
        lowest_w = base_z + lowest
        verdict = "✔ 贴地" if abs(lowest_w) < 0.003 else ("偏低(穿地)" if lowest_w < 0 else "偏高(悬空)")
        land.append({"k": k, "base_z": base_z, "q_err_max": q_err_max,
                     "sole_lowest_w": lowest_w, "verdict": verdict})
        print(f"  {k:>7.2f}{base_z:>14.5f}{q_err_max:>24.6f}"
              f"{m['l_foot']['theta_deg']:>12.4f}{m['r_foot']['theta_deg']:>12.4f}"
              f"{lowest_w:>16.5f}  {verdict}")
    result["landing"] = land

    # ============ 建议值 ============
    k_rec = 0.20
    rec = next(r for r in rows if abs(r["k"] - k_rec) < 1e-9)
    print()
    print("=" * 100)
    print(f"★ D4 建议默认姿态（k = {k_rec} rad = {math.degrees(k_rec):.2f}°）")
    hr()
    print(f"    r_knee_y  = {-k_rec:+.3f}      l_knee_y  = {+k_rec:+.3f}")
    print(f"    r_hip_y   = {+k_rec/2:+.3f}      l_hip_y   = {-k_rec/2:+.3f}")
    print(f"    r_ankle_y = {+k_rec/2:+.3f}      l_ankle_y = {-k_rec/2:+.3f}")
    print(f"    STANDING_PELVIS_Z → {rec['standing_pelvis_z']:.5f}")
    print("    其余关节（hip_x / hip_z）保持 0.0")
    print("    镜像约定：右腿 = -左腿（全部 5 个关节一致）")
    result["recommended"] = {
        "k": k_rec, "hip_y": k_rec / 2, "knee_y": -k_rec, "ankle_y": k_rec / 2,
        "standing_pelvis_z": rec["standing_pelvis_z"],
        "mirror_rule": "r = -l", "zeros": ["hip_x", "hip_z"],
    }
    print("=" * 100)

    with open(os.path.join(OUT_DIR, args_cli.out), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  结果已写入 {os.path.join(OUT_DIR, args_cli.out)}")
    finish(0)


if __name__ == "__main__":
    main()
