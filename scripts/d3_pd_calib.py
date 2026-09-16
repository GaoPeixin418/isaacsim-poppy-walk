#!/usr/bin/env python3
"""D3 · PD 增益标定（最终修正版）。

================================================================================
前两版错在哪 —— 这个坑值得单独记住
================================================================================
两版标定都得出「关节误差恒等于指令值、且对 kp 完全不敏感」这种物理上不可能的结果。

真凶是一行缺失的调用：

    robot.set_joint_position_target(target)      # 只是把目标写进【内部缓冲】
    robot.write_data_to_sim()                    # ← 缺这一行，目标永远不会进 PhysX
    sim.step()

Isaac Lab 的文档写得很明确：
    set_joint_position_target: "This function does not apply the joint targets to the
    simulation. It only fills the buffers... To apply the joint targets, call the
    write_data_to_sim function."

于是 PhysX 里的驱动目标一直停在 0（初始化时的默认值），关节被"按"在零位：
    · 实测"误差" = |指令|（因为实际值恒为 0，指令是 0.2 就显示 0.2）
    · 关节速度恒为 0（关节根本没被允许动）
    · 加大 kp 完全无效（目标没变，误差当然不变）

为什么平时写训练代码不会踩到这个坑：
    ManagerBasedRLEnv.step() 内部会调用 scene.write_data_to_sim()，
    这一步由框架代劳。而写独立探针脚本时没有这层保护，就必须自己补上。

教训：**当实测结果"物理上不可能"时，不要再调参数，要去验证数据通路本身。**
      本次是靠一个最小化的输入/输出自检（写 → 读 → 驱动 → USD）定位的，
      而不是继续猜物理。

================================================================================
本脚本测量的三件事
================================================================================
  Part 2  静运动学：屈膝让腿长怎么变（解释之前"站立高度随屈膝升高"的疑点）
  Part 3  悬空法标定 PD：脚离地、只受重力 → 稳态误差 × kp = 该关节要扛的重力矩
  Part 4  接触法复核：真实站在地上，确认最终选的增益可用
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="D3 PD 标定（最终版）")
parser.add_argument("--out", type=str, default="pd_calib.json")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

PHYS_DT = 0.005
DEVICE = args_cli.device or "cuda:0"
EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0
KG_RATIO = EFFORT_MX64 / EFFORT_MX28

STANDING_PELVIS_Z = 0.42125     # 由 d3_static_pd_probe.py 收敛标定实测（零位姿态）
BENT_K_STANDING_Z = 0.43210     # 同上，k=0.20
LIFT_Z = 0.03                   # 悬空法抬升量
SKY_Z = 1.0                     # 纯静运动学测量用的高度（脚完全离地）

LEG_JOINTS = ["r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
              "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y"]

SOLE_OFFSET = 0.0350            # foot link 原点到脚底平面的距离（实测）
FOOT_LINK_Z_REL_PELVIS = -0.3860

DAMP_ZETA = 1.0
J_EFF_EST = 0.02


def kd_for(kp: float) -> float:
    """kd = 2·ζ·sqrt(kp·J_eff)。kd 应随 sqrt(kp) 增长，不是线性。"""
    return 2.0 * DAMP_ZETA * math.sqrt(max(kp, 1e-6) * J_EFF_EST)


def build_pose(k: float, ankle_comp: float | None = None) -> dict[str, float]:
    """屈膝 k 的 rest pose（镜像约定 r = -l，已由 5 对关节验证）。

    髋补 +k/2、踝补 +k/2 以保持脚掌水平（三根俯仰轴都绕 base 的 X 轴，故角度直接相加）。
    """
    ac = k / 2.0 if ankle_comp is None else ankle_comp
    pose = {n: 0.0 for n in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * (k / 2.0)
        pose[f"{side}_ankle_y"] = s * ac
    return pose


def build_poppy_cfg(kp28, kp_hip) -> ArticulationCfg:
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
            pos=(0.0, 0.0, STANDING_PELVIS_Z), joint_pos={".*": 0.0}),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"], effort_limit=EFFORT_MX64,
                effort_limit_sim=EFFORT_MX64, velocity_limit_sim=8.2,
                stiffness=kp_hip, damping=kd_for(kp_hip)),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
                effort_limit=EFFORT_MX28, effort_limit_sim=EFFORT_MX28,
                velocity_limit_sim=7.0, stiffness=kp28, damping=kd_for(kp28)),
        },
    )


# ---------------------------------------------------------------------------
def pose_to_tensor(robot, pose):
    v = torch.zeros(len(robot.joint_names), device=DEVICE)
    for i, n in enumerate(robot.joint_names):
        v[i] = float(pose.get(n, 0.0))
    return v.unsqueeze(0).repeat(robot.num_instances, 1)


def write_root(robot, z):
    n = robot.num_instances
    rp = torch.zeros(n, 7, device=DEVICE)
    rp[:, 3] = 1.0
    rp[:, 2] = z
    robot.write_root_pose_to_sim(rp)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))


def step(robot, sim, base_z, pose_t, n, teleport_first=False):
    """跑 n 步。**必须调 write_data_to_sim** —— 这正是前两版的 bug。"""
    if teleport_first:
        robot.write_joint_state_to_sim(pose_t, torch.zeros_like(pose_t))
    for _ in range(n):
        write_root(robot, base_z)
        robot.set_joint_position_target(pose_t)
        robot.write_data_to_sim()            # ← 关键
        sim.step()
        robot.update(PHYS_DT)


def set_gains(robot, kp28, kp_hip):
    """运行时改刚度。ImplicitActuator 只在初始化时写刚度，运行时改不会被覆盖
    （_apply_actuator_model 对 implicit 是空操作），所以这样改是安全有效的。"""
    stiff = torch.zeros(robot.num_instances, len(robot.joint_names), device=DEVICE)
    damp = torch.zeros_like(stiff)
    for i, jn in enumerate(robot.joint_names):
        kp = kp_hip if jn.endswith("hip_y") else kp28
        stiff[:, i] = kp
        damp[:, i] = kd_for(kp)
    robot.write_joint_stiffness_to_sim(stiff)
    robot.write_joint_damping_to_sim(damp)
    return stiff, damp


def quat_to_matrix(q):
    w, x, y, z = [float(v) for v in q]
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]


def foot_tilt_deg(robot):
    out = []
    for name in ("l_foot", "r_foot"):
        i = list(robot.body_names).index(name)
        R = quat_to_matrix(robot.data.body_quat_w[0][i])
        out.append(math.degrees(math.acos(max(-1.0, min(1.0, R[2][1])))))
    return max(out)


def measure_steady(robot, sim, pose_t, base_z, settle_s=1.5, measure_s=0.25):
    """先收敛，再在稳态窗口内取平均。返回逐关节明细。"""
    n_settle = int(settle_s / PHYS_DT)
    step(robot, sim, base_z, pose_t, n_settle, teleport_first=True)

    n = max(1, int(measure_s / PHYS_DT))
    q_sum = torch.zeros_like(pose_t)
    v_abs_max = 0.0
    for _ in range(n):
        write_root(robot, base_z)
        robot.set_joint_position_target(pose_t)
        robot.write_data_to_sim()
        sim.step()
        robot.update(PHYS_DT)
        q_sum += robot.data.joint_pos
        v_abs_max = max(v_abs_max, float(robot.data.joint_vel[0].abs().max()))
    q_mean = q_sum / n

    comp = getattr(robot.data, "computed_torque", None)
    err = (q_mean - pose_t)[0]
    lim = robot.data.joint_pos_limits[0]
    joints = {}
    for i, jn in enumerate(robot.joint_names):
        lo, hi = float(lim[i][0]), float(lim[i][1])
        q = float(q_mean[0][i])
        joints[jn] = {
            "target": float(pose_t[0][i]), "actual": q, "err": float(err[i]),
            "err_deg": math.degrees(float(err[i])),
            "vel": float(robot.data.joint_vel[0][i]),
            "limit": [lo, hi],
            "at_limit": bool(min(abs(q - lo), abs(q - hi)) < 2e-3),
            # implicit 执行器的 applied_torque 永远是 0（PhysX 不暴露），
            # 但 Isaac Lab 会用 PD 公式算一个近似值放进 computed_torque
            "computed_torque": float(comp[0][i]) if comp is not None else float("nan"),
        }
    return {"joint_err_max": float(err.abs().max()),
            "joint_err_rms": float((err ** 2).mean().sqrt()),
            "max_abs_joint_vel": v_abs_max,
            "foot_tilt_deg": foot_tilt_deg(robot),
            "joints": joints}


def finish():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE,
                                                    gravity=(0.0, 0.0, -9.81)))
    g = sim_utils.GroundPlaneCfg()
    g.func("/World/ground", g)

    kp0 = 12.5
    robot = Articulation(build_poppy_cfg(kp0, kp0 * KG_RATIO))
    sim.reset()

    names = list(robot.joint_names)
    lim = robot.data.joint_pos_limits[0]
    print("=" * 104)
    print("D3 · PD 增益标定（最终修正版：补上 write_data_to_sim）")
    print("=" * 104)
    print(f"总质量 {float(robot.data.default_mass.sum()):.3f} kg；"
          f"关节 {len(names)} 个；刚体 {len(robot.body_names)} 个")
    print()
    print("Part 1 · 关节限位与执行器（标定对象）")
    print("-" * 104)
    print(f"  {'关节':>11}{'下限(rad)':>12}{'上限(rad)':>12}{'下限(°)':>10}{'上限(°)':>10}"
          f"{'力矩上限':>10}{'型号':>9}{'kp初始':>9}")
    for i, jn in enumerate(names):
        lo, hi = float(lim[i][0]), float(lim[i][1])
        mx = EFFORT_MX64 if jn.endswith("hip_y") else EFFORT_MX28
        model = "MX-64" if jn.endswith("hip_y") else "MX-28"
        kp = kp0 * KG_RATIO if jn.endswith("hip_y") else kp0
        print(f"  {jn:>11}{lo:>12.4f}{hi:>12.4f}{math.degrees(lo):>10.2f}"
              f"{math.degrees(hi):>10.2f}{mx:>10.2f}{model:>9}{kp:>9.1f}")

    # ================= Part 2 · 静运动学：屈膝对腿长的影响 =================
    print()
    print("Part 2 · 静运动学测量（基座抬到 1.0 m 悬空，纯几何，无接触无重力影响）")
    print("-" * 104)
    print("  目的：解释之前那个疑点 —— 为什么屈膝后'站立高度'反而升高了？")
    print("        （屈膝应该让腿变短、骨盆变低才对）")
    print()
    print(f"  {'屈膝 k':>8}{'foot link z':>13}{'相对骨盆':>12}{'脚底 z':>11}{'脚掌倾角':>11}"
          f"{'推算站立高度':>14}{'实测标定值':>12}{'差(mm)':>9}")
    print("-" * 104)
    kin_rows = []
    for k in (0.0, 0.10, 0.20, 0.30, 0.45, 0.60):
        set_gains(robot, 400.0, 400.0 * KG_RATIO)
        p = build_pose(k)
        pt = pose_to_tensor(robot, p)
        step(robot, sim, SKY_Z, pt, 40, teleport_first=True)
        fi = list(robot.body_names).index("l_foot")
        fz = float(robot.data.body_pos_w[0][fi][2])
        pz = float(robot.data.root_pos_w[0][2])
        rel = fz - pz
        sole_z = fz + 0.0 - SOLE_OFFSET          # 脚底平面相对骨盆原点的高度
        pred_height = -rel + SOLE_OFFSET
        actual = {0.0: 0.42125, 0.10: 0.42533, 0.20: 0.43210,
                  0.30: 0.43737, 0.45: 0.44337, 0.60: 0.44600}.get(k)
        print(f"  {k:>8.2f}{fz:>13.5f}{rel:>12.5f}{sole_z:>11.5f}{foot_tilt_deg(robot):>11.4f}"
              f"{pred_height:>14.5f}{(actual if actual else float('nan')):>12.5f}"
              f"{((pred_height - actual) * 1000 if actual else float('nan')):>9.2f}")
        kin_rows.append({"k": k, "foot_z": fz, "pelvis_z": pz,
                         "foot_rel_pelvis": rel, "sole_rel_pelvis": sole_z,
                         "tilt_deg": foot_tilt_deg(robot),
                         "predicted_standing_height": pred_height,
                         "empirical_standing_height": actual,
                         "joints_actual": [float(x) for x in robot.data.joint_pos[0]]})

    # ================= Part 3 · 悬空法 PD 标定 =================
    print()
    print("Part 3 · PD 标定（悬空法：基座抬高 3 cm，脚完全离地 → 只受重力）")
    print("-" * 104)
    print("  稳态时 kp·(指令-实际) = 该关节承担的重力矩，所以 所需力矩 = kp × 稳态误差。")
    print("  自校验：如果 PD 真的按 kp 工作，不同 kp 算出的『所需力矩』应当基本一致。")
    print()
    print(f"  {'姿态':>10}{'kp(MX-28)':>11}{'kp(hip)':>9}{'误差max(rad)':>14}{'误差(°)':>9}"
          f"{'误差rms':>10}{'关节最大速度':>14}{'τ_req=kp·err':>14}  判定")
    print("-" * 104)
    z_lift = STANDING_PELVIS_Z + LIFT_Z
    rows = []
    for k in (0.0, 0.20):
        for kp28 in (6.0, 12.5, 25.0, 50.0, 100.0):
            set_gains(robot, kp28, kp28 * KG_RATIO)
            m = measure_steady(robot, sim, pose_to_tensor(robot, build_pose(k)), z_lift)
            tau = m["joint_err_max"] * kp28
            verdict = "★未收敛，无效★" if m["max_abs_joint_vel"] > 0.05 else "有效"
            print(f"  {'k=' + format(k, '.2f'):>10}{kp28:>11.1f}{kp28 * KG_RATIO:>9.1f}"
                  f"{m['joint_err_max']:>14.5f}{math.degrees(m['joint_err_max']):>9.3f}"
                  f"{m['joint_err_rms']:>10.5f}{m['max_abs_joint_vel']:>14.5f}"
                  f"{tau:>14.5f}  {verdict}")
            rows.append({"k": k, "kp28": kp28, "tau_req_from_err": tau,
                         "joints": m["joints"],
                         **{a: b for a, b in m.items() if a != "joints"}})
        print()

    print("  逐关节明细（取 kp(MX-28)=50 那一组）")
    print(f"  {'姿态':>8}{'关节':>11}{'指令':>10}{'实际':>10}{'误差(rad)':>12}{'误差(°)':>9}"
          f"{'τ_req(N·m)':>12}{'力限':>8}{'占比':>8}{'kp@0.02':>10}{'kp@0.05':>10}")
    print("-" * 104)
    tau_req = {}
    for k in (0.0, 0.20):
        rec = next(r for r in rows if r["k"] == k and r["kp28"] == 50.0)
        for jn, j in rec["joints"].items():
            mx = EFFORT_MX64 if jn.endswith("hip_y") else EFFORT_MX28
            tq = 50.0 * abs(j["err"])
            tau_req[f"{jn}@k={k}"] = {"err": j["err"], "tau_req": tq, "limit": mx,
                                      "kp_at_0.02": tq / 0.02, "kp_at_0.05": tq / 0.05}
            print(f"  {k:>8.2f}{jn:>11}{j['target']:>10.4f}{j['actual']:>10.4f}"
                  f"{j['err']:>12.5f}{j['err_deg']:>9.3f}{tq:>12.4f}{mx:>8.2f}"
                  f"{tq / mx * 100:>7.1f}%{tq / 0.02:>10.1f}{tq / 0.05:>10.1f}")
        print()

    # ================= Part 4 · 接触法复核 =================
    print("Part 4 · 接触法复核（基座放在标定站立高度，脚真实受力）")
    print("-" * 104)
    print(f"  {'姿态':>10}{'kp(MX-28)':>11}{'误差max(rad)':>14}{'误差(°)':>9}"
          f"{'关节最大速度':>14}{'脚掌倾角(°)':>13}  判定")
    print("-" * 104)
    contact = []
    for k, zc in ((0.0, STANDING_PELVIS_Z), (0.20, BENT_K_STANDING_Z)):
        for kp28 in (12.5, 25.0, 50.0):
            set_gains(robot, kp28, kp28 * KG_RATIO)
            m = measure_steady(robot, sim, pose_to_tensor(robot, build_pose(k)), zc)
            verdict = "★未收敛★" if m["max_abs_joint_vel"] > 0.05 else (
                "合适" if m["joint_err_max"] < 0.05 else "偏软")
            print(f"  {'k=' + format(k, '.2f'):>10}{kp28:>11.1f}{m['joint_err_max']:>14.5f}"
                  f"{math.degrees(m['joint_err_max']):>9.3f}{m['max_abs_joint_vel']:>14.5f}"
                  f"{m['foot_tilt_deg']:>13.4f}  {verdict}")
            contact.append({"k": k, "kp28": kp28, "base_z": zc,
                            **{a: b for a, b in m.items() if a != "joints"}})

    p = os.path.join(OUT_DIR, args_cli.out)
    with open(p, "w") as fh:
        json.dump({"standing_pelvis_z": STANDING_PELVIS_Z,
                   "bent_k_0.20_standing_z": BENT_K_STANDING_Z,
                   "kd_design": {"zeta": DAMP_ZETA, "J_eff": J_EFF_EST},
                   "kinematics": kin_rows,
                   "airborne_sweep": rows,
                   "tau_req": tau_req,
                   "contact_check": contact}, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {p}")
    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
