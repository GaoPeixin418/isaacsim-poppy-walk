#!/usr/bin/env python3
"""D3 前置实验 ③：Poppy 腿部运动学识别（轴/方向/镜像关系）—— 补上跳过的一步

上一版为什么失败（写下来避免重犯）
----------------------------------
`d3_rest_pose_probe.py` 直接上动力学：把姿态保持住、从 z=1.0 m 放开基座，
结果直腿姿态也倒了（基座 0.063 m、倾角 89.5°）。

错在**跳过了运动学识别**。三个具体的错：
  1. 从 1 m 高放开 → 落地速度 3.5 m/s，**好姿态也会被砸倒**。我没先量出站立高度。
  2. 假设"全零姿态 = 直腿直立"。Poppy 的 Dynamixel 零位完全可能是折腿姿态 ——
     从没验证过。
  3. 假设髋/膝/踝三根轴的"正方向"一致。限位表已经提示左右腿是镜像坐标系，
     但**镜像只说明了左右关系，没说明哪个符号是"前"**。

正确顺序：**先运动学（几何是什么）→ 再动力学（能不能站住）**。

本脚本做两件事
--------------
**Part 1 · 运动学识别**（准静态，每步重置基座 → 重力只造成 0.12 mm 误差，可忽略）

  a) 零位体位表：各刚体相对 pelvis 的位置 + 各刚体质量
  b) **数值雅可比**：每个关节单独转+0.3 rad，看末端（脚）往哪走。
     这就是"这根轴的正方向指向哪"的直接答案 —— 比翻 URDF 猜约定可靠得多。
  c) 左右镜像验证：r_某关节 = -l_某关节 是否真的给出对称形变

**Part 2 · 释放高度扫描**
  对若干候选姿态，扫释放高度，找"站立高度的平台区" ——
  这才是正确的落地高度，避免 3.5 m/s 的砸地。

用法
----
    python scripts/d3_pose_ident.py --part kin      --headless
    python scripts/d3_pose_ident.py --part height   --headless
"""

from __future__ import annotations

import argparse
import json
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Poppy 腿部运动学识别")
parser.add_argument("--part", type=str, default="kin", choices=["kin", "height"])
parser.add_argument("--amp", type=float, default=0.3, help="雅可比用的关节扰动量（rad）")
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
# 常量（与 d3_rest_pose_probe.py 保持一致，避免两份脚本用不同增益做实验）
# ---------------------------------------------------------------------------

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]

#: 关节限位（rad），来自 D3 探针实验①的逐项核对
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

KP_HIP_Y, KD_HIP_Y = 20.0, 0.6
KP_OTHER, KD_OTHER = 8.0, 0.3
EFFORT_MX28, EFFORT_MX64 = 2.5, 6.0

CTRL_DT = 0.02
PHYS_DT = 0.005
DECIMATION = int(round(CTRL_DT / PHYS_DT))

DEVICE = args_cli.device or "cuda:0"


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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5), joint_pos={".*": 0.0}),
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


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def pose_to_tensor(robot: Articulation, pose: dict[str, float]) -> torch.Tensor:
    vals = torch.zeros(len(robot.joint_names), device=DEVICE)
    for i, name in enumerate(robot.joint_names):
        vals[i] = float(pose.get(name, 0.0))
    return vals.unsqueeze(0).repeat(robot.num_instances, 1)


def reset_robot(robot: Articulation, pose_t: torch.Tensor, base_z: float) -> None:
    n = robot.num_instances
    root_pose = torch.zeros(n, 7, device=DEVICE)
    root_pose[:, 3] = 1.0
    root_pose[:, 2] = base_z
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))
    robot.write_joint_state_to_sim(pose_t, torch.zeros_like(pose_t))


def measure_kinematics(robot: Articulation, sim: SimulationContext,
                       pose_t: torch.Tensor, settle_steps: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """准静态测量：每步把基座位姿与关节状态重置到目标值，再步进一次。

    为什么可以这样做：每步只让重力作用 5 ms，基座位移 ≈ ½g·dt² = 0.12 mm，
    对"看脚往哪个方向走"这个量级的问题完全可忽略。
    好处是**不用切换重力**（PhysX 的重力在启动时就写进 USD 了，运行中改不可靠）。

    返回 (相对位置 (num_bodies,3), 刚体姿态四元数 (num_bodies,4))，位置以 pelvis 为原点。
    """
    for _ in range(settle_steps):
        reset_robot(robot, pose_t, base_z=0.5)
        robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)
    # 最后一次重置后只步进一次，尽量让读数接近纯几何解
    reset_robot(robot, pose_t, base_z=0.5)
    robot.set_joint_position_target(pose_t)
    sim.step()
    robot.update(PHYS_DT)

    rel_pos = (robot.data.body_pos_w[0] - robot.data.root_pos_w[0])
    return rel_pos, robot.data.body_quat_w[0]


def quat_angle_deg(q1: torch.Tensor, q2: torch.Tensor) -> float:
    """两个四元数之间的夹角（度）—— 用来衡量"脚掌相对零位转了多少"。"""
    d = float(torch.abs(torch.sum(q1 * q2)))
    d = min(1.0, d)
    return float(math.degrees(2.0 * math.acos(d)))


def rel_rot_axis(q_ref: torch.Tensor, q_new: torch.Tensor):
    """求 q_ref → q_new 的相对旋转的【轴】与【角】。

    做法：dq = q_new * conj(q_ref)（世界系下 q_ref 到 q_new 的增量旋转），
    再把 dq 转成轴角。轴方向直接告诉我们是"俯仰(±X)""侧摆(±Y)"还是"偏航(±Z)"。

    【为什么必须测这个】
    URDF 里关节叫 hip_y / knee_y / ankle_y，字段是 axis 加 rpy —— 这层层坐标变换
    靠人读太容易错。而"这根轴到底是俯仰还是侧摆"是搭站立姿态的硬前提，
    所以直接把它测出来。这就是"能测量的不要靠推理"。
    """
    w, x, y, z = (float(q_ref[0]), float(q_ref[1]), float(q_ref[2]), float(q_ref[3]))
    qr_conj = torch.tensor([w, -x, -y, -z], device=q_ref.device)
    dq = torch.tensor([
        float(q_new[0]) * float(qr_conj[0]) - float(q_new[1]) * float(qr_conj[1])
        - float(q_new[2]) * float(qr_conj[2]) - float(q_new[3]) * float(qr_conj[3]),
        float(q_new[0]) * float(qr_conj[1]) + float(q_new[1]) * float(qr_conj[0])
        + float(q_new[2]) * float(qr_conj[3]) - float(q_new[3]) * float(qr_conj[2]),
        float(q_new[0]) * float(qr_conj[2]) - float(q_new[1]) * float(qr_conj[3])
        + float(q_new[2]) * float(qr_conj[0]) + float(q_new[3]) * float(qr_conj[1]),
        float(q_new[0]) * float(qr_conj[3]) + float(q_new[1]) * float(qr_conj[2])
        - float(q_new[2]) * float(qr_conj[1]) + float(q_new[3]) * float(qr_conj[0]),
    ], device=q_ref.device)
    dw = float(dq[0])
    if dw < 0:                      # 取短弧，保证角度在 [0, 180°]
        dq = -dq
        dw = -dw
    dw = min(1.0, dw)
    ang = math.degrees(2.0 * math.acos(dw))
    v = dq[1:].cpu().numpy()
    n = float((v ** 2).sum() ** 0.5)
    axis = (v / n).tolist() if n > 1e-6 else [0.0, 0.0, 0.0]
    return axis, ang


def axis_label(axis) -> str:
    """把旋转轴归类成人类可读的标签，例如 "俯仰(绕X)+"。"""
    if axis is None:
        return "-"
    vals = [abs(v) for v in axis]
    if max(vals) < 0.2:
        return "-"
    idx = vals.index(max(vals))
    names = ["俯仰 pitch 绕X", "侧摆 roll  绕Y", "偏航 yaw   绕Z"]
    sign = "+" if axis[idx] > 0 else "-"
    return f"{names[idx]} {sign}"


# ---------------------------------------------------------------------------
# Part 1 · 运动学识别
# ---------------------------------------------------------------------------

def part_kinematics(robot: Articulation, sim: SimulationContext) -> dict:
    body_names = list(robot.body_names)
    idx = {n: i for i, n in enumerate(body_names)}

    print("=" * 96)
    print("Part 1 · 运动学识别")
    print("=" * 96)

    # --- a) 刚体质量表 ---
    print("\n[a] 刚体质量（Poppy 标称整机 3.5 kg，含电池与结构件；URDF 里的质量可能偏低）")
    masses = robot.data.default_mass[0]
    print(f"{'刚体':<16}{'质量 kg':>10}")
    print("-" * 28)
    for i, n in enumerate(body_names):
        print(f"{n:<16}{float(masses[i]):>10.4f}")
    print("-" * 28)
    print(f"{'合计':<16}{float(masses.sum()):>10.4f}   ← 与标称 3.5 kg 的差值是后续力矩估算要修正的地方")

    # --- b) 零位体位表 ---
    zero_pose = {n: 0.0 for n in LEG_JOINTS}
    zero_t = pose_to_tensor(robot, zero_pose)
    rel0, quat0 = measure_kinematics(robot, sim, zero_t)

    print("\n[b] 全零姿态下各刚体相对 pelvis 的位置（世界系）")
    print(f"{'刚体':<16}{'x m':>10}{'y m':>10}{'z m':>10}")
    print("-" * 48)
    for n in body_names:
        p = rel0[idx[n]]
        print(f"{n:<16}{float(p[0]):>10.4f}{float(p[1]):>10.4f}{float(p[2]):>10.4f}")

    foot_z = float(rel0[idx["r_foot"]][2])
    print(f"\n  → 零位时脚部相对 pelvis 的 z = {foot_z:.4f} m")
    print(f"  → 说明：{'全零姿态是【伸展/直腿】姿态' if foot_z < -0.15 else '全零姿态是【折腿】姿态（脚没伸到底）'}")

    # --- c) 数值雅可比：单关节 ±amp，看脚怎么走、脚掌怎么转 ---
    print(f"\n[c] 数值雅可比：单关节转 ±{args_cli.amp:.2f} rad，脚部相对 pelvis 的位置变化 + 脚掌旋转轴")
    print("    位移量判断「这根轴把脚送到哪」，旋转轴判断「这根轴属于俯仰/侧摆/偏航」")
    print(f"{'扰动关节':<12}{'方向':>6}{'Δr_foot x':>11}{'Δr_foot y':>11}{'Δr_foot z':>11}"
          f"{'Δl_foot y':>11}{'右脚旋转轴':>20}{'左脚旋转轴':>20}")
    print("-" * 106)

    jac_rows = []
    for jname in LEG_JOINTS:
        for sgn in (+1.0, -1.0):
            pose = dict(zero_pose)
            pose[jname] = sgn * args_cli.amp
            rel, quat = measure_kinematics(robot, sim, pose_to_tensor(robot, pose))
            d_r = rel[idx["r_foot"]] - rel0[idx["r_foot"]]
            d_l = rel[idx["l_foot"]] - rel0[idx["l_foot"]]
            ax_r, ang_r = rel_rot_axis(quat0[idx["r_foot"]], quat[idx["r_foot"]])
            ax_l, ang_l = rel_rot_axis(quat0[idx["l_foot"]], quat[idx["l_foot"]])
            print(f"{jname:<12}{sgn:>+6.0f}{float(d_r[0]):>11.4f}{float(d_r[1]):>11.4f}{float(d_r[2]):>11.4f}"
                  f"{float(d_l[1]):>11.4f}{axis_label(ax_r) + f' {ang_r:.1f}°':>20}"
                  f"{axis_label(ax_l) + f' {ang_l:.1f}°':>20}")
            jac_rows.append({
                "joint": jname, "sign": sgn,
                "d_r_foot": [float(x) for x in d_r],
                "d_l_foot": [float(x) for x in d_l],
                "r_foot_rot_axis": ax_r, "r_foot_rot_deg": ang_r,
                "l_foot_rot_axis": ax_l, "l_foot_rot_deg": ang_l,
            })

    # --- d) 镜像验证 ---
    print("\n[d] 左右镜像验证：r_某关节 = -l_某关节 是否给出【镜像对称】的形变")
    print("    注意横轴是 X（零位时左右脚分别在 x=±0.0665），所以镜像 = x 相反、y/z 相同")
    print("    （之前这里用错了横轴，把本来正确的镜像误判成不对称 —— 记下来当教训）")
    print(f"{'关节对':<12}{'r 脚位移 (x,y,z)':>30}{'l 脚位移 (x,y,z)':>30}{'镜像误差':>12}")
    print("-" * 86)
    mirror_rows = []
    for suffix in ["hip_x", "hip_z", "hip_y", "knee_y", "ankle_y"]:
        pose = dict(zero_pose)
        pose[f"r_{suffix}"] = args_cli.amp
        pose[f"l_{suffix}"] = -args_cli.amp
        rel, _ = measure_kinematics(robot, sim, pose_to_tensor(robot, pose))
        dr = rel[idx["r_foot"]] - rel0[idx["r_foot"]]
        dl = rel[idx["l_foot"]] - rel0[idx["l_foot"]]
        # 完美镜像：Δx 相反，Δy / Δz 相同
        sym_err = max(abs(float(dr[0]) + float(dl[0])),
                      abs(float(dr[1]) - float(dl[1])),
                      abs(float(dr[2]) - float(dl[2])))
        verdict = "★对称★" if sym_err < 0.005 else f"误差 {sym_err * 1000:.1f} mm"
        print(f"{suffix:<12}({float(dr[0]):+.4f},{float(dr[1]):+.4f},{float(dr[2]):+.4f})"
              f"       ({float(dl[0]):+.4f},{float(dl[1]):+.4f},{float(dl[2]):+.4f})"
              f"{verdict:>12}")
        mirror_rows.append({"joint": suffix, "d_r": [float(x) for x in dr],
                            "d_l": [float(x) for x in dl], "symmetry_err": sym_err})

    result = {
        "part": "kinematics",
        "body_names": body_names,
        "joint_names": list(robot.joint_names),
        "masses": {n: float(masses[i]) for i, n in enumerate(body_names)},
        "total_mass": float(masses.sum()),
        "zero_pose_rel_pos": {n: [float(x) for x in rel0[idx[n]]] for n in body_names},
        "zero_pose_foot_z": foot_z,
        "jacobian": jac_rows,
        "mirror": mirror_rows,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "kinematics.json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {os.path.join(OUT_DIR, 'kinematics.json')}")
    return result


# ---------------------------------------------------------------------------
# Part 2 · 释放高度扫描
# ---------------------------------------------------------------------------

#: 待测姿态：(标签, 屈膝量 k, 髋补偿系数 c, 踝补偿系数 d)
#: c/d 的符号含义：正 = 与同侧膝盖同号方向，负 = 反号方向
HEIGHT_POSES = [
    ("零位(全0)", 0.0, 0.0, 0.0),
    ("屈膝0.3/髋+0.5/踝+0.5", 0.3, 0.5, 0.5),
    ("屈膝0.3/髋-0.5/踝-0.5", 0.3, -0.5, -0.5),
    ("屈膝0.3/髋-0.5/踝+0.5", 0.3, -0.5, 0.5),
    ("屈膝0.3/髋+0.5/踝-0.5", 0.3, 0.5, -0.5),
]

HEIGHT_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.70]


def build_pose(k: float, c: float, d: float) -> dict[str, float]:
    """镜像约定 r_joint = -l_joint。k>0 为屈膝。"""
    pose = {name: 0.0 for name in LEG_JOINTS}
    for side, s in (("r", -1.0), ("l", +1.0)):
        pose[f"{side}_knee_y"] = s * k
        pose[f"{side}_hip_y"] = -s * c * k / 2.0
        pose[f"{side}_ankle_y"] = s * d * k / 2.0
    return pose


def run_free(robot: Articulation, sim: SimulationContext, pose_t: torch.Tensor,
             base_z: float, seconds: float, measure_seconds: float = 0.5) -> dict:
    """自由动力学：复位后放开，返回测量窗口内的统计量。"""
    reset_robot(robot, pose_t, base_z)
    robot.set_joint_position_target(pose_t)

    total = int(seconds / PHYS_DT)
    ms = int(measure_seconds / PHYS_DT)
    start = total - ms
    tilts, hs = [], []
    xy0 = None
    for step in range(total):
        if step % DECIMATION == 0:
            robot.set_joint_position_target(pose_t)
        sim.step()
        robot.update(PHYS_DT)
        if step >= start:
            grav = torch.tensor([0.0, 0.0, -1.0], device=DEVICE).repeat(robot.num_instances, 1)
            pg = quat_rotate_inverse(robot.data.root_quat_w, grav)
            tilts.append(float(torch.rad2deg(torch.acos((-pg[0, 2]).clamp(-1.0, 1.0)))))
            hs.append(float(robot.data.root_pos_w[0, 2]))
            if xy0 is None:
                xy0 = robot.data.root_pos_w[0, :2].clone()
    drift = float(torch.norm(robot.data.root_pos_w[0, :2] - xy0))
    t = torch.tensor(tilts)
    h = torch.tensor(hs)
    return {"tilt_mean": float(t.mean()), "tilt_max": float(t.max()),
            "h_mean": float(h.mean()), "h_min": float(h.min()), "drift": drift}


def part_height(robot: Articulation, sim: SimulationContext) -> dict:
    print("=" * 96)
    print("Part 2 · 释放高度扫描 —— 找站立高度的平台区")
    print("=" * 96)
    print("上一版的错误：从 1.0 m 放开 → 落地速度 3.5 m/s，好姿态也被砸倒。")
    print("正确做法：扫一遍释放高度，站立时基座高度会收敛到一个平台值，那就是真实站立高度。\n")

    all_rows = []
    for label, k, c, d in HEIGHT_POSES:
        pose = build_pose(k, c, d)
        pose_t = pose_to_tensor(robot, pose)
        print(f"\n姿态：{label}")
        print(f"{'释放高度':>9} {'落地高度':>9} {'倾角均值':>9} {'倾角峰值':>9} {'水平漂移':>9}  判定")
        print("-" * 68)
        rows = []
        for h0 in HEIGHT_GRID:
            res = run_free(robot, sim, pose_t, base_z=h0, seconds=2.0, measure_seconds=0.5)
            ok = res["tilt_mean"] < 10.0 and res["h_mean"] > 0.6 * h0
            verdict = "站住" if ok else ("摇晃" if res["tilt_mean"] < 30.0 else "倒了")
            print(f"{h0:>9.2f} {res['h_mean']:>9.4f} {res['tilt_mean']:>8.2f}° "
                  f"{res['tilt_max']:>8.2f}° {res['drift']:>9.4f}  {verdict}")
            rows.append({"release_z": h0, **res, "ok": ok})
        all_rows.append({"label": label, "k": k, "c": c, "d": d, "rows": rows})

    result = {"part": "height_sweep", "grid": HEIGHT_GRID, "results": all_rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "height_sweep.json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {os.path.join(OUT_DIR, 'height_sweep.json')}")
    return result


# ---------------------------------------------------------------------------
# 主流程

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
        dt=PHYS_DT, device=DEVICE, gravity=(0.0, 0.0, -9.81)))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.85))
    light_cfg.func("/World/Light", light_cfg)

    robot = Articulation(build_poppy_cfg())
    sim.reset()

    print(f"关节顺序（USD 输出顺序，和 URDF 书写顺序无关）：{robot.joint_names}")
    print(f"刚体顺序：{robot.body_names}\n")

    if args_cli.part == "kin":
        part_kinematics(robot, sim)
    else:
        part_height(robot, sim)

    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
