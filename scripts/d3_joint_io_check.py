#!/usr/bin/env python3
"""关节写入/读取通路自检 —— 一个决定性的最小实验。

================================================================================
为什么写这个脚本
================================================================================
上一轮标定出现了一个在物理上不可能的结果：
    膝盖指令 -0.2 rad，无论 kp 取 6 还是 100，实测"误差"恒为 0.20000 rad，
    关节速度恒为 0.00000。

恒定的误差 + 零速度 + 对增益完全不敏感 —— 这三件事放在一起只有一个解释：
**关节没有停在目标角，而是停在 0 位**（于是 |误差| = |指令| = 0.2）。

那到底是"指令没写进去"、"驱动器没出力"、还是"我读回来的数不对"？
这三种情况的修法完全不同，继续猜下去纯属浪费算力。所以做一个最小的通路测试，
每一段只验证一件事：

  Part A  写：给每个关节写一个互不相同的值，读回来 → 验证写/读的**关节顺序**一致
  Part B  读：目标值是否真的进了 Isaac Lab 的 joint_pos_target 缓冲
  Part C  驱动：给一个远离平衡的目标，看关节是否朝目标移动（区分"不动/掉到限位/跟到目标"）
  Part D  USD：直接读 USD 里的 drive:angular:physics:stiffness，
          确认 write_joint_stiffness_to_sim 到底把什么数写进了哪里
          —— 这条是回答"三层单位约定"那个老坑的最直接办法

================================================================================
每一步的判读表（先写清楚判据，再看数字，避免自欺欺人）
================================================================================
  A 读回值 == 写入值                → 顺序与读写通路正常
  A 读回值 == 全 0                  → 写入没生效（或读的是未刷新的缓冲）
  B joint_pos_target == 设定值      → 目标进了控制缓冲
  B joint_pos_target == 0           → set_joint_position_target 写错了关节
  C 关节角 → 目标（误差≈0 或 ≤ 1e-2）→ 驱动器工作正常，问题在别处
  C 关节角 → 0                      → 驱动器没出力
  C 关节角 → 限位（膝盖 -2.34）      → 驱动器刚度≈0，重力把关节压到底
  D drive stiffness == 我写的 kp    → write_joint_stiffness_to_sim 单位即配置单位
  D drive stiffness == kp×57.296    → 被按"度"解释，需要除以 180/π
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="关节读写通路自检")
parser.add_argument("--out", type=str, default="joint_io_check.json")
parser.add_argument("--kp", type=float, default=20.0)
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
STANDING_PELVIS_Z = 0.42125
LIFT_Z = 0.03
KP = args_cli.kp


def kd_for(kp):
    return 2.0 * math.sqrt(kp * 0.02)


def build_cfg(kp28):
    kp_hip = kp28 * KG_RATIO
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


def write_root(robot, z):
    n = robot.num_instances
    rp = torch.zeros(n, 7, device=DEVICE)
    rp[:, 3] = 1.0
    rp[:, 2] = z
    robot.write_root_pose_to_sim(rp)
    robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=DEVICE))


def run(robot, sim, base_z, steps, *, joint_target=None):
    for _ in range(steps):
        write_root(robot, base_z)
        if joint_target is not None:
            robot.set_joint_position_target(joint_target)
        sim.step()
        robot.update(PHYS_DT)


def finish():
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE,
                                                    gravity=(0.0, 0.0, -9.81)))
    g = sim_utils.GroundPlaneCfg()
    g.func("/World/ground", g)

    robot = Articulation(build_cfg(KP))
    sim.reset()

    names = list(robot.joint_names)
    J = len(names)
    out = {"joint_names": names, "kp_requested": KP}

    print("=" * 96)
    print("关节读写通路自检")
    print("=" * 96)
    print(f"joint_names ({J} 个):")
    for i, n in enumerate(names):
        print(f"  [{i}] {n}")
    print(f"body_names ({len(robot.body_names)} 个): {list(robot.body_names)}")

    # ================= Part A · 写入 → 读回 =================
    print()
    print("Part A · 写入→读回（给每个关节写互不相同的值，验证顺序）")
    print("-" * 96)
    # 值都在限位内：|0.05| < knee 的 0.0611
    # 注意容差：写入后必须 step 一次让 PhysX 刷新缓冲，而这一步里默认驱动器
    # 会把关节往 joint_pos_target(=0) 拉一点。5 ms 内的漂移量级 ~6e-4 rad，
    # 所以容差取 2e-3，既能判顺序又不会被这点漂移误判。
    pattern = torch.zeros(1, J, device=DEVICE)
    for i in range(J):
        pattern[0, i] = 0.05 if i % 2 == 0 else -0.04
    robot.write_joint_state_to_sim(pattern, torch.zeros_like(pattern))
    sim.step()
    robot.update(PHYS_DT)
    q_after_write = robot.data.joint_pos[0].clone()
    print(f"  {'关节':>11}{'写入值':>11}{'读回值':>11}{'一致?':>8}")
    a_ok = True
    for i, n in enumerate(names):
        same = abs(float(q_after_write[i]) - float(pattern[0, i])) < 2e-3
        a_ok &= same
        print(f"  {n:>11}{float(pattern[0, i]):>11.4f}{float(q_after_write[i]):>11.4f}"
              f"{('✓' if same else '✗'):>8}")
    print(f"  → Part A 判定：{'顺序与读写通路正常' if a_ok else '★写入或读取有问题★'}")
    out["part_a_write_read_ok"] = a_ok

    # ================= Part B · 目标缓冲 =================
    print()
    print("Part B · set_joint_position_target 是否进了 goal 缓冲")
    print("-" * 96)
    tgt = torch.zeros(1, J, device=DEVICE)
    for i, n in enumerate(names):
        if n.endswith("knee_y"):
            tgt[0, i] = -0.5 if n.startswith("r") else 0.5
        elif n.endswith("hip_y"):
            tgt[0, i] = 0.2 if n.startswith("r") else -0.2
    robot.set_joint_position_target(tgt)
    sim.step()
    robot.update(PHYS_DT)
    stored = None
    for attr in ("joint_pos_target", "joint_pos_target_sim"):
        v = getattr(robot.data, attr, None)
        if v is not None:
            stored = (attr, v[0].clone())
            break
    print(f"  {'关节':>11}{'设定目标':>11}{'缓冲里的值':>13}{'一致?':>8}")
    b_ok = True
    if stored is None:
        print("  ★ robot.data 里没有 joint_pos_target 缓冲，无法核对 ★")
        b_ok = False
    else:
        print(f"  （读取自 robot.data.{stored[0]}）")
        for i, n in enumerate(names):
            same = abs(float(stored[1][i]) - float(tgt[0, i])) < 1e-5
            b_ok &= same
            print(f"  {n:>11}{float(tgt[0, i]):>11.4f}{float(stored[1][i]):>13.4f}"
                  f"{('✓' if same else '✗'):>8}")
    print(f"  → Part B 判定：{'目标已正确进入控制缓冲' if b_ok else '★目标没进缓冲★'}")
    out["part_b_target_buffer_ok"] = b_ok

    # ================= Part C · 驱动器是否出力 =================
    print()
    print(f"Part C · 驱动器出力测试（基座抬高 {LIFT_Z} m 悬空，kp(MX-28)={KP}）")
    print("-" * 96)
    print("  判据：")
    print("    关节角 ≈ 目标（误差 ≤ 0.02）        → 驱动器正常")
    print("    关节角 ≈ 0                         → 驱动器不出力")
    print("    膝盖冲到 -2.34（限位）              → 刚度≈0，被重力压到底")
    print()
    robot.write_joint_state_to_sim(torch.zeros(1, J, device=DEVICE),
                                   torch.zeros(1, J, device=DEVICE))
    run(robot, sim, STANDING_PELVIS_Z + LIFT_Z, 60)          # 先让它稳定在零位
    # 带轨迹地跑，这样能区分「朝目标移动」和「卡住不动」
    idx_rk = names.index("r_knee_y")
    idx_ra = names.index("r_ankle_y")
    trace = []
    for s in range(360):
        write_root(robot, STANDING_PELVIS_Z + LIFT_Z)
        robot.set_joint_position_target(tgt)
        sim.step()
        robot.update(PHYS_DT)
        if s % 30 == 0:
            trace.append((s * PHYS_DT,
                          float(robot.data.joint_pos[0][idx_rk]),
                          float(robot.data.joint_pos[0][idx_ra])))
    print("  轨迹（r_knee_y 目标 %.2f，r_ankle_y 目标 %.2f）：" % (float(tgt[0, idx_rk]), float(tgt[0, idx_ra])))
    print("    " + " | ".join(f"t={t:.2f} knee={k:+.4f} ankle={a:+.4f}" for t, k, a in trace))
    q_c = robot.data.joint_pos[0].clone()
    v_c = robot.data.joint_vel[0].clone()
    err_c = (q_c - tgt[0]).abs()
    print(f"  {'关节':>11}{'目标':>10}{'实际':>10}{'误差':>10}{'速度':>10}"
          f"{'限位':>18}  判读")
    for i, n in enumerate(names):
        lo = float(robot.data.joint_pos_limits[0][i][0])
        hi = float(robot.data.joint_pos_limits[0][i][1])
        e = float(err_c[i])
        q = float(q_c[i])
        at_lim = min(abs(q - lo), abs(q - hi)) < 2e-3
        if e <= 0.02:
            verdict = "跟到目标 ✓"
        elif at_lim:
            verdict = "★停在限位（刚度≈0）★"
        elif abs(q) < 0.02:
            verdict = "★不动（驱动器没出力）★"
        else:
            verdict = "部分跟踪"
        print(f"  {n:>11}{float(tgt[0, i]):>10.4f}{q:>10.4f}{e:>10.4f}{float(v_c[i]):>10.4f}"
              f"{'[%+.3f,%+.3f]' % (lo, hi):>18}  {verdict}")
    print(f"  → Part C 判定：最大误差 {float(err_c.max()):.5f} rad，"
          f"最大速度 {float(v_c.abs().max()):.5f} rad/s")
    out["part_c"] = {"err_max": float(err_c.max()),
                     "vel_max": float(v_c.abs().max()),
                     "target": tgt[0].tolist(), "actual": q_c.tolist()}

    # ================= Part D · USD 里的实际刚度 =================
    print()
    print("Part D · USD 里的 drive:angular:physics:stiffness 实测值")
    print("-" * 96)
    print("  先通过 write_joint_stiffness_to_sim 写入一组已知值，再去 USD 里读回来。")
    print("  这样能直接回答：这个 API 的参数单位和配置里的 stiffness 单位是否一致。")
    print()
    stiff = torch.zeros(1, J, device=DEVICE)
    damp = torch.zeros(1, J, device=DEVICE)
    for i, n in enumerate(names):
        kp = 137.0 if n.endswith("knee_y") else 13.0
        stiff[0, i] = kp
        damp[0, i] = 1.0
    robot.write_joint_stiffness_to_sim(stiff)
    robot.write_joint_damping_to_sim(damp)
    sim.step()
    robot.update(PHYS_DT)

    import omni.usd  # noqa: E402
    from pxr import UsdPhysics  # noqa: E402

    stage = omni.usd.get_context().get_stage()
    usd_rows = []
    print(f"  {'USD 关节prim':>52}{'stiffness':>12}{'damping':>10}{'写入kp':>10}"
          f"{'比值':>10}  判读")
    for prim in stage.Traverse():
        if prim.GetTypeName() != "PhysicsRevoluteJoint":
            continue
        s_attr = prim.GetAttribute("drive:angular:physics:stiffness")
        d_attr = prim.GetAttribute("drive:angular:physics:damping")
        if not s_attr or not s_attr.IsValid():
            continue
        sv = s_attr.Get()
        dv = d_attr.Get() if d_attr and d_attr.IsValid() else None
        leaf = prim.GetName()
        matched = next((n for n in names if n in leaf or leaf in n), None)
        kp_written = None
        if matched:
            kp_written = 137.0 if matched.endswith("knee_y") else 13.0
        ratio = (float(sv) / kp_written) if (sv is not None and kp_written) else float("nan")
        judge = ""
        if kp_written:
            if abs(ratio - 1.0) < 0.02:
                judge = "与写入值相同（单位与配置一致）"
            elif abs(ratio - 180.0 / math.pi) < 0.5:
                judge = "★= kp×57.296 → 被按'度'解释★"
            elif abs(ratio - math.pi / 180.0) < 0.001:
                judge = "★= kp/57.296 → 被按'弧度'解释★"
            else:
                judge = f"比值 {ratio:.4f} 需人工判断"
        print(f"  {prim.GetPath().pathString:>52}{float(sv):>12.4f}"
              f"{(float(dv) if dv is not None else float('nan')):>10.4f}"
              f"{(kp_written if kp_written else float('nan')):>10.1f}{ratio:>10.4f}  {judge}")
        usd_rows.append({"prim": prim.GetPath().pathString, "name": leaf,
                         "usd_stiffness": float(sv),
                         "usd_damping": float(dv) if dv is not None else None,
                         "kp_written": kp_written, "ratio": ratio})

    out["usd_drive"] = usd_rows

    p = os.path.join(OUT_DIR, args_cli.out)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {p}")
    print("\n完成。")
    finish()


if __name__ == "__main__":
    main()
