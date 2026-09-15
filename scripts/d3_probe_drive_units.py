#!/usr/bin/env python3
"""
D3 前置实验：关节刚度配置值（stiffness）在物理引擎里的实际单位测定

【为什么必须做这个实验】
  Isaac Lab 的 ImplicitActuatorCfg.stiffness 写进 PhysX 时到底是 N·m/rad 还是 N·m/deg？
  两者相差 180/pi = 57.296 倍。Poppy 用 MX-28（堵转约 2.5 N·m），
  这个差别就是"能站住"和"第一次迭代就爆掉"的区别。

  已知事实（已实测，见 A 节）：
    - Isaac Lab 把配置值**原样**写进 PhysX（read-back 得到 8.0 / 0.3，没有换算）
    - 但"原样写入"不等于"N·m/rad" —— 单位只能靠物理响应判定

【测量原理：关节力矩传递 + 静态刚度】
  对基座固定的串联链，施加在末端连杆上的力矩会**完整地**传给链条上所有与该
  力矩轴平行的关节；垂直的关节几乎不承受。稳态时角速度为零、阻尼项不贡献力矩：
        |tau_ext| = kp_eff * |delta_theta_rad|
    →   kp_eff = |tau_ext| / |delta_theta_rad|        [N·m/rad]
  用配置值 S=8 与 S=16 各测一次，看 kp_eff/S：
        ≈ 1.0      → 配置值就是 N·m/rad
        ≈ 57.296   → 配置值被当作 N·m/deg（要除以 57.296）

【上一版踩的两个坑（这版已修）】
  1. Isaac Lab 的外部力矩作用在**刚体自身坐标系**下，不是世界系。
     小腿的局部 Y 轴沿骨头方向，绕它加力矩等于"拧大腿"，
     力矩全被 r_hip_z（髋偏航）承担，膝盖根本没被加载。
     → 这版先自动扫描三个体轴，找出真正能加载膝关节的那个。
  2. settle() 内部重建了关节目标，把外部传入的目标覆盖掉了，
     导致"关节限位验证"实际是让关节停在 0 位测的，结论无效。
     → 这版把 target 作为参数传入。
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import time
import traceback
from collections import deque

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="测定关节刚度配置值在物理引擎中的实际单位")
parser.add_argument("--usd", type=str, required=True, help="机器人 USD 路径")
parser.add_argument("--out", type=str, default="", help="结果 JSON 输出路径（可选）")
parser.add_argument("--stiffness-a", type=float, default=8.0, help="测试刚度值 A")
parser.add_argument("--stiffness-b", type=float, default=16.0, help="测试刚度值 B")
parser.add_argument("--damping", type=float, default=0.3, help="阻尼（与转换时同量级）")
parser.add_argument("--ext-torque", type=float, default=2.0, help="外部力矩大小 [N·m]")
parser.add_argument("--settle-steps", type=int, default=1500, help="每个工况最大步数")
parser.add_argument("--avg-steps", type=int, default=150, help="尾部平均步数")
parser.add_argument("--tol", type=float, default=1e-5, help="收敛判据：每 100 步最大位移变化 [rad]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

DEVICE = "cuda:0"
RAD2DEG = 180.0 / math.pi
REPORT: dict = {}
USE_RENDER_KW = True
BODY_ID = 0


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(title)
        print("=" * 78)
    sys.stdout.flush()


def step_sim(sim) -> None:
    if USE_RENDER_KW:
        sim.step(render=False)
    else:
        sim.step()


def build_sim() -> SimulationContext:
    # 重力关掉：唯一的外载荷就是我们施加的已知力矩，反推时没有别的力矩干扰
    return SimulationContext(SimulationCfg(dt=0.005, device=DEVICE, gravity=(0.0, 0.0, 0.0)))


def build_robot_cfg(stiffness: float, damping: float) -> ArticulationCfg:
    """articulation_props 属于 spawner（UsdFileCfg）的字段；固定基座必须写在 spawn 这层，
    与官方 anymal.py 写法一致。"""
    return ArticulationCfg(
        prim_path="/World/Poppy",
        spawn=sim_utils.UsdFileCfg(
            usd_path=args_cli.usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.2), joint_pos={".*": 0.0}),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=stiffness,
                damping=damping,
                friction=0.0,
            )
        },
    )


def read_gains(robot: Articulation) -> dict:
    """读回 PhysX 里实际生效的增益（v2.1.0 的接口名）。"""
    view = robot.root_physx_view
    out: dict = {}
    for key, name in (("stiffness", "get_dof_stiffnesses"), ("damping", "get_dof_dampings")):
        fn = getattr(view, name, None)
        if fn is None:
            out[key] = f"{name} 不存在"
            continue
        try:
            out[key] = [round(v, 6) for v in fn()[0].tolist()]
        except Exception as exc:
            out[key] = f"{name}() 失败: {exc}"
    return out


def set_uniform_gains(robot: Articulation, stiffness: float, damping: float) -> None:
    """不同版本的 write_joint_* 接口签名不一样，这里运行时适配。"""
    n = len(robot.joint_names)
    st = torch.full((1, n), float(stiffness), device=DEVICE)
    dp = torch.full((1, n), float(damping), device=DEVICE)
    sig = inspect.signature(robot.write_joint_stiffness_to_sim)
    if "damping" in sig.parameters:
        robot.write_joint_stiffness_to_sim(stiffness=st, damping=dp)
        return
    robot.write_joint_stiffness_to_sim(st)
    write_damp = getattr(robot, "write_joint_damping_to_sim", None)
    if write_damp is not None:
        write_damp(dp)


def zero_target(robot) -> torch.Tensor:
    return torch.zeros((robot.num_instances, len(robot.joint_names)), device=DEVICE)


def settle(robot, sim, target, steps_cap, torque_vec=None, avg_steps=150, tol=1e-5, label=""):
    """推进仿真到稳态，返回（尾部若干步的平均关节位置, 实际步数）。

    target     : 关节位置目标，形状 (num_instances, num_joints)，由调用方提供
    torque_vec : 施加在小腿上的力矩（刚体坐标系），None 表示不加外载荷。
                 注意：外部力/力矩缓冲会一直保留到被覆盖，所以每步都显式写入，
                 否则上一个工况的载荷会串进下一个工况。
    """
    forces = torch.zeros((robot.num_instances, 1, 3), device=DEVICE)
    torques = torch.zeros((robot.num_instances, 1, 3), device=DEVICE)
    if torque_vec is not None:
        torques[0, 0, :] = torch.tensor(torque_vec, device=DEVICE, dtype=torques.dtype)

    tail = deque(maxlen=avg_steps)
    prev = None
    settled_at = steps_cap
    t0 = time.time()
    for i in range(steps_cap):
        robot.set_joint_position_target(target)
        robot.set_external_force_and_torque(forces, torques, body_ids=[BODY_ID])
        robot.write_data_to_sim()
        step_sim(sim)
        robot.update(sim.get_physics_dt())
        tail.append(robot.data.joint_pos[0].clone())
        if (i + 1) % 100 == 0:
            cur = tail[-1]
            if prev is not None and (cur - prev).abs().max().item() < tol:
                settled_at = i + 1
                break
            prev = cur.clone()
    elapsed = time.time() - t0
    if label:
        print(f"    {label}: {settled_at}/{steps_cap} 步, {elapsed:.1f}s", flush=True)
    return torch.stack(list(tail), dim=0).mean(dim=0), settled_at


def delta_after_load(robot, sim, stiffness, torque_vec, label):
    """统一刚度 → 无载荷基线 → 施加已知力矩 → 返回各关节稳态偏角。"""
    set_uniform_gains(robot, stiffness, args_cli.damping)
    z = zero_target(robot)
    base, _ = settle(robot, sim, z, args_cli.settle_steps, None,
                     args_cli.avg_steps, args_cli.tol, f"{label} 基线")
    res, _ = settle(robot, sim, z, args_cli.settle_steps, torque_vec,
                    args_cli.avg_steps, args_cli.tol, f"{label} 加载")
    return res - base


def show_delta(robot, delta, torque_vec, stiffness, title):
    print(f"  [{title}] 力矩 {torque_vec} N·m · 配置 stiffness={stiffness}")
    tmag = math.sqrt(sum(v * v for v in torque_vec))
    order = sorted(range(len(robot.joint_names)), key=lambda i: -abs(delta[i].item()))
    for i in order:
        nm = robot.joint_names[i]
        dv = delta[i].item()
        if abs(dv) < 5e-6:
            continue
        kp = tmag / abs(dv)
        print(f"      {nm:<12} Δθ = {dv:+.6f} rad ({dv * RAD2DEG:+8.4f} deg)"
              f"   kp_eff(若该关节独自承载) = {kp:9.3f}   kp_eff/S = {kp / stiffness:8.3f}")
    return order


def main() -> None:
    global BODY_ID, USE_RENDER_KW
    hr("D3 探针实验：关节刚度实际单位测定")
    print(f"USD          : {args_cli.usd}")
    print(f"测试刚度     : A={args_cli.stiffness_a}  B={args_cli.stiffness_b}")
    print(f"阻尼         : {args_cli.damping}")
    print(f"外部力矩     : {args_cli.ext_torque} N·m")
    print(f"rad->deg     : {RAD2DEG:.4f}")
    print(f"torch        : {torch.__version__}   cuda={torch.cuda.is_available()}")

    sim = build_sim()
    robot = Articulation(build_robot_cfg(args_cli.stiffness_a, args_cli.damping))
    sim.reset()
    robot.reset()
    USE_RENDER_KW = "render" in inspect.signature(sim.step).parameters
    print(f"sim.step 支持 render 参数: {USE_RENDER_KW}")

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    REPORT["joint_names"] = joint_names
    REPORT["body_names"] = body_names
    print(f"\n关节数 {len(joint_names)}: {joint_names}")
    print(f"刚体数 {len(body_names)}: {body_names}")

    lim = robot.data.joint_pos_limits[0].cpu()
    print("\n关节限位 [rad]：")
    report_limits = {}
    for i, nm in enumerate(joint_names):
        lo, hi = lim[i, 0].item(), lim[i, 1].item()
        print(f"    {nm:<12} [{lo:+.3f}, {hi:+.3f}]")
        report_limits[nm] = [lo, hi]
    REPORT["joint_limits"] = report_limits

    hr("A. 配置路径写回的增益（read-back）")
    gains = read_gains(robot)
    for k, v in gains.items():
        print(f"  {k:<10}: {v}")
    REPORT["readback"] = gains
    print("\n  解读：PhysX 里真实生效的数值与配置值一致 → Isaac Lab 原样写入，没有换算。")
    print("        但单位仍要靠下面的物理响应判定。")

    test_joint = next((c for c in ("r_knee_y", "l_knee_y") if c in joint_names), None)
    if test_joint is None:
        test_joint = next((nm for nm in joint_names if "knee" in nm), joint_names[0])
    tj = joint_names.index(test_joint)
    hip_joint = "r_hip_y" if "r_hip_y" in joint_names else joint_names[0]
    hj = joint_names.index(hip_joint)

    test_body = next((c for c in ("r_shin", "l_shin") if c in body_names), None)
    if test_body is None:
        test_body = next((nm for nm in body_names if "shin" in nm), body_names[-1])
    BODY_ID = body_names.index(test_body)
    print(f"\n待测关节: {test_joint}(膝) / {hip_joint}(髋，同轴参照)   施力刚体: {test_body}")

    # ---- B. 自动扫描三个体轴，找出真正能加载膝关节的那个 ----
    hr("B. 力矩轴扫描（外部力矩作用在刚体坐标系，必须先确定哪个轴是膝关节轴）")
    mag = args_cli.ext_torque
    axis_resp = {}
    for ax, axname in ((0, "X"), (1, "Y"), (2, "Z")):
        tv = [0.0, 0.0, 0.0]
        tv[ax] = mag
        d = delta_after_load(robot, sim, args_cli.stiffness_a, tv, f"轴 {axname} +")
        show_delta(robot, d, tv, args_cli.stiffness_a, f"轴 {axname} 正向")
        axis_resp[axname] = {"knee": d[tj].item(), "hip_y": d[hj].item()}
    REPORT["axis_scan"] = axis_resp

    best_axis_name = max(axis_resp, key=lambda k: abs(axis_resp[k]["knee"]))
    best_axis = {"X": 0, "Y": 1, "Z": 2}[best_axis_name]
    print(f"\n  ⇒ 膝关节响应最大的轴: {best_axis_name}"
          f"（Δθ_knee = {axis_resp[best_axis_name]['knee']:+.6f} rad）")

    # ---- C. 定符号：膝盖只能单向弯，选让膝盖真正动起来的那个方向 ----
    hr("C. 符号判定（膝盖单向限位，必须先找出不被限位挡住的方向）")
    signs = {}
    for sign in (+1.0, -1.0):
        tv = [0.0, 0.0, 0.0]
        tv[best_axis] = sign * mag
        d = delta_after_load(robot, sim, args_cli.stiffness_a, tv, f"{sign:+.0f}")
        show_delta(robot, d, tv, args_cli.stiffness_a, f"方向 {sign:+.0f}")
        signs[f"{sign:+.0f}"] = {"knee": d[tj].item(), "hip_y": d[hj].item()}
    REPORT["sign_scan"] = signs
    best_sign = max(signs, key=lambda k: abs(signs[k]["knee"]))
    print(f"\n  ⇒ 膝盖响应更大的方向: {best_sign}（限位没挡住的那一侧）")

    torque_vec = [0.0, 0.0, 0.0]
    torque_vec[best_axis] = float(best_sign) * mag
    print(f"  最终载荷向量（刚体坐标系）: {torque_vec}")

    # ---- D. 用两个不同刚度值测绝对值，两个关节互相印证 ----
    hr("D. 刚度绝对值测定（膝关节轴 + 自由方向）")
    measures = []
    for s in (args_cli.stiffness_a, args_cli.stiffness_b):
        d = delta_after_load(robot, sim, s, torque_vec, f"S={s}")
        show_delta(robot, d, torque_vec, s, f"S={s}")
        measures.append({"stiffness": s, "knee": d[tj].item(), "hip_y": d[hj].item()})
    REPORT["measures"] = measures

    hr("E. 结论：换算系数")
    ratios = []
    for m in measures:
        for key in ("knee", "hip_y"):
            dv = m[key]
            if abs(dv) > 1e-7:
                ratios.append((mag / abs(dv)) / m["stiffness"])
    if ratios:
        avg = sum(ratios) / len(ratios)
        print(f"各载荷关节给出的 kp_eff/S：[{', '.join(f'{r:.4f}' for r in ratios)}]")
        print(f"平均换算系数 c = {avg:.4f}\n")
        if abs(avg - 1.0) < 0.25:
            verdict = "N·m/rad"
            print("⇒ 结论：配置值就是 N·m/rad。可以直接按物理直觉填，")
            print("   例如想要 8 N·m/rad 就写 stiffness=8.0。")
        elif abs(avg - RAD2DEG) / RAD2DEG < 0.25:
            verdict = "N·m/deg"
            print("⇒ 结论：配置值被当作 N·m/deg（比物理直觉硬 57 倍）。")
            print(f"   要得到目标刚度，配置里写  目标kp / {RAD2DEG:.3f}")
            print(f"   例如想要 8 N·m/rad → stiffness = {8 / RAD2DEG:.6f}")
        else:
            verdict = f"未定（c = {avg:.4f}）"
            print("⇒ c 既不接近 1 也不接近 57.296，需要继续排查：")
            print("   膝与髋给出的一致吗？若两者不一致，说明力矩分配假设不成立。")
        REPORT["conversion_factor"] = avg
        REPORT["verdict"] = verdict
    else:
        REPORT["verdict"] = "无有效测量"
        print("没有有效测量值 —— 关节在被加载方向上几乎没动")

    # ---- F. 反关节可行性验证（膝盖单向限位是否在物理上真正生效） ----
    hr("F. 反关节可行性验证")
    set_uniform_gains(robot, args_cli.stiffness_a, args_cli.damping)
    report_limit = {}
    for sweep in (+0.5, -0.5):
        tgt = zero_target(robot)
        tgt[0, tj] = sweep
        res, _ = settle(robot, sim, tgt, 800, None, 200, args_cli.tol, f"目标 {sweep:+.3f}")
        got = res[tj].item()
        print(f"  目标 {test_joint} = {sweep:+.3f} rad ({sweep * RAD2DEG:+.1f} deg)"
              f"  →  实际 {got:+.4f} rad ({got * RAD2DEG:+.2f} deg)")
        report_limit[f"{sweep:+.3f}"] = got
    REPORT["limit_check"] = report_limit
    print("\n  解读：膝盖限位是 [-2.339, +0.061] rad。若 +0.5 的目标被卡在约 +0.06 rad，")
    print("        说明限位在物理层面真的生效 —— '反着弯膝盖'这条作弊解被封死。")

    hr("实验结束")
    if args_cli.out:
        os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
        with open(args_cli.out, "w", encoding="utf-8") as f:
            json.dump(REPORT, f, ensure_ascii=False, indent=2)
        print(f"结果已写入 {args_cli.out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        hr("!! 实验异常 !!")
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
