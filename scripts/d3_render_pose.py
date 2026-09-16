#!/usr/bin/env python3
"""D3 前置实验 ⑤：渲染 + 动态限位验证 —— 一眼定案"哪边是前""膝盖往哪弯"

前面几个实验留下的待定问题
--------------------------
1. **朝向**：运动学识别说"髋 +方向把脚送到 -Y"、"膝 +方向也把脚送到 -Y"。
   但到底 -Y 是前还是后？必须先定这个，否则"膝盖该往哪弯"无从判断。
2. **限位符号**：USD 里左右膝盖的关节轴在世界系里是【平行】的（点积 +1.0），
   而限位却是镜像的（右 [-134°,+3.5°] / 左 [-3.5°,+134°]）。
   这两者不可能同时成立 —— 需要动态实验确认到底哪个方向被卡住。
3. **落地高度**：上一版从 1.0 m 放开（落地速度 3.5 m/s）把好姿态也砸倒了。
   运动学识别给出脚在 pelvis 下方 0.386 m，所以正确的释放高度是 ~0.42 m。

本脚本一次跑完：
  · 每个测试姿态：先落地稳定，**打印"指令关节角 vs 实际关节角"**（这就是限位是否生效的直接证据）
  · 再从 3 个视角渲染 PNG（正面 / 侧面 / 斜俯视）
  · 图片存到 out/d3_rest_pose/，回传本机目视确认

用法
----
    python scripts/d3_render_pose.py --headless --enable_cameras
    python scripts/d3_render_pose.py --base_z 0.42 --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import zlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Poppy 姿态渲染 / 动态限位验证")
parser.add_argument("--base_z", type=float, default=0.42, help="释放高度（m）")
parser.add_argument("--settle", type=float, default=1.5, help="落地稳定时长（s）")
parser.add_argument("--tag", type=str, default="probe", help="输出文件前缀")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_rotate_inverse  # noqa: E402

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

PHYS_DT = 0.005
DECIMATION = 4
DEVICE = args_cli.device or "cuda:0"

KP_HIP_Y, KD_HIP_Y = 20.0, 0.6
KP_OTHER, KD_OTHER = 8.0, 0.3


# ---------------------------------------------------------------------------
# 待测姿态
# ---------------------------------------------------------------------------
def make(overrides: dict) -> dict:
    base = {n: 0.0 for n in [
        "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
        "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y"]}
    base.update(overrides)
    return base


TEST_CASES = [
    ("zero", "全零姿态（验证零位是否就是直立）", make({})),
    ("knee_same_sign", "双膝同符号 -0.6（同符号=同世界方向，若左膝被卡住说明限位在起作用）",
     make({"r_knee_y": -0.6, "l_knee_y": -0.6})),
    ("knee_mirror_sign", "双膝反符号（r=-0.6, l=+0.6）",
     make({"r_knee_y": -0.6, "l_knee_y": +0.6})),
    ("knee_small", "双膝同符号 -0.25（小幅屈膝，若成立即为可用的 rest pose）",
     make({"r_knee_y": -0.25, "l_knee_y": -0.25,
           "r_hip_y": +0.125, "l_hip_y": -0.125,
           "r_ankle_y": -0.125, "l_ankle_y": +0.125})),
]


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
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.42), joint_pos={".*": 0.0}),
        actuators={
            "hip_pitch": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_y"],
                effort_limit=6.0, effort_limit_sim=6.0,
                velocity_limit_sim=8.2, stiffness=KP_HIP_Y, damping=KD_HIP_Y,
            ),
            "rest": ImplicitActuatorCfg(
                joint_names_expr=[".*hip_x", ".*hip_z", ".*knee_y", ".*ankle_y"],
                effort_limit=2.5, effort_limit_sim=2.5,
                velocity_limit_sim=7.0, stiffness=KP_OTHER, damping=KD_OTHER,
            ),
        },
    )


def pose_to_tensor(robot, pose, device=DEVICE):
    vals = torch.zeros(len(robot.joint_names), device=device)
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
# 相机 / PNG
# ---------------------------------------------------------------------------
def _norm(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-9 else list(v)


def _sub(a, b):
    return [x - y for x, y in zip(a, b)]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)):
    """convention="world"：相机前向 = +X，上向 = +Z。三列分别是相机 x/y/z 轴在世界中的方向。"""
    f = _norm(_sub(target, pos))
    u = _norm(up)
    z = _norm(_sub(u, [f[0] * _dot(u, f), f[1] * _dot(u, f), f[2] * _dot(u, f)]))
    y = _cross(z, f)
    m = [[f[0], y[0], z[0]], [f[1], y[1], z[1]], [f[2], y[2], z[2]]]
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        q = ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        q = ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        q = ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)
    return tuple(float(x) for x in q)


def write_png(path, arr):
    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    blob = b"\x89PNG\r\n\x1a\n"
    blob += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
    blob += chunk(b"IDAT", zlib.compress(raw, 6))
    blob += chunk(b"IEND", b"")
    with open(path, "wb") as fh:
        fh.write(blob)


def make_cameras(base_z: float):
    """建 3 个固定机位。注意 convention="world" 下相机前向是 +X、上向是 +Z。"""
    target = (0.0, 0.0, base_z * 0.5)
    d = 1.15
    views = {
        "front": (0.0, -d, base_z * 0.85),        # 站在 -Y 侧看 → 若机器人朝 -Y，这是正面
        "side": (-d, 0.0, base_z * 0.85),         # 从 -X 侧看 → 最能看清膝盖弯曲方向
        "top34": (-d * 0.62, -d * 0.62, base_z + 0.55),  # 斜俯视 → 看脚掌朝向
    }
    cams = {}
    for name, pos in views.items():
        cfg = CameraCfg(
            prim_path=f"/World/Cam_{name}",
            update_period=0.0,
            height=640,
            width=800,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=20.0, focus_distance=400.0,
                horizontal_aperture=20.955, clipping_range=(0.05, 100.0)),
            offset=CameraCfg.OffsetCfg(pos=pos, rot=look_at_quat(pos, target), convention="world"),
        )
        cams[name] = Camera(cfg)
    return cams


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sim = SimulationContext(sim_utils.SimulationCfg(dt=PHYS_DT, device=DEVICE, gravity=(0.0, 0.0, -9.81)))

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    dome_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.85, 0.85, 0.9))
    dome_cfg.func("/World/DomeLight", dome_cfg)
    sun_cfg = sim_utils.DistantLightCfg(intensity=2500.0, color=(1.0, 0.98, 0.95))
    sun_cfg.func("/World/SunLight", sun_cfg, translation=(1.5, -2.0, 3.0))

    robot = Articulation(build_poppy_cfg())
    cams = make_cameras(args_cli.base_z)
    sim.reset()

    print(f"释放高度 {args_cli.base_z} m（运动学识别：脚在 pelvis 下方 0.386 m）")
    print(f"关节顺序：{robot.joint_names}\n")

    summary = []
    for tag, desc, pose in TEST_CASES:
        pose_t = pose_to_tensor(robot, pose)
        reset_robot(robot, pose_t, args_cli.base_z)
        robot.set_joint_position_target(pose_t)

        steps = int(args_cli.settle / PHYS_DT)
        for step in range(steps):
            if step % DECIMATION == 0:
                robot.set_joint_position_target(pose_t)
            sim.step()
            robot.update(PHYS_DT)

        achieved = robot.data.joint_pos[0].detach().cpu()
        cmd = pose_t[0].detach().cpu()
        grav = torch.tensor([0.0, 0.0, -1.0], device=DEVICE).repeat(robot.num_instances, 1)
        pg = quat_rotate_inverse(robot.data.root_quat_w, grav)
        tilt = float(torch.rad2deg(torch.acos((-pg[0, 2]).clamp(-1.0, 1.0))))
        h = float(robot.data.root_pos_w[0, 2])

        print("=" * 88)
        print(f"[{tag}] {desc}")
        print(f"  落地后：基座高度 {h:.4f} m，倾角 {tilt:.2f}°")
        print(f"  {'关节':<12}{'指令 rad':>12}{'实际 rad':>12}{'差 rad':>11}   限位是否卡住")
        print("  " + "-" * 62)
        clamped = []
        for i, jn in enumerate(robot.joint_names):
            diff = float(achieved[i] - cmd[i])
            flag = ""
            if abs(diff) > 0.02:
                flag = "★被限位/求解器卡住"
                clamped.append(jn)
            print(f"  {jn:<12}{float(cmd[i]):>12.4f}{float(achieved[i]):>12.4f}{diff:>11.4f}   {flag}")

        imgs = []
        for cname, cam in cams.items():
            cam.update(PHYS_DT)
            rgb = cam.data.output["rgb"]
            img = rgb[0, :, :, :3].to(torch.uint8).cpu().numpy()
            path = os.path.join(OUT_DIR, f"{args_cli.tag}_{tag}_{cname}.png")
            write_png(path, img)
            imgs.append(path)
        print(f"  图片：{', '.join(os.path.basename(p) for p in imgs)}")

        summary.append({
            "tag": tag, "desc": desc, "pose_cmd": pose,
            "achieved": {jn: float(achieved[i]) for i, jn in enumerate(robot.joint_names)},
            "clamped_joints": clamped,
            "settled_height": h, "settled_tilt_deg": tilt, "images": imgs,
        })

    with open(os.path.join(OUT_DIR, f"{args_cli.tag}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {os.path.join(OUT_DIR, args_cli.tag + '_summary.json')}")
    simulation_app.close()


if __name__ == "__main__":
    main()
