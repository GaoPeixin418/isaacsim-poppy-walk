#!/usr/bin/env python3
"""D3 前置实验 ④：直接读 USD 里的关节轴向与脚掌几何 —— 定案"哪边是前"

为什么需要这个
--------------
运动学识别给出了两条互相矛盾的信息：

  · 限位表说：右膝的可弯方向是**负**（[-2.339, +0.061]），左膝的可弯方向是**正**（[-0.061, +2.339]）
  · 数值雅可比说：右膝 +0.3 rad → 右脚往 -Y 走；左膝 +0.3 rad → 左脚也往 -Y 走
    （同符号 → 同方向的位移，这是"左右镜像轴"或"平行轴"都能解释的，雅可比本身分不出来）

要理清只有两条路：
  1. **直接读 USD 的 `physics:axis`**：轴向到底是平行还是反平行？限位和轴向是否自洽？
  2. **直接量脚掌几何**：脚掌网格在 Y 方向不对称，脚尖那一侧就是"前方"。

两条都在这个脚本里，纯 USD 读取，不需要启动 Isaac Sim（秒级完成）。

输出
----
  out/d3_rest_pose/usd_introspect.json
"""

from __future__ import annotations

import json
import os

from pxr import Gf, Usd, UsdGeom, UsdPhysics

REPO = os.environ.get("POPPY_REPO", "/data/poppy/poppy-walking")
POPPY_USD = os.path.join(REPO, "assets", "poppy", "poppy.usd")
OUT_DIR = os.path.join(REPO, "out", "d3_rest_pose")

LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]


def get_float(prim, attr_name, default=None):
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        return default
    v = a.Get()
    if v is None:
        return default
    try:
        return float(v)
    except TypeError:
        return v


def get_axis_vec(prim, attr_name="physics:axis"):
    """读 `physics:axis`。

    注意：USD Physics 里这个属性是 **uniform token**，取值只能是 "X" / "Y" / "Z"，
    表示转轴是【关节坐标系的哪一根主轴】—— 不是直接给一个向量。
    所以想知道轴在世界里的指向，必须结合 localRot0（关节坐标系相对父刚体的旋转）来推。
    这一点很反直觉，也是"关节朝向"类 bug 的温床。
    """
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        return None
    v = a.Get()
    if v is None:
        return None
    token = str(v).upper()
    table = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0), "Z": (0.0, 0.0, 1.0)}
    return list(table.get(token, (0.0, 0.0, 1.0))), token


def get_vec(prim, attr_name):
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        return None
    v = a.Get()
    return [float(x) for x in v] if v is not None else None


def get_quat(prim, attr_name):
    """USD 的 quat 属性是 Gf.Quatf/Quatd，实部在前 (w, (x,y,z))。"""
    a = prim.GetAttribute(attr_name)
    if not a or not a.IsValid():
        return None
    v = a.Get()
    if v is None:
        return None
    im = v.GetImaginary()
    return [float(v.GetReal()), float(im[0]), float(im[1]), float(im[2])]


def quat_to_matrix(q):
    """四元数 (w,x,y,z) → 3x3 numpy 风格嵌套列表。"""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]


def mat_mul(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    stage = Usd.Stage.Open(POPPY_USD)
    if stage is None:
        raise RuntimeError(f"打不开 USD: {POPPY_USD}")

    default_prim = stage.GetDefaultPrim()
    print(f"USD            : {POPPY_USD}")
    print(f"默认 prim      : {default_prim.GetPath()}  (类型 {default_prim.GetTypeName()})")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    # ---------------------------------------------------------------
    # 1. 关节：轴向、局部坐标系、限位、驱动
    # ---------------------------------------------------------------
    joints = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        name = prim.GetName()
        if name not in LEG_JOINTS:
            continue
        joints[name] = prim

    print(f"\n找到腿部 revolute 关节 {len(joints)} / {len(LEG_JOINTS)} 个")

    print("\n[1] 关节轴向（`physics:axis` 是 token，表示关节坐标系的哪根主轴）")
    print(f"{'关节':<12}{'axis token':>11}{'lower':>10}{'upper':>10}{'drive 刚度':>12}{'drive 阻尼':>11}")
    print("-" * 80)
    joint_rows = {}
    for name in LEG_JOINTS:
        p = joints.get(name)
        if p is None:
            print(f"{name:<12}{'缺失!':>11}")
            continue
        axis_pair = get_axis_vec(p)
        axis, token = axis_pair if axis_pair else (None, "None")
        lo = get_float(p, "physics:lowerLimit")
        hi = get_float(p, "physics:upperLimit")
        stiff = get_float(p, "drive:angular:physics:stiffness")
        damp = get_float(p, "drive:angular:physics:damping")
        print(f"{name:<12}{token:>11}{(f'{lo:+.4f}' if lo is not None else 'None'):>10}"
              f"{(f'{hi:+.4f}' if hi is not None else 'None'):>10}"
              f"{(f'{stiff:.6f}' if stiff is not None else 'None'):>12}"
              f"{(f'{damp:.6f}' if damp is not None else 'None'):>11}")
        joint_rows[name] = {"axis": axis, "axis_token": token, "lower": lo, "upper": hi,
                            "drive_stiffness": stiff, "drive_damping": damp,
                            "path": str(p.GetPath())}

    # ---- 计算世界坐标系下的关节轴（rest pose） ----
    print("\n[2] 关节轴在世界坐标系下的方向（用 localRot0 + 父刚体世界旋转推出）")
    print("    【这是判断左右腿关节轴是否镜像的最终依据】")
    print(f"{'关节':<12}{'父刚体':>12}{'token':>6}{'轴(世界)':>24}{'关节原点(世界)':>28}")
    print("-" * 84)
    world_axes = {}
    for name in LEG_JOINTS:
        p = joints.get(name)
        if p is None:
            continue
        rel = UsdPhysics.Joint(p)
        b0 = rel.GetBody0Rel().GetTargets()
        b1 = rel.GetBody1Rel().GetTargets()
        body0 = stage.GetPrimAtPath(b0[0]) if b0 else None
        body1 = stage.GetPrimAtPath(b1[0]) if b1 else None

        local_rot0 = get_quat(p, "physics:localRot0") or [1.0, 0.0, 0.0, 0.0]
        local_pos0 = get_vec(p, "physics:localPos0") or [0.0, 0.0, 0.0]
        local_rot1 = get_quat(p, "physics:localRot1") or [1.0, 0.0, 0.0, 0.0]
        local_pos1 = get_vec(p, "physics:localPos1") or [0.0, 0.0, 0.0]

        if body0 is not None:
            m0 = xform_cache.GetLocalToWorldTransform(body0)
            R0 = [[m0[i][j] for j in range(3)] for i in range(3)]
            t0 = [m0[3][i] for i in range(3)]
        else:
            R0 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            t0 = [0.0, 0.0, 0.0]

        joint_rot = mat_mul(R0, quat_to_matrix(local_rot0))
        axis = joint_rows[name]["axis"] or [0.0, 0.0, 1.0]
        axis_w = mat_vec(joint_rot, axis)
        origin_w = [t0[i] + sum(R0[i][k] * local_pos0[k] for k in range(3)) for i in range(3)]

        body0_name = body0.GetName() if body0 is not None else "-"
        ax_s = "(" + ", ".join(f"{x:+.3f}" for x in axis_w) + ")"
        or_s = "(" + ", ".join(f"{x:+.4f}" for x in origin_w) + ")"
        print(f"{name:<12}{body0_name:>12}{joint_rows[name]['axis_token']:>6}{ax_s:>24}{or_s:>28}")

        # 关节坐标系的三根轴在世界里的指向，便于看清 localRot0 到底把轴拧到哪去了
        cols = [[joint_rot[i][j] for i in range(3)] for j in range(3)]
        frame_s = " | ".join("(" + ",".join(f"{c:+.2f}" for c in col) + ")" for col in cols)
        print(f"{'':12}{'  ↳ 关节系三轴(世界) X|Y|Z = '}{frame_s}")

        world_axes[name] = {"body0": body0_name, "axis_world": axis_w, "origin_world": origin_w,
                            "local_pos0": local_pos0, "local_pos1": local_pos1,
                            "local_rot0": local_rot0, "local_rot1": local_rot1,
                            "axis_token": joint_rows[name]["axis_token"],
                            "joint_frame_axes_world": cols,
                            "body1": body1.GetName() if body1 is not None else None}

    # ---- 自洽性检查：左右镜像关节的轴是否反平行 ----
    print("\n[3] 左右镜像自洽性检查")
    print("    镜像约定下，左腿关节轴应当是右腿关节轴关于矢状面的镜像（→ 轴向近似反平行）")
    print("    同时：如果轴向反平行，则【同一个物理动作】在左右腿需要【相同】的数值符号")
    print(f"{'右关节':<12}{'左关节':<12}{'轴点积':>10}{'右限位':>20}{'左限位':>20}{'限位符号是否一致':>18}")
    print("-" * 100)
    mirror_rows = []
    for suffix in ["hip_x", "hip_z", "hip_y", "knee_y", "ankle_y"]:
        rn, ln = f"r_{suffix}", f"l_{suffix}"
        if rn not in world_axes or ln not in world_axes:
            continue
        ar = world_axes[rn]["axis_world"]
        al = world_axes[ln]["axis_world"]
        dot = sum(ar[i] * al[i] for i in range(3))
        rlo, rhi = joint_rows[rn]["lower"], joint_rows[rn]["upper"]
        llo, lhi = joint_rows[ln]["lower"], joint_rows[ln]["upper"]
        # 判据：把右腿的弯曲方向（离 0 更远的那一侧）镜像到左腿，看左腿限位是否在镜像后的同一侧
        r_flex = -1 if abs(rlo) > abs(rhi) else +1
        l_flex = -1 if abs(llo) > abs(lhi) else +1
        consistent = (r_flex == l_flex) if dot < 0 else (r_flex != l_flex)
        print(f"{rn:<12}{ln:<12}{dot:>10.3f}"
              f"{f'[{rlo:+.3f}, {rhi:+.3f}]':>20}{f'[{llo:+.3f}, {lhi:+.3f}]':>20}"
              f"{('一致' if consistent else '★不一致★'):>18}")
        mirror_rows.append({"right": rn, "left": ln, "axis_dot": dot,
                            "r_limits": [rlo, rhi], "l_limits": [llo, lhi],
                            "r_flex_sign": r_flex, "l_flex_sign": l_flex,
                            "consistent": bool(consistent)})

    # ---------------------------------------------------------------
    # 2. 脚掌网格几何：Y 方向不对称 → 长的那侧是脚尖 → 前方
    # ---------------------------------------------------------------
    print("\n[4] 全部 mesh prim 的世界 AABB（rest pose）—— 用几何不对称性判断【哪边是前】")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    mesh_rows = {}
    all_meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        bnd = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mn, mx = bnd.GetMin(), bnd.GetMax()
        rec = {
            "path": path, "name": prim.GetName(),
            "min": [float(x) for x in mn], "max": [float(x) for x in mx],
            "size": [float(mx[i] - mn[i]) for i in range(3)],
            "center": [float(0.5 * (mn[i] + mx[i])) for i in range(3)],
        }
        all_meshes.append(rec)
        for link in ["r_foot", "l_foot", "r_shin", "l_shin"]:
            if f"/{link}/" in path:
                mesh_rows.setdefault(link, []).append(rec)

    print(f"  阶段内 mesh 总数：{len(all_meshes)}")
    for link in ["r_foot", "l_foot"]:
        print(f"\n  ── {link} 的网格 ──")
        for rec in mesh_rows.get(link, []):
            mn, mx = rec["min"], rec["max"]
            print(f"    {rec['path']}")
            print(f"      x[{mn[0]:+.4f},{mx[0]:+.4f}] y[{mn[1]:+.4f},{mx[1]:+.4f}] z[{mn[2]:+.4f},{mx[2]:+.4f}]")
            print(f"      尺寸 {rec['size'][0]:.4f} x {rec['size'][1]:.4f} x {rec['size'][2]:.4f} m"
                  f"   中心 y = {rec['center'][1]:+.4f} m")
    if not mesh_rows:
        print("  （没按 /<link>/ 路径匹配到，下面列出所有 mesh 路径供排查）")
        for rec in all_meshes[:40]:
            print(f"    {rec['path']}  尺寸 {rec['size'][0]:.3f}x{rec['size'][1]:.3f}x{rec['size'][2]:.3f}")

    result = {
        "usd": POPPY_USD,
        "joints": joint_rows,
        "world_axes": world_axes,
        "mirror_check": mirror_rows,
        "meshes": mesh_rows,
        "all_meshes": all_meshes,
    }
    with open(os.path.join(OUT_DIR, "usd_introspect.json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(f"\n[保存] {os.path.join(OUT_DIR, 'usd_introspect.json')}")


if __name__ == "__main__":
    main()
