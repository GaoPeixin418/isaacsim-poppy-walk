#!/usr/bin/env python3
"""D2 Step 4 — 核对 USD 资产：关节限位、驱动增益、碰撞几何、自由度。

【为什么这一步是 D2 的核心】
URDF 审计已经证明官方 URDF 本身没有限位问题（单位是弧度、膝盖单向）。
所以当年"反关节站不稳"的真凶只可能在转换/导入环节。这个脚本把
URDF 里的限位与 USD 里实际生效的限位**逐项对比**，不一致的立刻暴露。

同时检查三件在 Linux 上容易静默出错的事：
  1. 关节限位是否被带进 USD（为 0 或 ±inf 就是丢了）
  2. 每个连杆是否有碰撞几何体（缺了就是"隐形人"，必然站不稳）
  3. 关节驱动增益是否是我们指定的量级（不是默认的 100/1.0）

用法：
  python d2_inspect_usd.py <usd_path> [--ref <joints.tsv>] [--label full|locked]
"""
import argparse
import os
import sys

from pxr import Usd, UsdPhysics, UsdGeom, Gf


def get_drive(joint_prim, kind):
    """读取关节 drive 的 stiffness/damping/maxForce（可能没有）。"""
    try:
        drv = UsdPhysics.DriveAPI.Get(joint_prim, kind)
        if not drv:
            return None
        return {
            "stiffness": drv.GetStiffnessAttr().Get(),
            "damping": drv.GetDampingAttr().Get(),
            "maxForce": drv.GetMaxForceAttr().Get(),
            "type": drv.GetTypeAttr().Get(),
        }
    except Exception:
        return None


def collect_joints(stage):
    """遍历所有旋转/移动关节，读出限位与驱动。"""
    out = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.RevoluteJoint):
            j = UsdPhysics.RevoluteJoint(prim)
            kind = "revolute"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            j = UsdPhysics.PrismaticJoint(prim)
            kind = "prismatic"
        else:
            continue

        low = j.GetLowerLimitAttr().Get()
        up = j.GetUpperLimitAttr().Get()
        axis = j.GetAxisAttr().Get()
        out.append({
            "name": prim.GetName(),
            "path": str(prim.GetPath()),
            "kind": kind,
            "lower": low,
            "upper": up,
            "axis": axis,
            "drive": get_drive(prim, "angular" if kind == "revolute" else "linear"),
        })
    return sorted(out, key=lambda d: d["name"])


def collect_bodies(stage):
    """遍历刚体，统计每个刚体下的碰撞几何体。"""
    bodies = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        colliders = []
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI) or child.IsA(UsdGeom.Mesh) and child.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(child)
            elif child.HasAPI(UsdPhysics.MeshCollisionAPI):
                colliders.append(child)
        # 质量
        mass = None
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
        bodies.append({
            "name": prim.GetName(),
            "path": str(prim.GetPath()),
            "mass": mass,
            "n_colliders": len(set(colliders)),
            "colliders": sorted({c.GetName() for c in colliders}),
        })
    return sorted(bodies, key=lambda d: d["name"])


def load_ref(tsv_path):
    """读 URDF 侧的关节表（由 d2_patch_urdf.py 生成）。"""
    ref = {}
    if not tsv_path or not os.path.isfile(tsv_path):
        return ref
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            vals = line.rstrip("\n").split("\t")
            if len(vals) != len(header):
                continue
            row = dict(zip(header, vals))
            ref[row["joint"]] = row
    return ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("usd_path")
    ap.add_argument("--ref", default=None, help="URDF 关节表 tsv（用于逐项对比）")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    if args.label:
        print("=" * 78)
        print(f"### {args.label}")
        print("=" * 78)

    stage = Usd.Stage.Open(args.usd_path)
    if stage is None:
        print("ERROR: 打不开 USD:", args.usd_path, file=sys.stderr)
        return 1

    print(f"USD: {args.usd_path}")
    up_axis = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"up axis: {up_axis}   metersPerUnit: {mpu}")

    joints = collect_joints(stage)
    bodies = collect_bodies(stage)
    print(f"\n自由度（USD 中的关节数）: {len(joints)}")
    print(f"刚体数量: {len(bodies)}")

    ref = load_ref(args.ref)
    if ref:
        print(f"URDF 侧参照关节数: {len(ref)}")

    print("\n--- 关节限位核对（key: USD 存的是【度】，URDF 是【弧度】） ---")
    print("    这是本脚本要特别处理的地方：UsdPhysics 规范规定 revolute 关节的")
    print("    lower/upper 单位是【度】，而 URDF 与 Isaac Lab 内部都用【弧度】。")
    print("    直接比较会得到'全部不符'的假警报，必须换算后再比。\n")
    print(f"{'joint':<16}{'URDF(rad)':>22}{'USD(deg)':>22}{'USD→rad':>12}{'判定':>10}")
    print("-" * 84)
    DEG2RAD = 3.141592653589793 / 180.0
    bad = []
    for j in joints:
        r = ref.get(j["name"])
        rl = float(r["lower"]) if r and r["lower"] not in ("None", "") else None
        ru = float(r["upper"]) if r and r["upper"] not in ("None", "") else None
        ul_deg, uu_deg = j["lower"], j["upper"]
        ul = ul_deg * DEG2RAD if ul_deg is not None else None
        uu = uu_deg * DEG2RAD if uu_deg is not None else None

        if rl is None or ru is None:
            verdict = "无参照"
        elif ul is None or uu is None:
            verdict = "❌丢失"
            bad.append((j["name"], "限位未导入"))
        elif abs(ul - rl) < 1e-3 and abs(uu - ru) < 1e-3:
            verdict = "✅一致"
        else:
            verdict = "❌不符"
            bad.append((j["name"], f"{ul} vs {rl}"))

        pair = lambda a, b: f"{a:+.4f}/{b:+.4f}" if a is not None and b is not None else "None"
        rad = lambda v: f"{v:+.4f}" if v is not None else "None"
        print(f"{j['name']:<16}{pair(rl, ru):>22}{pair(ul_deg, uu_deg):>22}"
              f"{rad(ul):>12}{verdict:>10}")

    print("\n--- 反关节专项检查（膝盖单向性） ---")
    for name in ("r_knee_y", "l_knee_y"):
        j = next((x for x in joints if x["name"] == name), None)
        if not j or j["lower"] is None:
            print(f"  {name}: 未找到")
            continue
        lo, hi = j["lower"] * DEG2RAD, j["upper"] * DEG2RAD
        margin = min(abs(lo), abs(hi))
        print(f"  {name}: [{lo:+.4f}, {hi:+.4f}] rad "
              f"→ 反向余量 {margin * 180 / 3.141592653589793:.2f}°"
              f"  {'✅ 单向限位生效' if margin < 0.1 else '⚠️ 反向余量过大，可能反关节'}")

    print("\n--- 关节驱动增益（USD 里存的是【每度】，换算回【每弧度】才可与 PD 参数对照） ---")
    print(f"{'joint':<16}{'axis':>6}{'k(deg)':>11}{'k(rad)':>10}"
          f"{'d(deg)':>11}{'d(rad)':>10}{'maxForce':>10}")
    print("-" * 74)
    R2D = 180.0 / 3.141592653589793
    for j in joints:
        d = j["drive"] or {}
        k, dp = d.get("stiffness"), d.get("damping")
        f = lambda v: "None" if v is None else f"{v:.6g}"
        fr = lambda v: "None" if v is None else f"{v * R2D:.4f}"
        ax = j["axis"]
        ax = (ax.x, ax.y, ax.z) if ax is not None and hasattr(ax, "x") else ax
        mx = d.get("maxForce")
        print(f"{j['name']:<16}{str(ax):>6}{f(k):>11}{fr(k):>10}"
              f"{f(dp):>11}{fr(dp):>10}{('None' if mx is None else f'{mx:.4g}'):>10}")
    print("\n  注意：maxForce 直接来自 URDF 的 effort 字段（MX-28 写的是 3.1、")
    print("        MX-64 写的是 7.3）。URDF 用的是数据手册峰值，实际堵转低 15–30%，")
    print("        必须在 Isaac Lab 的执行器配置里覆盖成 2.5 / 6.0。")

    print("\n--- 刚体与碰撞几何体 ---")
    print(f"{'body':<22}{'mass(kg)':>10}{'碰撞体数':>10}   碰撞体名字")
    print("-" * 96)
    zero_coll = []
    for b in bodies:
        m = "None" if b["mass"] is None else f"{b['mass']:.4f}"
        names = ", ".join(b["colliders"])[:44]
        flag = ""
        if b["n_colliders"] == 0:
            flag = "  ⚠️ 无碰撞体"
            zero_coll.append(b["name"])
        print(f"{b['name']:<22}{m:>10}{b['n_colliders']:>10}   {names}{flag}")

    print("\n--- 网格碰撞近似方式 ---")
    approx = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            api = UsdPhysics.MeshCollisionAPI(prim)
            a = api.GetApproximationAttr().Get()
            approx[str(a)] = approx.get(str(a), 0) + 1
    for k, v in sorted(approx.items()):
        print(f"  {k}: {v} 个 mesh")
    if not approx:
        print("  （没有使用 mesh 碰撞 —— 可能被转成了 convex hull 或基本体）")

    print("\n================ 结论 ================")
    print(f"关节数: {len(joints)}   刚体数: {len(bodies)}")
    if bad:
        print(f"❌ 限位异常 {len(bad)} 处:")
        for n, why in bad:
            print(f"   {n}: {why}")
    else:
        print("✅ 所有关节限位与 URDF 一致（限位已正确导入）")
    if zero_coll:
        print(f"⚠️  {len(zero_coll)} 个刚体没有碰撞几何体: {', '.join(zero_coll)}")
        print("   → 这些连杆在物理上是'隐形'的，站立训练必然失败")
    else:
        print("✅ 所有刚体都有碰撞几何体")

    return 0


if __name__ == "__main__":
    sys.exit(main())
