#!/usr/bin/env python3
"""D2 Step 2 — 把 Poppy 官方 URDF 改写成 Isaac Sim 可用的版本。

做三件事：

1) **网格路径改写**：`package://meshes/X.STL` → `meshes/X.STL`
   `package://` 是 ROS 的包解析协议，Isaac Sim 不认。不改写会导致导入后
   所有几何体丢失（USD 变成一个只有关节、没有身体的空壳）。
   改写成相对路径后，网格必须放在 URDF 同目录的 `meshes/` 子目录下。

2) **文件名大小写修正**（**这一步是关键**）：官方仓库里 10 个网格的扩展名
   是小写 `.stl`，而 URDF 一律引用 `.STL`。Windows/macOS 不区分大小写，
   所以本地永远看不出问题；**Linux 区分大小写，这 10 个文件会静默加载失败**。
   其中 4 个（l_hip / l_hip_motor / l_shin / l_foot）构成整条左腿的碰撞几何
   —— 左腿没有碰撞体，一站起来必然向左塌，与"反关节站不稳"高度吻合。

3) **锁定上身**（可选）：把 15 个非腿部关节从 `revolute` 改成 `fixed`，
   配合转换时的 `--merge-joints` 把上身各连杆合并成刚体，
   使 articulation 从 25 DOF 降到 10 DOF（只剩双腿）。
   好处：物理求解更快更稳、动作空间与设计一致、杜绝"上身乱甩"这类解。
   代价：放弃 Poppy 的 5 自由度多关节躯干特色（7 天冲刺不做加法）。

用法：
  python d2_patch_urdf.py --mode full        # 只改网格路径
  python d2_patch_urdf.py --mode lockupper   # 改网格路径 + 锁定上身
  python d2_patch_urdf.py --mode lockupper --out /path/custom.urdf
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET

URDF_DIR = "/data/poppy/poppy-walking/assets/poppy/urdf"
SRC_NAME = "Poppy_Humanoid_orig.urdf"
MESH_SUBDIR = "meshes"

# 腿部 10 个自由度（每条腿 5 个：侧摆/旋转/俯仰/膝/踝）
LEG_JOINTS = [
    "r_hip_x", "r_hip_z", "r_hip_y", "r_knee_y", "r_ankle_y",
    "l_hip_x", "l_hip_z", "l_hip_y", "l_knee_y", "l_ankle_y",
]
# 上身 15 个（腰腹 3 + 胸 2 + 头 2 + 双臂 8）
UPPER_JOINTS = [
    "abs_y", "abs_x", "abs_z",
    "bust_y", "bust_x",
    "head_z", "head_y",
    "l_shoulder_y", "l_shoulder_x", "l_arm_z", "l_elbow_y",
    "r_shoulder_y", "r_shoulder_x", "r_arm_z", "r_elbow_y",
]

# fixed 关节按 URDF 规范不应带 axis / limit / dynamics，否则部分解析器会报错
STRIP_FOR_FIXED = ["axis", "limit", "dynamics", "calibration", "safety_controller"]


def fix_mesh_case(root: ET.Element, urdf_dir: str) -> list:
    """修正网格文件名的大小写不匹配。

    【为什么必须做这一步】
    官方仓库里有 10 个文件的扩展名是小写的 `.stl`（其余 42 个是大写 `.STL`），
    但 URDF 里一律引用 `.STL`。Windows/macOS 的文件系统不区分大小写，
    所以这个问题在本地永远不会暴露；**但 Linux 区分大小写**，
    结果是这 10 个文件在服务器上"文件不存在"，
    导入器只能静默跳过 → 对应的连杆没有碰撞体。

    受害的 10 个文件里，有 4 个（l_hip / l_hip_motor / l_shin / l_foot）
    构成**整条左腿的碰撞几何**。也就是说：在 Linux 上按原样导入，
    机器人左腿完全没有碰撞体，一站起来必然向左塌 —— 这与
    "反关节站不稳"的现象高度吻合，是当年失败的头号嫌疑。

    返回 [(URDF里的写法, 磁盘上的实际文件名), ...]
    """
    mesh_dir = os.path.join(urdf_dir, MESH_SUBDIR)
    if not os.path.isdir(mesh_dir):
        return []
    actual = {n.lower(): n for n in os.listdir(mesh_dir)}

    fixes = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        d, base = os.path.split(fn)
        real = actual.get(base.lower())
        if real and real != base:
            mesh.set("filename", os.path.join(d, real) if d else real)
            fixes.append((base, real))
    return sorted(set(fixes))


def patch_mesh_paths(root: ET.Element) -> int:
    """把所有 package://meshes/ 前缀改成 meshes/，返回改写数量。"""
    n = 0
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if fn and fn.startswith("package://meshes/"):
            mesh.set("filename", fn.replace("package://meshes/", f"{MESH_SUBDIR}/"))
            n += 1
        elif fn and fn.startswith("package://"):
            # 其他形式的包路径也一并处理，避免漏网
            mesh.set("filename", fn.split("package://", 1)[1])
            n += 1
    return n


def lock_upper_body(root: ET.Element) -> list:
    """把上身关节改成 fixed 并清掉 axis/limit/dynamics。返回被改的关节名。"""
    changed = []
    for j in root.findall("joint"):
        name = j.get("name")
        if name not in UPPER_JOINTS:
            continue
        if j.get("type") == "fixed":
            continue
        j.set("type", "fixed")
        for tag in STRIP_FOR_FIXED:
            el = j.find(tag)
            if el is not None:
                j.remove(el)
        changed.append(name)
    return changed


def check_meshes(root: ET.Element, urdf_dir: str) -> tuple:
    """校验每个引用的网格文件是否真实存在，返回 (ok, missing)。"""
    ok, missing = 0, []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        path = fn if os.path.isabs(fn) else os.path.join(urdf_dir, fn)
        if os.path.isfile(path):
            ok += 1
        else:
            missing.append(fn)
    return ok, sorted(set(missing))


def joint_table(root: ET.Element) -> list:
    """输出关节清单，供后续与 USD 里的限位逐项核对。"""
    rows = []
    for j in root.findall("joint"):
        lim = j.find("limit")
        rows.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": (j.find("parent").get("link") if j.find("parent") is not None else "?"),
            "child": (j.find("child").get("link") if j.find("child") is not None else "?"),
            "lower": (lim.get("lower") if lim is not None else None),
            "upper": (lim.get("upper") if lim is not None else None),
            "effort": (lim.get("effort") if lim is not None else None),
            "velocity": (lim.get("velocity") if lim is not None else None),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "lockupper"], default="lockupper")
    ap.add_argument("--urdf-dir", default=URDF_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src = os.path.join(args.urdf_dir, SRC_NAME)
    if not os.path.isfile(src):
        print("ERROR: 找不到原始 URDF:", src, file=sys.stderr)
        return 1

    default_out = {
        "full": "Poppy_Humanoid.urdf",
        "lockupper": "Poppy_Humanoid_locked.urdf",
    }[args.mode]
    dst = args.out or os.path.join(args.urdf_dir, default_out)

    tree = ET.parse(src)
    root = tree.getroot()
    print(f"原始 URDF: {src}")
    print(f"robot name: {root.get('name')}   关节总数: {len(root.findall('joint'))}")

    n_mesh = patch_mesh_paths(root)
    print(f"\n[1] 网格路径改写: {n_mesh} 处  package://meshes/ → meshes/")

    case_fixes = fix_mesh_case(root, args.urdf_dir)
    print(f"[2] 文件名大小写修正: {len(case_fixes)} 处（Linux 大小写敏感，不改必然丢几何体）")
    for ref, real in case_fixes:
        print(f"    {ref}  →  {real}")

    ok, missing = check_meshes(root, args.urdf_dir)
    print(f"[3] 网格存在性校验: {ok} 个存在, {len(missing)} 个缺失")
    for m in missing:
        print("    缺失:", m)
    if missing:
        print("    ⚠️  仍有缺失网格 → 转换后会丢几何体，必须先补齐")
    else:
        print("    ✅ 全部网格可解析")

    if args.mode == "lockupper":
        changed = lock_upper_body(root)
        print(f"\n[4] 锁定上身: {len(changed)} 个关节 revolute → fixed")
        print("    " + ", ".join(changed))
    else:
        print("\n[4] 锁定上身: 跳过（full 模式，保留全部 25 DOF）")

    # 统计改写后的关节构成
    types = {}
    for j in root.findall("joint"):
        types[j.get("type")] = types.get(j.get("type"), 0) + 1
    print(f"\n改写后关节构成: {types}")
    n_rev = types.get("revolute", 0)
    print(f"          → 可动自由度 = {n_rev}")

    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)
    print(f"\n输出: {dst}  ({os.path.getsize(dst)} 字节)")

    rows = joint_table(root)
    out_tab = dst.replace(".urdf", "_joints.tsv")
    with open(out_tab, "w") as f:
        f.write("joint\ttype\tparent\tchild\tlower\tupper\teffort\tvelocity\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in
                              ("name", "type", "parent", "child",
                               "lower", "upper", "effort", "velocity")) + "\n")
    print(f"关节表: {out_tab}（用于与 USD 逐项核对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
