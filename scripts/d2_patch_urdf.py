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

4) **镜像轴向一致性修正**（D3 发现，见下）：把左腿"漏翻符号"的关节轴修正。

---

## 第 4 步为什么必须做（D3 的核心发现）

对镜像布置的机器人，左腿关节的正确写法有两种**自洽**组合：

  | 约定 | 左腿 axis | 左腿 limit | 效果 |
  |---|---|---|---|
  | A | 与右腿**符号相反** | 与右腿**镜像** | 同一物理动作 = 左右相反数值 |
  | B | 与右腿**相同** | 与右腿**相同** | 同一物理动作 = 左右相同数值 |

**混用（axis 相同 + limit 镜像）是错误的**：它会让同一个数值在左右腿产生
同一个世界方向的运动，但限位却按镜像去限制 —— 结果是"一条腿能往这边弯、
另一条腿只能往那边弯"，对称的屈膝动作在物理上不可能实现。

官方 Poppy URDF 里就踩了这个坑：

  ```
  r_hip_y    axis (-1 0 0)   limit [-85°, +105°]   ← 约定 A ✔
  l_hip_y    axis (+1 0 0)   limit [-104°, +84°]   ← 约定 A ✔
  r_ankle_y  axis (-1 0 0)   limit [-45°, +45°]    ← 约定 A ✔
  l_ankle_y  axis (+1 0 0)   limit [-45°, +45°]    ← 约定 A ✔
  r_knee_y   axis (0 0 -1)   limit [-134°, +3.5°]  ← 约定 A ✔
  l_knee_y   axis (0 0 -1)   limit [-3.5°, +134°]  ← 【axis 忘了翻符号】✘
  r_hip_z    axis (0 -1 0)   limit [-90°, +25°]    ← 约定 A ✔
  l_hip_z    axis (0 -1 0)   limit [-25°, +90°]    ← 【axis 忘了翻符号】✘
  ```

`l_knee_y` 的 `axis` 漏了符号翻转 —— **左膝因此只能朝"反着弯"的方向运动**。
这极可能就是当年"反关节站不稳"的真凶：策略为了让左腿跟上，只能启用
膝盖的反向自由度，视觉上就是左腿反关节。

（`l_hip_x` 的限位近似对称 [[-28.5°,+30°] vs [-30°,+28.5°]，影响只有 1.5°，
一并修正只为保持约定统一。）

## 判定规则（本脚本自动执行，不做任何人工猜测）

**第一版规则（已废弃，记录在这里当教训）**：
「限位是镜像的 且 左右 axis 相同 → 翻转」。它把 `l_hip_x` / `l_hip_z` / `l_knee_y`
三个都翻了。但其中**只有 `l_knee_y` 是真坏**：

  · **双向关节**（脚踝 ±45°、髋侧摆 ±30°、髋偏航 [-90°,+25°] 对 [-25°,+90°]）：
    翻转 axis 只是换了个"正方向"的定义，**能力完全一样**。翻了反而破坏
    「所有镜像关节统一 r = -l」这个干净约定，后面写奖励函数必踩坑。
  · **单向关节**（膝盖 [-134°,+3.5°]）：两种约定下可用的**对称行程**差 20 倍。

**现在的规则**：不看符号是否一致，而是直接算两种约定下的可用对称行程。

  同符号约定窗口 = [max(r_lo, l_lo), min(r_hi, l_hi)]
  反符号约定窗口 = [max(r_lo, -l_hi), min(r_hi, -l_lo)]

  膝盖：同符号窗口 0.12 rad（3.5°）/ 反符号窗口 2.40 rad（137°）→ 必须翻
  髋侧摆：同符号 0.995 rad / 反符号 1.02 rad → 几乎一样 → **不动**

  阈值：同符号窗口 < 0.3 rad 且 反符号窗口 > 1.0 rad 才翻。
  另外，若两种约定都窄（同符号 < 0.3 且反符号也 ≤ 1.0），打印 ★可疑★ 提示人工检查。

  这样既修好膝盖，又保住统一约定 —— 最终结果是：
  **`l_knee_y` 一处改动，之后所有镜像关节都遵守 r = -l。**

用法：
  python d2_patch_urdf.py --mode full        # 只改网格路径
  python d2_patch_urdf.py --mode lockupper   # 改网格路径 + 锁定上身 + 镜像轴向修正
  python d2_patch_urdf.py --mode lockupper --out /path/custom.urdf
  python d2_patch_urdf.py --mode lockupper --no-axis-fix   # 关掉第 5 步（做对照）
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


def fix_mirror_axes(root: ET.Element) -> list:
    """修正左腿"忘记翻转符号"的关节轴 —— 只修真正坏的关节。

    【判据的来历：第一版规则过度触发，这里改成有物理依据的版本】

    第一版规则是「限位是镜像的 且 左右 axis 相同 → 翻转」。
    它把 l_hip_x / l_hip_z / l_knee_y 三个都翻了。但其中只有 **l_knee_y 是真的坏**：

      · 对一个"双向对称"的关节（如脚踝 ±45°、髋侧摆 ±30°），
        翻转 axis 只是把"正方向"的定义换了个符号 —— 能力完全一样，翻与不翻都自洽。
      · 对一个"单向"关节（膝盖）就不一样了：它的两种约定给出的可用**对称行程**天差地别。

    所以我们真正要判的不是"符号是否一致"，而是：

      **在「同符号(r=l)」和「反符号(r=-l)」两种约定下，对称动作可用行程差多少？**

    记右腿限位 [r_lo, r_hi]、左腿 [l_lo, l_hi]，则

      同符号约定可用窗口 = [max(r_lo, l_lo), min(r_hi, l_hi)]
      反符号约定可用窗口 = [max(r_lo, -l_hi), min(r_hi, -l_lo)]

    膝盖代入（弧度）：
      同符号 → [max(-2.339, -0.061), min(0.061, 2.339)] = [-0.061, 0.061]，窗口 0.12 rad（7°）
      反符号 → [max(-2.339, -2.339), min(0.061, 0.061)] = [-2.339, 0.061]，窗口 2.40 rad（137°）
      → 天差地别，说明膝盖的 axis 必须翻

    髋侧摆代入：
      同符号 → 窗口 0.995 rad；反符号 → 窗口 1.02 rad  → 几乎一样，**不该动**

    所以阈值取：同符号窗口 < 0.3 rad 且 反符号窗口 > 1.0 rad 时才翻转。
    这样既修好了膝盖，又保住了「所有镜像关节统一用 r = -l」这个干净约定
    （否则会出现"髋侧摆要同符号、膝盖要反符号"的混乱约定，后面写奖励函数必踩坑）。

    返回 [(关节名, 旧 axis, 新 axis, 判定依据), ...]
    """
    def axis_of(j):
        a = j.find("axis")
        if a is None or not a.get("xyz"):
            return None
        return [float(x) for x in a.get("xyz").split()]

    def limits_of(j):
        lim = j.find("limit")
        if lim is None:
            return None
        lo, hi = lim.get("lower"), lim.get("upper")
        if lo is None or hi is None:
            return None
        return float(lo), float(hi)

    by_name = {j.get("name"): j for j in root.findall("joint")}
    fixes = []
    for suffix in ["hip_x", "hip_z", "hip_y", "knee_y", "ankle_y"]:
        jr, jl = by_name.get(f"r_{suffix}"), by_name.get(f"l_{suffix}")
        if jr is None or jl is None:
            continue
        ar, al = axis_of(jr), axis_of(jl)
        lr, ll = limits_of(jr), limits_of(jl)
        if ar is None or al is None or lr is None or ll is None:
            continue

        mirrored = (abs(ll[0] + lr[1]) < 1e-6) and (abs(ll[1] + lr[0]) < 1e-6)
        if not mirrored or al != ar:
            continue  # 限位不是镜像、或 axis 已经翻过符号 → 没问题

        same_window = min(lr[1], ll[1]) - max(lr[0], ll[0])
        opp_window = min(lr[1], -ll[0]) - max(lr[0], -ll[1])
        if same_window < 0.3 and opp_window > 1.0:
            new_axis = [(-x if x != 0 else 0.0) for x in al]   # 顺手把 -0 规范成 0
            jl.find("axis").set("xyz", " ".join(f"{x:g}" for x in new_axis))
            fixes.append((f"l_{suffix}",
                          " ".join(f"{x:g}" for x in al),
                          " ".join(f"{x:g}" for x in new_axis),
                          f"同符号对称窗口仅 {same_window:.3f} rad，"
                          f"反符号 {opp_window:.3f} rad → 单向关节，必须翻"))
        elif same_window < 0.3:
            fixes.append((f"l_{suffix}", "(未改)", "(未改)",
                          f"★可疑★ 同符号窗口仅 {same_window:.3f} rad 且反符号也窄"
                          f"({opp_window:.3f})，两种约定都不可用，需人工检查"))
    return fixes


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
        ax = j.find("axis")
        rows.append({
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": (j.find("parent").get("link") if j.find("parent") is not None else "?"),
            "child": (j.find("child").get("link") if j.find("child") is not None else "?"),
            "axis": (ax.get("xyz") if ax is not None else None),
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
    ap.add_argument("--no-axis-fix", action="store_true",
                    help="跳过镜像轴向修正（保留官方 bug，用于做对照实验）")
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

    if args.no_axis_fix:
        print("\n[5] 镜像轴向修正: 跳过（--no-axis-fix，保留官方 bug 做对照）")
    else:
        axis_fixes = fix_mirror_axes(root)
        real = [f for f in axis_fixes if f[1] != "(未改)"]
        print(f"\n[5] 镜像轴向修正: 实际改动 {len(real)} 处"
              f"（仅修「单向关节」的漏翻符号，双向关节不动以保持统一约定）")
        for name, old, new, why in axis_fixes:
            print(f"    {name:<11} axis {old:<8} → {new:<8}   {why}")

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
        f.write("joint\ttype\tparent\tchild\taxis\tlower\tupper\teffort\tvelocity\n")
        for r in rows:
            f.write("\t".join(str(r[k]) for k in
                              ("name", "type", "parent", "child", "axis",
                               "lower", "upper", "effort", "velocity")) + "\n")
    print(f"关节表: {out_tab}（用于与 USD 逐项核对）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
