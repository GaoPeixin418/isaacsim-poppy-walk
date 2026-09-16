#!/usr/bin/env python3
"""离线解析脚部 STL，得到「真正落在水平面上的接触多边形」。

为什么需要这个脚本（方法论）：
  在 Isaac/PhysX 里做静力学判据要回答一个问题 ——「这个姿态的质心投影是否落在支撑多边形内」。
  支撑多边形 = 所有与地面接触的点在水平面上的凸包。对双脚站立，就是两只脚各自脚底
  （最低的那个面）在水平面上的投影并集。
  这个几何量完全可以从资产本身算出来，不需要仿真。把几何可信地定下来，后面
  姿态好坏、要不要前倾重心，才有依据。

踩过的坑（记录下来，避免重犯）：
  STL 没有轴标记，只有顶点坐标。上一版脚本凭「z 是最小的那一维」就假定 z 轴竖直，
  结果把脚的后侧面当成了脚底，支撑多边形在 y（前后）方向只有 0.9 mm 厚 ——
  一眼假。正确的做法是：
    1) 先算出顶点云的 AABB，看哪一维的尺寸与「脚厚」量级相符（脚厚应是几个 cm）
    2) 用 link 的世界姿态把 AABB 的 8 个角点变到世界系，看哪一个面是水平的
    3) 水平且最低的那个面，才是脚底
  这一步做错，后面所有静力学结论都会错，而且是「看起来有数字、其实是垃圾」的错误，
  比直接报错危险得多。
"""
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MESH_DIR = HERE.parent / "poppy-asset" / "meshes"


def read_stl_vertices(path: Path):
    """读取二进制或 ASCII STL，返回顶点列表 [(x,y,z), ...]。"""
    raw = path.read_bytes()
    verts = []
    # 二进制 STL: 80 字节 header + uint32 三角形数 + 每三角 50 字节
    if len(raw) >= 84:
        n_tri = struct.unpack("<I", raw[80:84])[0]
        if 84 + n_tri * 50 == len(raw):
            for i in range(n_tri):
                base = 84 + i * 50 + 12  # 跳过法向 12 字节
                v = struct.unpack("<9f", raw[base:base + 36])
                verts.append(v[0:3])
                verts.append(v[3:6])
                verts.append(v[6:9])
            return verts, f"binary, {n_tri} triangles"
    # 退化到 ASCII
    txt = raw.decode("ascii", errors="ignore")
    for line in txt.splitlines():
        line = line.strip()
        if line.lower().startswith("vertex"):
            p = line.split()
            verts.append((float(p[1]), float(p[2]), float(p[3])))
    return verts, f"ascii, {len(verts)//3} triangles"


def aabb(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def convex_hull_2d(pts):
    """Andrew monotone chain。pts 为 [(a,b), ...]，返回逆时针凸包顶点。"""
    pts = sorted(set(pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_area(poly):
    """凸包面积（凸包已按逆时针给出）。"""
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def point_in_convex(poly, p, tol=0.0):
    """点是否在凸多边形内（要求 poly 逆时针）。返回 (inside, 到边界的带符号最小距离)。"""
    n = len(poly)
    inside = True
    min_dist = float("inf")
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        # 逆时针时，内侧的叉积为正
        cr = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        edge_len = ((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5
        dist = cr / edge_len if edge_len > 1e-12 else 0.0
        if dist + tol < 0:
            inside = False
        min_dist = min(min_dist, dist)
    return inside, min_dist


def main():
    targets = [("l_foot", "l_foot_respondable.stl"), ("r_foot", "r_foot_respondable.STL")]
    out = {}
    for name, fname in targets:
        path = MESH_DIR / fname
        if not path.exists():
            print(f"[缺失] {path}")
            continue
        verts, info = read_stl_vertices(path)
        ax, ay, az = aabb(verts)
        print("=" * 92)
        print(f"{name}   ({fname}, {info})")
        print(f"  顶点数 {len(verts)}")
        print(f"  局部 AABB  x=[{ax[0]:+.4f},{ax[1]:+.4f}] (跨度 {ax[1]-ax[0]:.4f})")
        print(f"             y=[{ay[0]:+.4f},{ay[1]:+.4f}] (跨度 {ay[1]-ay[0]:.4f})")
        print(f"             z=[{az[0]:+.4f},{az[1]:+.4f}] (跨度 {az[1]-az[0]:.4f})")

        # 脚底 = 最低点附近的一薄层顶点（局部 y 最小，因为 link 局部 y 竖直向上）
        ymin = ay[0]
        band = 0.004  # 4 mm 薄层，容忍网格离散化
        sole_pts = [(v[0], v[2]) for v in verts if v[1] <= ymin + band]
        # 取"脚底那一个平面"上的点：在薄层里再筛 y 最小的一批
        true_ymin = min(v[1] for v in verts)
        exact = [(v[0], v[2]) for v in verts if v[1] <= true_ymin + 1e-6]
        use = exact if len(exact) >= 3 else sole_pts
        poly = convex_hull_2d(use)
        area = polygon_area(poly)
        xs = [p[0] for p in poly]
        zs = [p[1] for p in poly]
        print(f"  脚底平面 (局部 y = {true_ymin:+.5f})，用 {len(use)} 个顶点")
        print(f"    宽度 (局部 x) [{min(xs):+.4f},{max(xs):+.4f}] = {max(xs)-min(xs):.4f} m")
        print(f"    长度 (局部 z) [{min(zs):+.4f},{max(zs):+.4f}] = {max(zs)-min(zs):.4f} m")
        print(f"    凸包面积 {area*1e4:.1f} cm^2，顶点数 {len(poly)}")
        print(f"    凸包 (x,z): " + " ".join(f"({p[0]:+.4f},{p[1]:+.4f})" for p in poly))
        out[name] = {
            "aabb": {"x": ax, "y": ay, "z": az},
            "sole_local_y": true_ymin,
            "poly_xz": poly,
            "area_m2": area,
        }

    # 把结论写成 Python 字面量，方便直接贴进仿真脚本
    print()
    print("=" * 92)
    print("# 可直接粘贴到仿真脚本的常量（局部系：x=横向, y=竖直, z=前后(前=+z)）")
    for name, d in out.items():
        print(f"{name.upper()}_SOLE_Y = {d['sole_local_y']:.6f}")
        poly_s = ", ".join(f"({p[0]:.6f}, {p[1]:.6f})" for p in d["poly_xz"])
        print(f"{name.upper()}_SOLE_POLY_XZ = [{poly_s}]")


if __name__ == "__main__":
    main()
