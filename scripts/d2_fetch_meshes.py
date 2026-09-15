#!/usr/bin/env python3
"""D2 Step 1 — 从 poppy-humanoid 官方仓库拉取 Poppy 人形机器人的 STL 网格。

背景：官方 URDF 里引用的网格写的是 `package://meshes/xxx.STL`。
`package://` 是 ROS 的包解析协议，Isaac Sim 不认，所以必须：
  1) 把网格文件下载到本地
  2) 把 URDF 里的路径改写成相对路径
否则 USD 转换会丢掉所有几何体（表现为导入后是一个空壳）。

用法：
  python d2_fetch_meshes.py --probe          # 只列出仓库里的模型文件清单，不下载
  python d2_fetch_meshes.py                  # 下载 STL 到目标目录
  python d2_fetch_meshes.py --dest /path     # 指定目标目录
"""
import argparse
import collections
import json
import os
import sys
import urllib.request

REPO = "poppy-project/poppy-humanoid"
REF = "master"
DEFAULT_DEST = "/data/poppy/poppy-walking/assets/poppy/urdf/meshes"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="只列清单，不下载")
    ap.add_argument("--dest", default=DEFAULT_DEST)
    args = ap.parse_args()

    api = f"https://api.github.com/repos/{REPO}/git/trees/{REF}?recursive=1"
    tree = json.loads(fetch(api).decode())
    if tree.get("truncated"):
        print("WARNING: GitHub 返回的 tree 被截断了，清单可能不完整", file=sys.stderr)
    paths = [t["path"] for t in tree["tree"]]

    stl = sorted(p for p in paths if p.lower().endswith(".stl"))
    print(f"仓库文件总数: {len(paths)}    STL 数量: {len(stl)}")

    print("\n--- STL 所在目录 ---")
    for d, c in sorted(collections.Counter(p.rsplit("/", 1)[0] for p in stl).items()):
        print(f"  {d}   ({c} 个)")

    print("\n--- 其他模型格式 ---")
    for ext in (".dae", ".obj", ".ply", ".stl"):
        print(f"  {ext}: {sum(1 for p in paths if p.lower().endswith(ext))}")

    print("\n--- URDF / xacro ---")
    for p in paths:
        if p.lower().endswith((".urdf", ".xacro")):
            print("  ", p)

    if args.probe:
        print("\n[probe 模式] 未下载任何文件")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    ok = fail = 0
    names = []
    for p in stl:
        url = f"https://raw.githubusercontent.com/{REPO}/{REF}/{p}"
        out = os.path.join(args.dest, os.path.basename(p))
        try:
            with open(out, "wb") as f:
                f.write(fetch(url))
            ok += 1
            names.append(os.path.basename(p))
        except Exception as e:  # noqa: BLE001
            print("FAIL", p, e)
            fail += 1

    total = sum(
        os.path.getsize(os.path.join(args.dest, f)) for f in os.listdir(args.dest)
    )
    print(f"\n下载完成: 成功 {ok} / 失败 {fail}")
    print(f"目标目录: {args.dest}   总大小: {total / 1e6:.2f} MB")

    print("\n--- 文件名模式统计 ---")
    pat = collections.Counter()
    for n in names:
        low = n.lower()
        if "_visual" in low:
            pat["*_visual.STL（视觉网格）"] += 1
        elif "_respondable" in low:
            pat["*_respondable.STL（碰撞网格）"] += 1
        else:
            pat["其他（可能是通用网格）"] += 1
    for k, v in pat.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
