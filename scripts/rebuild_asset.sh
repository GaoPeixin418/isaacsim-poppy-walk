#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# 一条命令重建 Poppy 资产（URDF 修补 → USD 转换 → 关节核对）
#
# 为什么要做成脚本：D2/D3 阶段反复发现资产问题（大小写、轴向、限位），
# 每次都要"重来一遍"。把它固化成一个入口，才能做到
#   「改一处 → 一条命令重建 → 结果可复现」，
# 这也是简历里"可复现训练仓库"那句话的底气。
#
# 前置条件：assets/poppy/urdf/Poppy_Humanoid_orig.urdf 与 meshes/ 已存在
#   若 meshes 缺失，先跑：python scripts/d2_fetch_meshes.py
#
# 用法：
#   bash scripts/rebuild_asset.sh              # 含镜像轴向修正（推荐）
#   bash scripts/rebuild_asset.sh --no-axis-fix   # 保留官方 bug，做对照实验
# ---------------------------------------------------------------------------
set -euo pipefail

# 仓库位置从脚本自身推导（不写死，本地/服务器都能用）
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URDF_DIR=$REPO/assets/poppy/urdf
AXIS_FLAG="${1:-}"

cd "$REPO"

echo "############ 步骤 1/3：URDF 修补 ############"
bash scripts/run.sh scripts/d2_patch_urdf.py --mode lockupper \
  --urdf-dir "$URDF_DIR" ${AXIS_FLAG}

echo
echo "############ 步骤 2/3：URDF → USD 转换 ############"
bash scripts/d2_convert.sh

echo
echo "############ 步骤 3/3：USD 关节限位核对 ############"
bash scripts/run.sh scripts/d2_inspect_usd.py 2>&1 | tail -40 || true

echo
echo "ASSET_REBUILD_DONE"
