#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Poppy 项目统一运行入口
#
# 为什么需要它：
#   1. Isaac Sim 首次启动会交互式询问 EULA，headless 下直接 EOFError 崩掉
#      → 必须预置 OMNI_KIT_ACCEPT_EULA=YES
#   2. conda 环境在数据盘 /data/poppy/envs/poppy，每次都要 source + activate
#   3. 脚本要以仓库根目录为工作目录，否则 out/ 之类的相对路径会跑到别处
#
# 用法：
#   bash scripts/run.sh scripts/d3_rest_pose_probe.py --mode sweep --headless
#   bash scripts/run.sh -m isaaclab.train ...          # 直接跑 Isaac Lab 训练入口
# ---------------------------------------------------------------------------
set -euo pipefail

export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONUNBUFFERED=1

# 注：~/.cache 在 D1 已经软链到 /data/poppy/cache_real，缓存不会落到 59GB 的系统盘，
# 所以这里不要再去覆盖 XDG_CACHE_HOME —— 多一层覆盖只会多一个出错点。

# 可移植性：设了 POPPY_PYTHON（本地/其他机器）就直接用它，跳过服务器的 conda。
# 本地用法示例（Isaac Lab 自带的 python）：
#   POPPY_PYTHON=<IsaacLab>/isaaclab.sh 会展开成 python -p ... 不方便，
#   实际上本地推荐：POPPY_PYTHON=<IsaacLab>/_isaac_sim/python.exe bash scripts/run.sh ...
if [ -n "${POPPY_PYTHON:-}" ]; then
  cd "$(dirname "$0")/.."
  exec "$POPPY_PYTHON" "$@"
fi

# 服务器默认：数据盘 conda 环境
source /miniconda3/etc/profile.d/conda.sh
conda activate poppy

cd "$(dirname "$0")/.."
exec python "$@"
