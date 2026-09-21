#!/bin/bash
# ============================================================================
# v8 一键接力：前进指令修正版训练 + 自动判卷 + 自动录侧面视频
# 背景：v1–v7 的前进指令发在基座 x 通道，而解剖学正前方 = 基座 +y（相机标定
#       实验 cam_calib.py 实证），v7 学出的是向右横移。v8 把指令换到 y 通道。
# 用法（服务器开机后，本文件先上传到 /data/poppy/poppy-walking/scripts/）：
#   setsid bash /data/poppy/poppy-walking/scripts/run_v8.sh < /dev/null > /dev/null 2>&1 &
#   # 然后随时：tail -f /data/poppy/poppy-walking/out/d5_train_v8.log
# 全程无人值守约 2~2.5 小时，产物：
#   out/d5_train_v8.log          训练日志
#   out/d5_eval_v8/metrics.json  判卷指标（重点看 new 字段 heading_error）
#   out/videos/…episode-1.mp4   正侧面跟拍视频
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data/poppy/envs/poppy/bin/python
RUN_BASE=logs/rsl_rl/poppy_walk

# ---- 训练前先确认环境数与 v7 一致（防止配置漂移） ----
V7_ENV_YAML=$(ls -d $RUN_BASE/2026-09-18_18-48-51/params/env.yaml 2>/dev/null | head -1)
if [ -n "$V7_ENV_YAML" ]; then
  V7_NUM_ENVS=$(grep -A1 '^num_envs:' "$V7_ENV_YAML" | grep -oE '[0-9]+' | head -1)
  echo "[v8] v7 num_envs = ${V7_NUM_ENVS:-unknown}"
fi
NUM_ENVS=${V7_NUM_ENVS:-2048}

# ---- 记录训练开始的最新 run 目录（判卷时只认这之后新建的目录） ----
BEFORE=$(ls -1d $RUN_BASE/2026-09-* 2>/dev/null | tail -1)

# ---- 1) 训练（全新 run，不 --resume；v7 的教训：resume 会叠加轮数+写新目录） ----
echo "[v8] training starts $(date), num_envs=$NUM_ENVS"
POPPY_PYTHON=$PY bash scripts/run.sh scripts/train_poppy.py --headless \
  --task Poppy-Walk-v0 --num_envs "$NUM_ENVS" --max_iterations 2500 \
  > out/d5_train_v8.log 2>&1
echo "[v8] training done $(date), exit=$?"

# ---- 2) 找本次训练的新 run 目录和最后一个 checkpoint ----
NEW_RUN=""
for d in $RUN_BASE/2026-09-*; do
  if [ -z "$BEFORE" ] || [ "$d" \> "$BEFORE" ]; then NEW_RUN="$d"; fi
done
CKPT=$(ls -1v "$NEW_RUN"/model_*.pt 2>/dev/null | tail -1)
echo "[v8] run dir = $NEW_RUN, checkpoint = $CKPT"
if [ -z "$CKPT" ]; then echo "[v8] ERROR: no checkpoint found"; exit 1; fi

# ---- 3) 判卷（Play 环境，注意 --headless 别漏） ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/d5_eval.py --headless \
  --task Poppy-Walk-Play-v0 --checkpoint "$CKPT" --out out/d5_eval_v8 \
  > out/d5_eval_v8_run.log 2>&1
echo "[v8] eval done $(date)"

# ---- 4) 抗扰（训练环境带推力） ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/d5_eval.py --headless \
  --task Poppy-Walk-v0 --checkpoint "$CKPT" --out out/d5_eval_v8_push \
  > out/d5_eval_v8_push_run.log 2>&1
echo "[v8] push eval done $(date)"

# ---- 5) 正侧面跟拍视频（headless 离屏渲染，--enable_cameras 必须带） ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/record_walk.py --headless --enable_cameras \
  --checkpoint "$CKPT" --seconds 12 --out out/videos \
  > out/record_v8.log 2>&1
echo "[v8] video done $(date) — ALL FINISHED, check:"
echo "[v8]   out/d5_eval_v8/metrics.json"
echo "[v8]   out/videos/"
