#!/bin/bash
# ============================================================================
# v9: fine-tune v8 -> true forward walking (command sign flip to -y)
#
# Why: URDF FK (knee protrudes toward base -y in default pose) + v8 video
# frames prove anatomical forward = base -y. cam_calib's eyeball call of
# "+y = front" was wrong; v8 walked backward at 0.234 m/s. Zero-cost test
# (v8 policy under vy=-0.25 command): 100% survival but mean_vy=+0.045 m/s
# (marching in place) -> no sign generalization, fine-tune required.
#
# Plan: resume from v8 model_7498, +2500 iters (rsl_rl adds to loaded
# iteration -> ~9999), then Play eval + push eval + headless follow-cam video.
#
# Usage (single Isaac instance at a time!):
#   setsid bash /data/poppy/poppy-walking/scripts/run_v9.sh < /dev/null > /dev/null 2>&1 &
#   tail -f /data/poppy/poppy-walking/out/d5_train_v9.log
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data/poppy/envs/poppy/bin/python
RUN_BASE=logs/rsl_rl/poppy_walk

LOAD_RUN=2026-09-21_17-46-25
LOAD_CKPT=model_7498.pt

# ---- num_envs: reuse the value v7/v8 used (from v7 env.yaml) ----
V7_ENV_YAML=$(ls -d $RUN_BASE/2026-09-18_18-48-51/params/env.yaml 2>/dev/null | head -1)
if [ -n "$V7_ENV_YAML" ]; then
  V7_NUM_ENVS=$(grep -A1 '^num_envs:' "$V7_ENV_YAML" | grep -oE '[0-9]+' | head -1)
  echo "[v9] num_envs = ${V7_NUM_ENVS:-unknown}"
fi
NUM_ENVS=${V7_NUM_ENVS:-2048}

BEFORE=$(ls -1d $RUN_BASE/2026-09-* 2>/dev/null | tail -1)

# ---- 1) fine-tune: resume from v8 final checkpoint, +2500 iters ----
echo "[v9] fine-tune starts $(date), num_envs=$NUM_ENVS, resume $LOAD_RUN/$LOAD_CKPT"
POPPY_PYTHON=$PY bash scripts/run.sh scripts/train_poppy.py --headless \
  --task Poppy-Walk-v0 --num_envs "$NUM_ENVS" --max_iterations 2500 \
  --resume --load_run "$LOAD_RUN" --checkpoint "$LOAD_CKPT" \
  > out/d5_train_v9.log 2>&1
echo "[v9] training done $(date), exit=$?"

# ---- 2) locate newest checkpoint (resume may write same dir or new dir) ----
CKPT=$(ls -1v $RUN_BASE/$LOAD_RUN/model_*.pt 2>/dev/null | tail -1)
if [ -z "$CKPT" ] || [ "$CKPT" = "$RUN_BASE/$LOAD_RUN/$LOAD_CKPT" ]; then
  NEW_RUN=""
  for d in $RUN_BASE/2026-09-*; do
    if [ -z "$BEFORE" ] || [ "$d" \> "$BEFORE" ]; then NEW_RUN="$d"; fi
  done
  if [ -n "$NEW_RUN" ]; then
    CKPT=$(ls -1v "$NEW_RUN"/model_*.pt 2>/dev/null | tail -1)
  fi
fi
echo "[v9] checkpoint = $CKPT"
if [ -z "$CKPT" ]; then echo "[v9] ERROR: no checkpoint found"; exit 1; fi

# ---- 3) Play eval (undisturbed, cmd vy=-0.25) ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/d5_eval.py --headless \
  --task Poppy-Walk-Play-v0 --checkpoint "$CKPT" --vx -0.25 --out out/d5_eval_v9 \
  > out/d5_eval_v9_run.log 2>&1
echo "[v9] eval done $(date)"

# ---- 4) push eval (random pushes) ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/d5_eval.py --headless \
  --task Poppy-Walk-v0 --checkpoint "$CKPT" --vx -0.25 --out out/d5_eval_v9_push \
  > out/d5_eval_v9_push_run.log 2>&1
echo "[v9] push eval done $(date)"

# ---- 5) follow-cam video (headless offscreen) ----
POPPY_PYTHON=$PY bash scripts/run.sh scripts/record_walk.py --headless --enable_cameras \
  --checkpoint "$CKPT" --vx -0.25 --seconds 12 --out out/videos \
  > out/record_v9.log 2>&1
echo "[v9] video done $(date) -- ALL FINISHED, check:"
echo "[v9]   out/d5_eval_v9/metrics.json"
echo "[v9]   out/videos/"
