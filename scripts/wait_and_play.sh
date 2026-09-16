#!/usr/bin/env bash
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
#
# 等 D4 训练跑完 → 自动录「站立」和「行走」两段演示视频。
#
# 为什么要写这个脚本，而不是手动分两次跑：
#   同一台机器上跑两个 Isaac Sim 实例（一个训练 + 一个播放）会互相抢
#   Vulkan 渲染上下文和 PhysX 的 GPU 资源，实测第二个实例会卡在初始化、
#   吃满 CPU 却一点画面都不出，同时把训练速度从 2.2 s/轮拖到 4.4 s/轮。
#   所以必须【串行】：训练结束 → 释放显存 → 再录像。
#   既然是串行，就让机器自己等，不用人盯着。
#
# 用法（在仓库根目录）：bash scripts/wait_and_play.sh
set -u

cd /data/poppy/poppy-walking

echo "[wait_and_play] 等待训练结束 ... $(date +%H:%M:%S)"
while pgrep -f "[t]rain_poppy" > /dev/null; do sleep 20; done
echo "[wait_and_play] 训练已结束 $(date +%H:%M:%S)"

# 等显存彻底回收。Isaac Sim 退出后 CUDA context 释放有延迟，
# 立刻起下一个实例容易拿到一个半死不活的显存状态。
sleep 20
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# ---- 站立（16 个环境并排，看「静止时能不能稳住」）----
echo "[wait_and_play] 录制站立视频 ..."
PYTHONUNBUFFERED=1 bash scripts/run.sh scripts/play_poppy.py \
    --headless --video --num_envs 16 --video_length 400 > out/play_stand.log 2>&1
echo "[wait_and_play] 站立 exit=$? $(date +%H:%M:%S)"

sleep 20

# ---- 行走（录长一点，看完整步态周期）----
echo "[wait_and_play] 录制行走视频 ..."
PYTHONUNBUFFERED=1 bash scripts/run.sh scripts/play_poppy.py \
    --headless --video --num_envs 16 --video_length 600 \
    --task Poppy-Walk-Play-v0 > out/play_walk.log 2>&1
echo "[wait_and_play] 行走 exit=$? $(date +%H:%M:%S)"

echo "[wait_and_play] === 产出视频 ==="
find logs -path "*videos*" -name "*.mp4" -exec ls -la {} \;
echo "[wait_and_play] 全部完成 $(date +%H:%M:%S)"
