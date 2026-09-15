#!/bin/bash
# D2 Step 3 — Poppy URDF → USD 转换（Isaac Lab 官方 convert_urdf.py）
#
# 转换两个版本：
#   1) full    —— 保留全部 25 DOF。用途：核对"关节限位有没有被 importer 带进 USD"，
#                 这是我们诊断当年反关节问题的关键一步。
#   2) locked  —— 上身 15 关节已改为 fixed，配合 --merge-joints 合并成刚体，
#                 articulation 只剩双腿 10 DOF。这是训练实际使用的资产。
#
# 关于 --joint-stiffness / --joint-damping：
#   这两个值会被烘焙进 USD 的关节 drive 里。这里给的是按 Poppy 力矩量级
#   （MX-28 堵转 2.5 N·m）估的保守起始值，环境配置里还会按关节分别覆盖
#   （hip_y 承重需要更高的 kp）。绝对不能沿用默认的 100/1.0 ——
#   那是给几十上百 N·m 的大机器人用的量级，对 Poppy 会直接数值爆炸。

set -u
PROJ=/data/poppy
URDF_DIR=$PROJ/poppy-walking/assets/poppy/urdf
OUT_DIR=$PROJ/poppy-walking/assets/poppy
LAB=$PROJ/src/IsaacLab
PY=$PROJ/envs/poppy/bin/python

export TMPDIR=$PROJ/tmp
export PIP_CACHE_DIR=$PROJ/cache/pip
# Kit 启动前会交互式询问是否接受 NVIDIA Omniverse EULA。
# 非交互式执行（nohup/CI）必须用环境变量替代人工确认，否则直接 EOFError 崩掉。
# 这是 NVIDIA 官方为无头/CI 场景提供的开关，等价于在提示符下回答 Yes。
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
mkdir -p "$TMPDIR"

cd "$LAB" || { echo "FAILED: 找不到 IsaacLab 目录"; exit 1; }

echo "##### 转换 1/2: full（25 DOF，限位核对用） #####"
"$PY" scripts/tools/convert_urdf.py \
  "$URDF_DIR/Poppy_Humanoid.urdf" \
  "$OUT_DIR/poppy_full.usd" \
  --joint-stiffness 8.0 --joint-damping 0.3 --joint-target-type position \
  --headless 2>&1 | tail -45
echo "----- 转换 1 结束 -----"
echo

echo "##### 转换 2/2: locked（10 DOF，训练用） #####"
"$PY" scripts/tools/convert_urdf.py \
  "$URDF_DIR/Poppy_Humanoid_locked.urdf" \
  "$OUT_DIR/poppy.usd" \
  --merge-joints \
  --joint-stiffness 8.0 --joint-damping 0.3 --joint-target-type position \
  --headless 2>&1 | tail -45
echo "----- 转换 2 结束 -----"
echo

echo "##### 产物 #####"
ls -la "$OUT_DIR" | grep -i -E "usd|\.txt" || true
find "$OUT_DIR" -name "*.usd*" -printf "%s\t%p\n" 2>/dev/null | sort -rn | head -20

echo "D2_CONVERT_DONE"
