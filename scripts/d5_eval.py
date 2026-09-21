#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""D5 策略评估：定量回答「它到底是真在走，还是在原地蹭」。

================================================================================
为什么光看"平均回报"不够
================================================================================
训练日志里的 `Mean reward` 是一个把 13 项奖励加权求和后的标量。这个数字涨了，
只能说明"策略更会拿分了"，**不能说明它学会了走路**。

D4 第一次训练就是活例子：回报从 -0.48 涨到 22.35，看起来一片大好，
但那 22 分里有 1.0 分/秒是"站在指令位置上不倒下"换来的。策略完全可以
【一步都不迈】，靠身体前后倾 + 脚下蹭地去糊弄速度跟踪 —— 回报照样涨。

所以评估必须回到**物理量**，而不是奖励量。本脚本记录四组信号：

  1. 足端高度时间序列 (foot height)
     —— 两脚是否交替抬离地面？峰谷差有多大？周期多长？
        判据：左右脚高度序列应当【反相】，峰谷差 ≳ 2 cm。
  2. 足底接触力时间序列 (contact force)
     —— 是否出现「左脚离地 / 右脚着地」的交替？还是两只脚一直压着地面？
  3. 基座实际速度 vs 指令速度
     —— 跟踪误差。注意要看【实际位移/时间】，不能只看奖励里的归一化值。
  4. 关节角时间序列
     —— 髋/膝/踝是否呈现周期性摆动？还是几乎一条直线（=没迈腿）？

================================================================================
用法
================================================================================
    # 测行走策略能不能以 0.2 m/s 连续走 12 s（里程碑 2 的验收条件）
    bash scripts/run.sh scripts/d5_eval.py --headless \
        --task Poppy-Walk-Play-v0 --vx 0.2 --duration 12

    # 测站立策略（指令速度 0）
    bash scripts/run.sh scripts/d5_eval.py --headless \
        --task Poppy-Stand-Play-v0 --vx 0.0 --duration 12

    # 带域随机化测（鲁棒性：有观测噪声 + 随机推力）
    bash scripts/run.sh scripts/d5_eval.py --headless \
        --task Poppy-Walk-v0 --vx 0.2 --duration 12

产出（默认写到 out/eval/<时间戳>/）：
    metrics.json    数值指标
    traces.png      四组时间序列图
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

# poppy_walk 的导入必须在 AppLauncher 之前，且它是「轻量」的：
# 只做 gym.register，env_cfg / agent_cfg 都是字符串延迟导入，
# 所以不会提前拉起 omni / pxr（详见 scripts/train_poppy.py 的说明）。
_SRC = os.path.join(_REPO, "source", "poppy_walk")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import poppy_walk  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Evaluate a trained Poppy policy.")
parser.add_argument("--task", type=str, default="Poppy-Walk-Play-v0")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="不传则自动挑 logs/rsl_rl/<实验名>/ 下训练轮数最大的")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--vx", type=float, default=0.2, help="指令前进速度 m/s")
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--wz", type=float, default=0.0, help="指令偏航角速度 rad/s")
parser.add_argument("--duration", type=float, default=12.0, help="评估时长（秒，仿真时间）")
parser.add_argument("--out", type=str, default=None, help="输出目录")
parser.add_argument("--stochastic", action="store_true",
                    help="用训练同款采样动作（均值+噪声），而非确定性均值动作")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# 以下才允许 import isaaclab / torch（Omniverse 的硬性约束）
# ---------------------------------------------------------------------------
import glob  # noqa: E402
import re  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def find_checkpoint(task: str) -> str:
    """按训练轮数找最大的 model_*.pt（不用"最新目录"，见 play_poppy.py 的说明）。"""
    exp = "poppy_walk" if "Walk" in task else "poppy_stand"
    root = os.path.join(_REPO, "logs", "rsl_rl", exp)
    best = None
    for path in glob.glob(os.path.join(root, "*", "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        if m:
            n = int(m.group(1))
            if best is None or n > best[0]:
                best = (n, path)
    if best is None:
        raise SystemExit(f"在 {root} 下没找到 model_*.pt")
    print(f"[eval] 检查点: {os.path.relpath(best[1], _REPO)}（第 {best[0]} 轮）")
    return best[1]


def pin_command(env, vx: float, vy: float, wz: float) -> None:
    """把速度指令钉死成想要的值。

    为什么要动私有属性 `_resampling_time_range`：
      VelocityCommand 每隔一段随机时间会【重采样】指令（训练时这是必须的，
      否则策略只会记住一个速度）。但评估时我们要的是一个恒定输入，
      而每步的 command 都是 manager 在 env.step() 内部算出来的 ——
      在外部赋值会被下一步的 compute() 覆盖。所以直接把重采样周期设成无限大，
      这样就只有我们写进去的那一个指令。

    v8 坐标系更正（2026-09-21）：解剖学正前方 = 基座 +y（相机标定实验实证）。
    --vx 参数语义改为"前进速度"，实际写进指令的 y 通道；x 通道（解剖学横向）锁 0。
    v1–v7 曾把它钉在 x 通道，等于让策略横着走——v8 修正。
    """
    term = env.unwrapped.command_manager.get_term("base_velocity")
    term._resampling_time_range = (1e9, 1e9)  # 永不重采样
    term.command[:, 0] = vx  # 解剖学横向（应恒为 0）
    term.command[:, 1] = vy  # 解剖学前向
    term.command[:, 2] = wz
    # 同时关掉"随机站立的那些环境"（它们会被强行置零指令）
    if hasattr(term, "cfg") and hasattr(term.cfg, "rel_standing_envs"):
        term.cfg.rel_standing_envs = 0.0


def main() -> None:
    ckpt = args.checkpoint or find_checkpoint(args.task)
    out_dir = args.out or os.path.join(
        _REPO, "out", "eval", f"{args.task}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ---- 环境 ----
    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    # 评估时把单集时长放宽到比评估时长更长，这样一次 reset 就能覆盖全程，
    # 不会因为 time_out 把 episode 切成几段而干扰统计。
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.duration + 5.0)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
    runner.load(ckpt)
    if args.stochastic:
        # 训练同款：ActorCritic.act() 内部按当前 std 采样（正态噪声），
        # 用来判别"均值动作坍缩成单腿模式、采样动作才是正常步态"的可能
        ac = getattr(runner.alg, "policy", None) or getattr(runner.alg, "actor_critic")
        policy = ac.act
    else:
        policy = runner.get_inference_policy(device=DEVICE)

    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]

    # 找脚部刚体在 body 列表里的下标
    foot_ids, foot_names = robot.find_bodies(".*foot")
    print(f"[eval] 脚部刚体: {foot_names} (ids={foot_ids})")

    # 注意: 接触力传感器的 body 顺序 ≠ 机器人本体的 body 顺序!
    # net_forces_w 的列按传感器自己跟踪的 body 列表排(含 pelvis/大腿/小腿等全部刚体),
    # 直接拿 robot 的 body id 去索引会把 r_shin 当成 l_foot(v7 判卷误诊事故, 2026-09-20:
    # 左脚"全程 0N 悬空"其实是右小腿不受力, 实际机器人在正常交替双脚)。
    # 取力必须用 sensor.find_bodies 得到传感器自己的下标。
    contact_sensor = unwrapped.scene.sensors["contact_forces"]
    force_ids, force_names = contact_sensor.find_bodies(foot_names)
    print(f"[eval] 传感器力下标: {force_names} (ids={force_ids})")

    # ---- 起步：让默认姿态先稳住，再钉指令 ----
    # 直接给非零指令，策略会从"站立默认姿态"突然被要求"往前走"，
    # 头几步是瞬态，统计进去会污染数据。先空跑 0.5 s 让它进入节律。
    obs, _ = env.get_observations()
    warm_up_steps = int(0.5 / unwrapped.step_dt)
    for _ in range(warm_up_steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)

    # v8 坐标系更正：--vx 语义 = 前进速度，写进 y 通道；x（解剖学横向）恒 0
    pin_command(unwrapped, 0.0, args.vx, args.wz)
    obs, _ = env.get_observations()

    # ---- 记录 ----
    n_steps = int(args.duration / unwrapped.step_dt)
    print(f"[eval] 仿真步长 {unwrapped.step_dt} s → 记录 {n_steps} 步（{args.duration} s）")

    rec: dict[str, list] = {
        "t": [], "foot_z": [], "foot_force": [], "base_lin_vel": [],
        "base_height": [], "joint_pos": [], "done": [], "base_xy": [],
    }

    t0 = time.time()
    for i in range(n_steps):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        rec["t"].append(i * unwrapped.step_dt)
        # body_pos_w: (N, B, 3)，取世界坐标 z 看脚抬多高
        rec["foot_z"].append(robot.data.body_pos_w[:, foot_ids, 2].cpu().numpy())
        # 接触力范数：0 表示离地（必须用传感器自己的 body 下标, 见上方注释）
        force = contact_sensor.data.net_forces_w[:, force_ids, :]
        rec["foot_force"].append(force.norm(dim=-1).cpu().numpy())
        # 基座速度（基座系）→ 取 x 前进方向
        rec["base_lin_vel"].append(robot.data.root_lin_vel_b[:, :3].cpu().numpy())
        rec["base_height"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
        rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy())
        rec["base_xy"].append(robot.data.root_pos_w[:, :2].cpu().numpy())
        rec["done"].append(dones.cpu().numpy().astype(bool))
        if i % 50 == 0:
            sim_t = i * unwrapped.step_dt
            print(f"  t={sim_t:5.1f}s  已终结 {int(np.sum(np.concatenate(rec['done'])))} 次")

    wall = time.time() - t0

    # ---- 统计 ----
    t = np.array(rec["t"])
    foot_z = np.array(rec["foot_z"])              # (T, N, 2)
    foot_f = np.array(rec["foot_force"])          # (T, N, 2)
    blv = np.array(rec["base_lin_vel"])           # (T, N, 3)
    h = np.array(rec["base_height"])              # (T, N)
    jp = np.array(rec["joint_pos"])               # (T, N, D)
    base_xy = np.array(rec["base_xy"])            # (T, N, 2)
    done = np.array(rec["done"])                  # (T, N)

    alive = ~done.cumsum(axis=0).astype(bool)     # 还没摔的那些步
    metrics = {
        "task": args.task,
        "checkpoint": os.path.relpath(ckpt, _REPO),
        "command": {"forward_vy": args.vx, "lateral_vx": 0.0, "wz": args.wz},
        "duration_s": args.duration,
        "num_envs": args.num_envs,
        "survival": {
            "mean_alive_time_s": float((alive.sum(axis=0) * unwrapped.step_dt).mean()),
            "min_alive_time_s": float((alive.sum(axis=0) * unwrapped.step_dt).min()),
            "frac_survived_full": float(alive[-1].mean()),
        },
        "velocity": {
            "note": "基座坐标系下的【瞬时】速度；v8 起前进方向 = 基座 +y（相机标定实证）",
            "cmd_forward": args.vx,
            "mean_vy": float(blv[..., 1][alive].mean()),
            "std_vy": float(blv[..., 1][alive].std()),
            "mean_vx_lateral": float(blv[..., 0][alive].mean()),
            "std_vx_lateral": float(blv[..., 0][alive].std()),
        },
        "base_height": {
            "mean": float(h[alive].mean()),
            "std": float(h[alive].std()),
        },
        "gait": {},
        "joint_range_rad": {
            name: float(jp[..., i][alive].max() - jp[..., i][alive].min())
            for i, name in enumerate(robot.joint_names)
        },
        "wall_time_s": wall,
    }

    # ---- 净位移速度：识破"原地摆动" ----
    # 基座系瞬时速度的平均 ≠ 平均前进速度。身体一前一后摆动时，
    # 前后两个半周期会互相抵消，让"平均速度"看起来接近 0，
    # 但人其实在往前挪。所以要用「走过的净路程 ÷ 用时」来当判据。
    dx = []
    for e in range(args.num_envs):
        idx = np.where(alive[:, e])[0]
        if idx.size >= 2:
            dx.append(float(np.linalg.norm(base_xy[idx[-1], e] - base_xy[idx[0], e])))
        else:
            dx.append(0.0)
    dx = np.array(dx)
    t_end = alive.sum(axis=0) * unwrapped.step_dt
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_v = np.where(t_end > 0, dx / t_end, 0.0)
    metrics["displacement"] = {
        "note": "净位移(米) / 存活时间(秒)，用来识破『原地摆动假装在走』",
        "mean_speed_m_s": float(disp_v.mean()),
        "mean_distance_m": float(dx.mean()),
        "per_env_speed": [float(v) for v in disp_v],
    }

    # ---- 步态指标：脚抬多高、左右是否反相 ----
    for j, name in enumerate(foot_names):
        z = foot_z[:, :, j][alive]
        f = foot_f[:, :, j][alive]
        metrics["gait"][name] = {
            "z_peak_to_peak_m": float(z.max() - z.min()) if z.size else float("nan"),
            "z_std_m": float(z.std()) if z.size else float("nan"),
            "air_ratio": float((f < 1.0).mean()) if f.size else float("nan"),
            "force_mean_N": float(f.mean()) if f.size else float("nan"),
        }
    if len(foot_names) == 2:
        a = foot_z[:, :, 0][alive].flatten()
        b = foot_z[:, :, 1][alive].flatten()
        n = min(a.size, b.size)
        if n > 10:
            metrics["gait"]["corr_left_right_z"] = float(np.corrcoef(a[:n], b[:n])[0, 1])

    # 关节角的"活动量"：走的策略一定比站的策略动得多得多
    metrics["joint_activity"] = {
        name: float(jp[..., i][alive].std()) for i, name in enumerate(robot.joint_names)
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2, ensure_ascii=False)

    # ---- 出图 ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n_env = min(3, args.num_envs)  # 只画前 3 个环境，否则线太乱
        alive_np = alive
        fig, axes = plt.subplots(5, 1, figsize=(13, 16), sharex=True)

        ax = axes[0]
        ax.set_title(f"{args.task}  cmd vx={args.vx} m/s  ({metrics['checkpoint']})", fontsize=11)
        for j, name in enumerate(foot_names):
            for e in range(n_env):
                ax.plot(t, np.where(alive_np[:, e], foot_z[:, e, j], np.nan),
                        lw=1.0, alpha=0.85, label=f"{name} env{e}")
        ax.set_ylabel("foot height z [m]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        ax = axes[1]
        for j, name in enumerate(foot_names):
            for e in range(n_env):
                ax.plot(t, np.where(alive_np[:, e], foot_f[:, e, j], np.nan), lw=1.0, alpha=0.85)
        ax.set_ylabel("foot contact force [N]")
        ax.grid(alpha=0.3)

        ax = axes[2]
        for e in range(n_env):
            ax.plot(t, np.where(alive_np[:, e], blv[:, e, 1], np.nan), lw=1.0, label=f"env{e} vy")
            ax.plot(t, np.where(alive_np[:, e], blv[:, e, 0], np.nan), lw=0.6, alpha=0.4, color="gray")
        ax.axhline(args.vx, color="k", ls="--", lw=1, label="command (forward)")
        ax.set_ylabel("base vy [m/s] (gray=vx drift)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

        ax = axes[3]
        for e in range(n_env):
            ax.plot(t, np.where(alive_np[:, e], h[:, e], np.nan), lw=1.0)
        ax.set_ylabel("base height [m]")
        ax.grid(alpha=0.3)

        ax = axes[4]
        for i, name in enumerate(robot.joint_names):
            if "hip_y" in name or "knee_y" in name or "ankle_y" in name:
                ax.plot(t, np.where(alive_np[:, 0], jp[:, 0, i], np.nan), lw=1.2,
                        label=f"{name} env0")
        ax.set_ylabel("pitch joints [rad]")
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "traces.png"), dpi=110)
        print(f"[eval] 图已保存: {os.path.join(out_dir, 'traces.png')}")
    except Exception as e:  # noqa: BLE001
        print(f"[eval] 画图失败（不影响指标）: {e}")

    # ---- 打印摘要 ----
    s = metrics["survival"]
    print("\n" + "=" * 68)
    print(f"任务            : {args.task}")
    print(f"指令速度        : 前进 vy={args.vx} m/s（基座 +y = 解剖学前向）")
    print(f"存活            : 平均 {s['mean_alive_time_s']:.2f} s / 最短 {s['min_alive_time_s']:.2f} s"
          f"  （跑满全程的环境比例 {s['frac_survived_full'] * 100:.1f}%）")
    print(f"实际前进速度    : {metrics['velocity']['mean_vy']:.3f} ± {metrics['velocity']['std_vy']:.3f} m/s"
          f"  （指令 {args.vx}，基座系瞬时 vy；横向漂移 vx = "
          f"{metrics['velocity']['mean_vx_lateral']:.3f}）")
    d = metrics["displacement"]
    print(f"净位移速度      : {d['mean_speed_m_s']:.3f} m/s"
          f"（{d['mean_distance_m']:.2f} m / {s['mean_alive_time_s']:.1f} s）")
    print(f"基座高度        : {metrics['base_height']['mean']:.4f} ± {metrics['base_height']['std']:.4f} m")
    for name, g in metrics["gait"].items():
        if isinstance(g, dict):
            print(f"  {name:14s} 抬脚峰谷差 {g['z_peak_to_peak_m'] * 100:5.2f} cm"
                  f"  离地时间占比 {g['air_ratio'] * 100:5.1f}%"
                  f"  平均接触力 {g['force_mean_N']:6.2f} N")
    if "corr_left_right_z" in metrics["gait"]:
        c = metrics["gait"]["corr_left_right_z"]
        print(f"  左右脚高度相关系数: {c:+.3f}  "
              f"（{'反相 → 在交替迈步 ✔' if c < -0.2 else '同相 → 大概率没在迈步 ✘'}）")
    print(f"墙钟耗时        : {wall:.1f} s")
    print("=" * 68)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
