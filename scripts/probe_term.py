#!/usr/bin/env python3
# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""v7 single_leg_lean 终止项探针。

背景: v7 评估显示左脚全程 0 N 受力但 12 s 未触发终止, 说明终止项实现有 bug。
本脚本加载训练好的策略, 在 Play 环境里逐帧打印终止项的全部中间量,
用于定位断点: fz / EMA / 体重 / loading_ratio / warmup / 项返回值 / done。

用法:
    bash scripts/run.sh scripts/probe_term.py --headless \
        --checkpoint logs/rsl_rl/poppy_walk/2026-09-18_18-48-51/model_2400.pt
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_SRC = os.path.join(_REPO, "source", "poppy_walk")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)
import poppy_walk  # noqa: E402,F401

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="Probe single_leg_lean termination internals.")
parser.add_argument("--task", type=str, default="Poppy-Walk-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--duration", type=float, default=12.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# omni 就绪之后才能 import isaaclab / torch
# ---------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry  # noqa: E402

try:
    from poppy_walk.tasks.manager_based.common import FOOT_BODIES
except ImportError:  # 兜底: 直接从模块里取
    from poppy_walk.tasks.manager_based.common.poppy_env_cfg import FOOT_BODIES  # type: ignore

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device=DEVICE, num_envs=args.num_envs)
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.duration + 5.0)
    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=None)

    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=DEVICE)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=DEVICE)

    uw = env.unwrapped
    robot = uw.scene["robot"]
    sensor = uw.scene.sensors["contact_forces"]

    # ---- 索引口径审计 ----
    print("[probe] FOOT_BODIES =", FOOT_BODIES, flush=True)
    try:
        print("[probe] sensor tracked body_names =", sensor.body_names, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[probe] sensor.body_names not available:", e, flush=True)
    sensor_foot_ids = None
    try:
        sensor_foot_ids, sensor_foot_names = sensor.find_bodies(FOOT_BODIES)
        print("[probe] sensor.find_bodies ids =", sensor_foot_ids, sensor_foot_names, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[probe] sensor.find_bodies failed:", e, flush=True)
    robot_foot_ids, robot_foot_names = robot.find_bodies(".*foot")
    print("[probe] robot.find_bodies ids =", robot_foot_ids, robot_foot_names, flush=True)
    try:
        print("[probe] termination active terms =", list(uw.termination_manager.active_terms), flush=True)
    except Exception as e:  # noqa: BLE001
        print("[probe] active_terms not available:", e, flush=True)

    # 终止项内部使用的 body_ids (由 manager 解析后存在 cfg 里)
    term_body_ids = None
    try:
        for name in uw.termination_manager.active_terms:
            if "lean" in name:
                cfg = getattr(uw.termination_manager.cfg, name, None)
                if cfg is not None and "sensor_cfg" in cfg.params:
                    term_body_ids = cfg.params["sensor_cfg"].body_ids
                    print(f"[probe] term '{name}' sensor_cfg.body_ids =", term_body_ids, flush=True)
    except Exception as e:  # noqa: BLE001
        print("[probe] term cfg introspection failed:", e, flush=True)

    obs, _ = env.get_observations()
    n_steps = int(args.duration / uw.step_dt)
    done_count = 0
    print(f"[probe] step_dt={uw.step_dt} n_steps={n_steps}", flush=True)

    for i in range(n_steps):
        actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        done_count += int(dones.sum().item())
        if i % 25 == 0 or i == n_steps - 1:
            ids = term_body_ids if term_body_ids is not None else (
                sensor_foot_ids if sensor_foot_ids is not None else robot_foot_ids
            )
            fz = sensor.data.net_forces_w[:, ids, 2]
            ema = getattr(uw, "_v7_load_ema_term", None)
            w = getattr(uw, "_v7_body_weight", None)
            try:
                tval = uw.termination_manager.get_term("single_leg_lean")
            except Exception:
                tval = None
            e0 = int(uw.episode_length_buf[0].item())
            line = (
                f"step {i:4d} t={i * uw.step_dt:5.2f}s ep_len={e0:4d} "
                f"fzL={fz[0, 0].item():7.3f} fzR={fz[0, 1].item():7.3f} "
            )
            if ema is not None and w is not None and w[0].item() > 0:
                line += (
                    f"emaL={ema[0, 0].item():7.3f} emaR={ema[0, 1].item():7.3f} "
                    f"W={w[0].item():7.3f} "
                    f"ratioL={ema[0, 0].item() / w[0].item():6.3f} "
                    f"ratioR={ema[0, 1].item() / w[0].item():6.3f} "
                )
            else:
                line += f"ema={None if ema is None else 'set'} w={None if w is None else w[0].item()} "
            if tval is not None:
                line += f"term0={bool(tval[0].item())} "
            line += f"done0={bool(dones[0].item())} dones_total={done_count}"
            print(line, flush=True)

    print("[probe] total dones over", n_steps, "steps:", done_count, flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
