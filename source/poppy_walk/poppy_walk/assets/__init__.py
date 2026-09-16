# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""Poppy 机器人资产配置。"""

from .poppy import (  # noqa: F401
    POPPY_CFG,
    STANDING_PELVIS_Z,
    WALK_ANKLE_Y,
    WALK_HIP_Y,
    WALK_KNEE_K,
    WALK_PELVIS_Z,
    walk_default_joint_pos,
)

__all__ = [
    "POPPY_CFG",
    "STANDING_PELVIS_Z",
    "WALK_ANKLE_Y",
    "WALK_HIP_Y",
    "WALK_KNEE_K",
    "WALK_PELVIS_Z",
    "walk_default_joint_pos",
]
