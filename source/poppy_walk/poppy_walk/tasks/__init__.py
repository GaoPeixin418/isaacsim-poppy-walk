# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""任务注册。导入本模块即完成 gym.register。"""

from . import manager_based  # noqa: F401

__all__ = ["manager_based"]
