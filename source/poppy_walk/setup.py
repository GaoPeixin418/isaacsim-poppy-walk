# Copyright (c) 2026, Peixin Gao
# SPDX-License-Identifier: BSD-3-Clause
"""poppy_walk 扩展包安装脚本。

为什么不用 pyproject.toml 里的 [project] 段写依赖：
  本包是装在已经配好的 Isaac Lab conda 环境里的，torch / isaaclab / rsl-rl
  都已经由 D1 的安装流程提供。在这里重复声明依赖只会引入版本冲突风险。
  所以只声明"有哪些包"，用 pip install -e 装成本地可编辑包即可。
"""

from setuptools import find_packages, setup

setup(
    name="poppy_walk",
    version="0.1.0",
    description="Isaac Lab tasks for the Poppy humanoid robot (stand / walk)",
    author="Peixin Gao",
    python_requires=">=3.10",
    # 用 find_packages 自动发现：poppy_walk / assets / tasks / ... 都是包
    packages=find_packages(include=["poppy_walk", "poppy_walk.*"]),
    include_package_data=True,
    zip_safe=False,
)
