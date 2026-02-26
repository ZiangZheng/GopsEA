#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

from setuptools import find_packages, setup

setup(
    name="gops_tasks",
    version="0.0.0",
    packages=find_packages(),
    author="Zaterval | ZiangZheng",
    maintainer="Ziang Zheng",
    maintainer_email="ziang_zheng@foxmail.com",
    url="https://github.com/ZiangZheng",
    license="BSD-3",
    description="",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
    ],
)
