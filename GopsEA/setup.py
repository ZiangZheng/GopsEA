from setuptools import find_packages, setup

setup(
    name="GopsEA",
    version="0.0.0",
    packages=find_packages(),
    author="Zaterval(interval-package) | ZiangZheng",
    maintainer="Ziang Zheng",
    maintainer_email="ziang_zheng@foxmail.com",
    url="https://github.com",
    license="BSD-3",
    description="GopsEA",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
    ],
)
