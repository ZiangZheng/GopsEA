# GopsEA

A reinforcement learning framework for robot control tasks, built on GOPS (General Optimal control Problem Solver) and integrated with Isaac Lab.

## Features

- 🤖 Robot control tasks (quadruped robots, manipulators, etc.)
- 🚀 Isaac Lab integration for high-performance physics simulation
- 📦 Modular architecture for easy extension
- 🔄 On-policy and off-policy algorithms support
- 📊 Built-in logging and visualization

## Requirements

- Python >= 3.8
- PyTorch >= 1.10.0
- Isaac Lab >= 0.21.0

## Installation

```bash
git clone <repository-url>
cd GopsEA
pip install -e GopsEA/
```

Make sure Isaac Lab is properly installed. See [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/).

## Quick Start

### Training

```bash
python scripts/GopsEA/train_lab.py --task <task_name> --num_envs <num> --rldevice cuda:0
```

<details>
<summary>Training arguments</summary>

- `--task`: Task name
- `--num_envs`: Number of parallel environments
- `--rldevice`: Training device (e.g., `cuda:0`)
- `--seed`: Random seed (default: 42)
- `--video`: Record training videos
- `--cfg`: Specify config file directly

</details>

## Project Structure

<details>
<summary>Click to expand</summary>

```
GopsEA/
├── GopsEA/              # Main package
│   ├── algorithms/      # RL algorithms
│   ├── buffer/          # Experience replay buffer
│   ├── components/      # Core components (Actor, Critic, etc.)
│   ├── networks/        # Neural network definitions
│   ├── runners/         # Training runners
│   └── utils/           # Utilities
├── scripts/             # Training and testing scripts
├── third_party/         # Third-party dependencies
└── data/                # Data directory
```

</details>

## Core Components

<details>
<summary>Algorithms, Components, and Runners</summary>

- **Algorithms**: On-policy (PPO) and off-policy (DDPG, SAC) algorithms
- **Components**: Actor, Critic, Encoder/Decoder, Normalizer
- **Runners**: BaseRunner for unified training interface and logging

</details>

## Development

<details>
<summary>Adding new algorithms and environments</summary>

### Adding New Algorithm

1. Create a new algorithm class in `GopsEA/algorithms/`
2. Inherit from `AlgorithmBase` and implement required methods
3. Register the algorithm in the corresponding runner

### Adding New Environment

1. Ensure the environment follows Gymnasium interface
2. Wrap the environment with `GopsEAEnvWrapper`
3. Define environment parameters in config files

</details>

## Authors

- **Zaterval (interval-package) | Ziang Zheng**
- Maintainer: Ziang Zheng (ziang_zheng@foxmail.com)

## Acknowledgments

<details>
<summary>Repos that used for building this framework.</summary>

- [GOPS](https://gitee.com/tsinghua-university-iDLab-GOPS/gops) - General Optimal control Problem Solver
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - NVIDIA's robotics simulation platform

</details>


