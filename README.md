# RenforceRL

<div align="center">

**A flexible and modular reinforcement learning framework for robotics, with a focus on world models and model-based policy optimization.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-BSD--3-green.svg)](LICENSE)

</div>

---

## 🚀 Overview

RenforceRL is a comprehensive reinforcement learning framework designed for robotics applications, with special emphasis on world model learning and model-based policy optimization (MBPO). The framework provides seamless integration with **Isaac Gym** and **Isaac Lab**, making it ideal for training and evaluating policies on various robotic tasks.

### Key Features

- 🌍 **World Model Support**: Flexible world model architecture for model-based RL
- 🤖 **Multi-Environment Support**: Native support for Isaac Gym and Isaac Lab environments
- 📊 **Multiple Algorithms**: Implementation of PPO, SAC, DSAC, and MBPO
- 🔄 **On-Policy & Off-Policy**: Support for both on-policy and off-policy training paradigms
- 🎯 **Robotics Focus**: Pre-configured tasks for locomotion and manipulation
- 🧩 **Modular Design**: Clean separation of components (runners, algorithms, networks, buffers)
- 📈 **Comprehensive Logging**: Built-in TensorBoard and Tqdm-style logging

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- [Isaac Lab](https://github.com/isaac-sim/Isaac-Lab) or [Isaac Gym](https://developer.nvidia.com/isaac-gym) installed

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd RenforceRL
```

2. Install dependencies and setup external modules:
```bash
bash scripts/setup_ext.sh
```

This script will:
- Clone required repositories (assetslib, robotlib)
- Install local packages in editable mode

3. Install the package:
```bash
pip install -e source/RenforceRL
pip install -e source/demo_tasks  # Optional: for demo tasks
```

---

## 🏃 Quick Start

### Training on Isaac Lab

Train a locomotion task using MBPO:

```bash
python scripts/renforce/train_lab.py \
    --task Isaac-UnitreeA1-Rough-v0 \
    --num_envs 4096 \
    --seed 42
```

### Training on Isaac Gym / Gymnasium

Train a standard Gymnasium environment:

```bash
python scripts/renforce/train_gym.py \
    --task Pendulum-v1 \
    --seed 42 \
    --rldevice cuda:0
```

### Evaluation / Playback

Evaluate a trained policy:

```bash
python scripts/renforce/play_lab.py \
    --task Isaac-UnitreeA1-Rough-v0 \
    --target <path-to-checkpoint> \
    --video  # Optional: record video
```

---

## 📚 Documentation

### Core Components

- **Runners**: Training and evaluation orchestration
  - `OnPolicyRunner`: For on-policy algorithms (PPO, MBPO)
  - `OffPolicyRunner`: For off-policy algorithms (SAC, DSAC)
  - `MBPOOnPolicyRunner`: Model-based policy optimization runner

- **Algorithms**: RL algorithm implementations
  - **On-Policy**: PPO, MBPO
  - **Off-Policy**: SAC, DSAC

- **World Models**: Model-based RL components
  - Base world model architecture
  - System dynamics models
  - Planning and inference utilities

- **Environment Wrappers**: Environment adapters
  - `RFImagineEnvWrapper`: World model imagination wrapper
  - `RFDynamicEnvWrapper`: Dynamic environment wrapper
  - `SimpleGymWrapper`: Gymnasium environment wrapper

### World Model Terminology

For detailed information about world model terminology and data pipeline, see [World Model Documentation](docs/world_model.md).

Key concepts:
- **Observations**: `policy`, `critic`, `dynamic` observation spaces
- **Rewards**: Shaped reward tensors and multi-dimensional reward vectors
- **Terminations**: Timeout and termination signals
- **Masks**: Validity masks for transitions

---

## 🎯 Supported Tasks

### Isaac Lab Tasks

The framework includes pre-configured tasks for various robots:

- **Quadrupeds**: Unitree A1, Go1, Go2; Anymal B/C/D
- **Humanoids**: H1
- **Environments**: Rough terrain, flat terrain locomotion

See `source/demo_tasks/demo_tasks/isaaclab/locomotion/` for task definitions.

### Gymnasium Tasks

Standard Gymnasium environments are supported:
- Classic control: Pendulum, CartPole, etc.
- MuJoCo: HalfCheetah, Walker2d, etc.

---

## 🏗️ Project Structure

```
RenforceRL/
├── source/
│   ├── RenforceRL/          # Main framework
│   │   ├── algorithms/       # RL algorithms (PPO, SAC, MBPO, etc.)
│   │   ├── runners/          # Training runners
│   │   ├── components/       # Networks, actors, critics, world models
│   │   ├── buffer/           # Replay buffers and data pipelines
│   │   └── utils/            # Utilities and wrappers
│   ├── demo_tasks/           # Task definitions
│   └── robotlib/             # Robot configuration library
├── scripts/
│   └── renforce/             # Training and evaluation scripts
├── data/                     # Data and assets
└── logs/                     # Training logs and checkpoints
```

---

## 🔧 Configuration

RenforceRL uses a configuration-based approach. Tasks and agents are configured via config classes:

```python
from RenforceRL import configclass
from RenforceRL.runners import MBPOOnPolicyRunnerCfg

@configclass
class MyTaskCfg(MBPOOnPolicyRunnerCfg):
    experiment_name = "my_experiment"
    max_iterations = 1000
    # ... configure policy, algorithm, etc.
```

See example configurations in `source/demo_tasks/` for reference.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📝 License

This project is licensed under the BSD-3 License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- Built with alignment to `rsl_rl` conventions
- Supports diffusion-guided generation for world models
- Inspired by TD-MPC2 and other model-based RL approaches

---

## 📖 Additional Resources

- [World Model Documentation](docs/world_model.md) - Detailed world model terminology and pipeline
- [Training Data Pipeline](docs/data_pipeline.md) - Data flow and processing
- [RobotLib](source/robotlib/README.md) - Robot configuration library

---

## 🔗 Related Projects

- [RobotLib](https://github.com/Renforce-Dynamics/robotlib) - Universal robot asset and configuration hub
- [AssetsLib](https://github.com/Renforce-Dynamics/assetslib) - Robot assets repository

---

## 📧 Contact

- Maintainer: Ziang Zheng
- Email: ziang_zheng@foxmail.com

---

## 🗺️ Roadmap

### Completed ✅
- [x] World model offline trainer (full version)
- [x] World model runner with on-policy algorithm
- [x] Evaluation with each term
- [x] Input with policy obs while loss logits with world model obs (Dynamic terms)

### In Progress 🚧
- [ ] Observation normalization for world model
- [ ] Evaluation inference for world model planner
- [ ] Fully offline training
- [ ] Distributional modeling for latent variables
- [ ] Additional task support

---

<div align="center">

**Made with ❤️ for the robotics RL community**

</div>
