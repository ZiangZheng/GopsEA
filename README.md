# GopsEA

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.21+-green.svg)](https://isaac-sim.github.io/IsaacLab/)

A **configuration-driven reinforcement learning framework** for robot control tasks, built on GOPS (General Optimal control Problem Solver) and deeply integrated with NVIDIA Isaac Lab.

[中文文档](README_CN.md)

---

## ✨ Features

- 🤖 **Robot Control**: Quadruped robots, humanoid robots, manipulators, and more
- 🚀 **Isaac Lab Integration**: High-performance GPU-accelerated physics simulation
- 🧠 **Multiple RL Algorithms**:
  - **On-Policy**: PPO (Proximal Policy Optimization)
  - **Off-Policy**: SAC (Soft Actor-Critic), DSAC (Distributional SAC)
  - **Model-Based**: MBPO (Model-Based PPO) with learned system dynamics
- 📦 **Modular Architecture**: Config-driven design with interchangeable components
- 🔄 **Asymmetric Observations**: Different observations for policy and critic networks
- 📊 **Built-in Visualization**: TensorBoard, Weights & Biases, and Neptune support

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd GopsEA

# Install GopsEA package
pip install -e GopsEA/

# Install Isaac Lab (follow official documentation)
# https://isaac-sim.github.io/IsaacLab/
```

### Training

#### PPO (On-Policy)

```bash
python scripts/GopsEA/train_lab.py \
    --task GopsEA-UnitreeA1Rough-PPO \
    --num_envs 4096 \
    --rldevice cuda:0
```

#### SAC (Off-Policy)

```bash
python scripts/GopsEA/train_lab.py \
    --task GopsEA-UnitreeA1Rough-SAC \
    --num_envs 4096 \
    --rldevice cuda:0
```

#### MBPO (Model-Based)

```bash
python scripts/GopsEA/train_lab.py \
    --task GopsEA-UnitreeA1Rough-MBPO \
    --num_envs 4096 \
    --rldevice cuda:0
```

### Available Tasks

| Robot | Environment | PPO | SAC | MBPO |
|-------|-------------|-----|-----|------|
| Unitree A1 | Rough Terrain | ✅ | ✅ | ✅ |
| Unitree A1 | Flat Terrain | ✅ | ✅ | ✅ |
| Unitree Go1 | Rough Terrain | ✅ | ✅ | ✅ |
| Unitree Go1 | Flat Terrain | ✅ | ✅ | ✅ |
| Unitree Go2 | Rough Terrain | ✅ | ✅ | ✅ |
| Unitree Go2 | Flat Terrain | ✅ | ✅ | ✅ |
| Anymal B | Rough Terrain | ✅ | ✅ | ✅ |
| Anymal B | Flat Terrain | ✅ | ✅ | ✅ |
| Anymal C | Rough Terrain | ✅ | ✅ | ✅ |
| Anymal C | Flat Terrain | ✅ | ✅ | ✅ |
| Anymal D | Rough Terrain | ✅ | ✅ | ✅ |
| Anymal D | Flat Terrain | ✅ | ✅ | ✅ |
| H1 | Rough Terrain | ✅ | ✅ | ✅ |
| H1 | Flat Terrain | ✅ | ✅ | ✅ |

---

## 🏗️ Architecture

### Config-Driven Design

GopsEA uses a powerful `@configclass` decorator (built on Python dataclasses) for type-safe, hierarchical configuration:

```python
from GopsEA import configclass
from GopsEA import runners, algorithms, components, networks

@configclass
class MyPPOCfg(runners.OnPolicyRunnerCfg):
    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[512, 256, 128],
                activations=[[("ELU", {})]] * 3 + [[]]
            )
        ),
        critic_cfg=components.VNetworkCfg(...)
    )
    algorithm = algorithms.PPOCfg(
        learning_rate=3e-4,
        num_learning_epochs=5,
        num_mini_batches=4,
    )
```

### Component Hierarchy

```
Runner (BaseRunner)
├── Algorithm (AlgorithmBase)
│   ├── Actor (ActorBase)
│   ├── Critic (CriticBase)
│   └── [Optional] SystemDynamics (SystemDynamicsBase)
├── Buffer (ReplayBufferBase)
│   ├── RolloutStorage (On-policy)
│   ├── DirectTransitionBuffer (Off-policy)
│   └── DynamicReplayBuffer (Model-based)
└── Normalizer (NormalizerBase)
```

### Key Design Patterns

1. **Template Method Pattern**: All components inherit from `ModuleBase` with `freeze()`/`defreeze()`/`frozen()` context manager
2. **Factory Pattern**: `construct_from_cfg()` for runtime instantiation with dimension injection
3. **Asymmetric Observations**: Built-in support for different policy/critic observations
4. **Lazy Initialization**: Storage initialized after environment dimensions are known

---

## 📁 Project Structure

```
GopsEA/
├── GopsEA/                     # Main package
│   ├── algorithms/             # RL algorithms
│   │   ├── on_policy/          # PPO, MBPO
│   │   ├── off_policy/         # SAC, DSAC
│   │   └── world_model_trainer/# System dynamics trainer
│   ├── buffer/                 # Experience replay buffers
│   │   ├── online_rollout/     # RolloutStorage (PPO)
│   │   └── direct_based/       # SAC, DynamicReplayBuffer
│   ├── components/             # Neural network components
│   │   ├── actor/              # Policy networks
│   │   ├── critic/             # Value networks
│   │   ├── normalizer/         # Observation normalization
│   │   └── world_models/       # System dynamics models (MBPO)
│   ├── networks/               # Network backbones (MLP, etc.)
│   ├── runners/                # Training loops
│   │   ├── on_policy/          # OnPolicyRunner
│   │   └── off_policy/         # OffPolicyRunner
│   └── utils/                  # Utilities
│       ├── configclass/        # Configuration system
│       ├── template/           # ModuleBase, etc.
│       └── env_wrapper/        # Environment wrappers
├── scripts/                    # Training scripts
├── third_party/                # Third-party dependencies
│   └── gops_tasks/             # Task definitions
├── data/                       # Training outputs
└── docs/                       # Documentation
```

---

## 🔧 Advanced Usage

### Custom Configuration

```python
# In third_party/gops_tasks/gops_tasks/isaaclab/locomotion/my_custom_agent.py
from GopsEA import configclass
from GopsEA import runners, algorithms, components, networks

@configclass
class MyCustomPPOCfg(runners.OnPolicyRunnerCfg):
    experiment_name = "MyCustomRobot"

    policy = components.ActorCriticPackCfg(
        actor_cfg=components.StateIndStdActorCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[1024, 512, 256],
                activations=[[("ELU", {})]] * 3 + [[]]
            ),
            use_log_std=False
        ),
        critic_cfg=components.VNetworkCfg(
            backbone_cfg=networks.MLPCfg(
                hidden_features=[1024, 512, 256],
                activations=[[("ELU", {})]] * 3 + [[]]
            )
        )
    )

    algorithm = algorithms.PPOCfg(
        learning_rate=1e-3,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        num_learning_epochs=5,
        num_mini_batches=4,
    )

    obs_normalize_cfg = components.NormalizerEmpiricalCfg()
```

### Adding New Algorithm

1. Create algorithm class in `GopsEA/algorithms/<category>/`
2. Inherit from `AlgorithmBase`
3. Implement `update()`, `act()`, `process_env_step()` methods
4. Create config class with `@configclass`
5. Register in `__init__.py`

### Adding New Environment

1. Create environment config in `third_party/gops_tasks/gops_tasks/isaaclab/`
2. Ensure Gymnasium interface compliance
3. Define `dim_params` with `policy_dim`, `critic_dim`, `action_dim`
4. Register in `__init__.py` with `gym.register()`

---

## 📊 Logging and Monitoring

GopsEA supports multiple logging backends:

```python
@configclass
class MyRunnerCfg(runners.OnPolicyRunnerCfg):
    logger_cfg = runners.LoggerBaseCfg(
        logger="tensorboard",  # or "wandb", "neptune"
        is_log_ep_info=True,
        is_log_update=True,
    )
```

View logs:
```bash
tensorboard --logdir data/MyExperiment/
```

---

## 🧪 Testing

```bash
# Test environment
python scripts/test_env.py --task GopsEA-UnitreeA1Rough-PPO

# Run unit tests
pytest tests/
```

---

## 📚 References

- [GOPS](https://gitee.com/tsinghua-university-iDLab-GOPS/gops) - General Optimal control Problem Solver
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - NVIDIA's robotics simulation platform
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) - NVIDIA's physics simulation platform

---

## 👥 Authors

- **Zaterval (interval-package) | Ziang Zheng**
- Maintainer: Ziang Zheng (ziang_zheng@foxmail.com)

---

## 📄 License

[Add your license information here]

---

## 🙏 Acknowledgments

This project builds upon:
- GOPS framework for RL algorithm implementations
- Isaac Lab for high-performance physics simulation
- RenforceRL for advanced architecture patterns
