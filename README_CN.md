# GopsEA

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-0.21+-green.svg)](https://isaac-sim.github.io/IsaacLab/)

一个**配置驱动的强化学习框架**，专用于机器人控制任务。基于 GOPS (General Optimal control Problem Solver) 构建，与 NVIDIA Isaac Lab 深度集成。

[English Documentation](README.md)

---

## ✨ 特性

- 🤖 **机器人控制**：四足机器人、人形机器人、机械臂等
- 🚀 **Isaac Lab 集成**：高性能 GPU 加速物理仿真
- 🧠 **多种 RL 算法**：
  - **On-Policy**: PPO (近端策略优化)
  - **Off-Policy**: SAC (软演员-评论家), DSAC (分布式 SAC)
  - **Model-Based**: MBPO (基于模型的 PPO)，支持学习系统动力学
- 📦 **模块化架构**：配置驱动设计，组件可灵活替换
- 🔄 **非对称观测**：支持策略网络和评论家网络使用不同观测
- 📊 **内置可视化**：支持 TensorBoard、Weights & Biases 和 Neptune

---

## 🚀 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd GopsEA

# 安装 GopsEA 包
pip install -e GopsEA/

# 安装 Isaac Lab (请参考官方文档)
# https://isaac-sim.github.io/IsaacLab/
```

### 训练

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

### 可用任务

| 机器人 | 环境 | PPO | SAC | MBPO |
|-------|-------------|-----|-----|------|
| Unitree A1 | 崎岖地形 | ✅ | ✅ | ✅ |
| Unitree A1 | 平坦地形 | ✅ | ✅ | ✅ |
| Unitree Go1 | 崎岖地形 | ✅ | ✅ | ✅ |
| Unitree Go1 | 平坦地形 | ✅ | ✅ | ✅ |
| Unitree Go2 | 崎岖地形 | ✅ | ✅ | ✅ |
| Unitree Go2 | 平坦地形 | ✅ | ✅ | ✅ |
| Anymal B | 崎岖地形 | ✅ | ✅ | ✅ |
| Anymal B | 平坦地形 | ✅ | ✅ | ✅ |
| Anymal C | 崎岖地形 | ✅ | ✅ | ✅ |
| Anymal C | 平坦地形 | ✅ | ✅ | ✅ |
| Anymal D | 崎岖地形 | ✅ | ✅ | ✅ |
| Anymal D | 平坦地形 | ✅ | ✅ | ✅ |
| H1 | 崎岖地形 | ✅ | ✅ | ✅ |
| H1 | 平坦地形 | ✅ | ✅ | ✅ |

---

## 🏗️ 架构

### 配置驱动设计

GopsEA 使用强大的 `@configclass` 装饰器（基于 Python dataclasses）实现类型安全的分层配置：

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

### 组件层级

```
Runner (BaseRunner)
├── Algorithm (AlgorithmBase)
│   ├── Actor (ActorBase)
│   ├── Critic (CriticBase)
│   └── [可选] SystemDynamics (SystemDynamicsBase)
├── Buffer (ReplayBufferBase)
│   ├── RolloutStorage (On-policy)
│   ├── DirectTransitionBuffer (Off-policy)
│   └── DynamicReplayBuffer (Model-based)
└── Normalizer (NormalizerBase)
```

### 关键设计模式

1. **模板方法模式**：所有组件继承自 `ModuleBase`，提供 `freeze()`/`defreeze()`/`frozen()` 上下文管理器
2. **工厂模式**：`construct_from_cfg()` 用于运行时通过维度注入实例化
3. **非对称观测**：内置支持策略和评论家使用不同观测
4. **延迟初始化**：Storage 在环境维度确定后才初始化

---

## 📁 项目结构

```
GopsEA/
├── GopsEA/                     # 主包
│   ├── algorithms/             # RL 算法
│   │   ├── on_policy/          # PPO, MBPO
│   │   ├── off_policy/         # SAC, DSAC
│   │   └── world_model_trainer/# 系统动力学训练器
│   ├── buffer/                 # 经验回放缓冲区
│   │   ├── online_rollout/     # RolloutStorage (PPO)
│   │   └── direct_based/       # SAC, DynamicReplayBuffer
│   ├── components/             # 神经网络组件
│   │   ├── actor/              # 策略网络
│   │   ├── critic/             # 价值网络
│   │   ├── normalizer/         # 观测归一化
│   │   └── world_models/       # 系统动力学模型 (MBPO)
│   ├── networks/               # 网络主干 (MLP 等)
│   ├── runners/                # 训练循环
│   │   ├── on_policy/          # OnPolicyRunner
│   │   └── off_policy/         # OffPolicyRunner
│   └── utils/                  # 工具函数
│       ├── configclass/        # 配置系统
│       ├── template/           # ModuleBase 等
│       └── env_wrapper/        # 环境包装器
├── scripts/                    # 训练脚本
├── third_party/                # 第三方依赖
│   └── gops_tasks/             # 任务定义
├── data/                       # 训练输出
└── docs/                       # 文档
```

---

## 🔧 高级用法

### 自定义配置

```python
# 在 third_party/gops_tasks/gops_tasks/isaaclab/locomotion/my_custom_agent.py 中
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

### 添加新算法

1. 在 `GopsEA/algorithms/<类别>/` 创建算法类
2. 继承 `AlgorithmBase`
3. 实现 `update()`, `act()`, `process_env_step()` 方法
4. 使用 `@configclass` 创建配置类
5. 在 `__init__.py` 中注册

### 添加新环境

1. 在 `third_party/gops_tasks/gops_tasks/isaaclab/` 创建环境配置
2. 确保符合 Gymnasium 接口规范
3. 定义 `dim_params`，包含 `policy_dim`, `critic_dim`, `action_dim`
4. 在 `__init__.py` 中使用 `gym.register()` 注册

---

## 📊 日志和监控

GopsEA 支持多种日志后端：

```python
@configclass
class MyRunnerCfg(runners.OnPolicyRunnerCfg):
    logger_cfg = runners.LoggerBaseCfg(
        logger="tensorboard",  # 或 "wandb", "neptune"
        is_log_ep_info=True,
        is_log_update=True,
    )
```

查看日志：
```bash
tensorboard --logdir data/MyExperiment/
```

---

## 🧪 测试

```bash
# 测试环境
python scripts/test_env.py --task GopsEA-UnitreeA1Rough-PPO

# 运行单元测试
pytest tests/
```

---

## 📚 参考

- [GOPS](https://gitee.com/tsinghua-university-iDLab-GOPS/gops) - 通用最优控制问题求解器
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) - NVIDIA 机器人仿真平台
- [Isaac Sim](https://developer.nvidia.com/isaac-sim) - NVIDIA 物理仿真平台

---

## 👥 作者

- **Zaterval (interval-package) | Ziang Zheng**
- 维护者：Ziang Zheng (ziang_zheng@foxmail.com)

---

## 📄 许可证

[在此添加许可证信息]

---

## 🙏 致谢

本项目基于以下项目构建：
- GOPS 框架 - 用于 RL 算法实现
- Isaac Lab - 用于高性能物理仿真
- RenforceRL - 用于先进的架构模式
