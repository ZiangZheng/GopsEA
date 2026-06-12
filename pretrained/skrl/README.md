# Pretrained skrl checkpoints

These checkpoints are copied from the original IsaacLearning PickPlace runs and
can be evaluated against the migrated IsaacLab tasks:

- `franka_cube_lift/best_agent.pt`
  - source: `/home/infinite/IsaacLearning/PickPlace/logs/skrl/franka_lift/2026-03-05_17-58-51_ppo_torch/checkpoints/best_agent.pt`
  - selected by 500-step rollout validation on the migrated task
  - validation: `mean_return=4990.205566`, `mean_step_reward=9.980411`
  - sha256: `312870901b9f0d4959dee02ebd99d764c41d951c2b3156280170c5d0d6224bac`
- `pickplace_simple/best_agent.pt`
  - source: `/home/infinite/IsaacLearning/PickPlace_simple/logs/skrl/franka_lift/2026-03-05_20-22-45_ppo_torch/checkpoints/best_agent.pt`
  - selected by best TensorBoard `Reward / Total reward (mean)`: 849.648 at step 33840
  - validation: `mean_return=1803.197998`, `mean_step_reward=3.606396`
  - sha256: `f30dd82d0714e1536cf940224ddfe033bedc704d6656e047222f5229f44e8dc0`
