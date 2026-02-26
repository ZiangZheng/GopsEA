import h5py
import torch
import numpy as np
from typing import Dict, Generator, Any, Optional

def make_next_obs(obs: np.ndarray) -> np.ndarray:
    """
    Construct next observations from obs (obs1).
    Output is aligned as:
        next_obs[t] = obs[t+1]
        next_obs[T-1] = obs[T-1]   # keep same length

    Args:
        obs: (T, D) numpy array.

    Returns:
        next_obs: (T, D) numpy array.
    """
    T = obs.shape[0]
    next_obs = np.empty_like(obs)
    next_obs[:-1] = obs[1:]
    next_obs[-1] = obs[-1]   # last aligned
    return next_obs

def load_hdf5_trajectories(hdf5_path: str) -> Generator[Dict[str, np.ndarray], None, None]:
    """
    Iterate through IsaacLab HDF5 trajectory file.
    
    Each group (e.g., demo_0, demo_1, ...) is treated as a single trajectory.
    Yields a dict containing all datasets under the group.

    Args:
        hdf5_path: Path to the .hdf5 file.

    Yields:
        A dict mapping dataset names to numpy arrays, e.g.:
        {
            "reward": (T, ) array,
            "termination": (T,) bool array,
            "timeout": (T,) bool array,
            "action": (T, A) array,
            "rewards": (T, R) array,
            "dynamic": (T, W) array,
            "policy": (T, P) array,
        }
    """
    with h5py.File(hdf5_path, "r") as f:
        data_io = f["data"]
        for group_name in data_io.keys():
            group = data_io[group_name]

            traj = {}
            for dataset_name in group.keys():
                ds = group[dataset_name]
                traj[dataset_name] = ds[()]  # Read as numpy array
            yield traj

DATA_TRAJ_MAPPING = {
    "policy":       "policy",
    "dynamic":      "dynamic",
    "action":       "action",
    "reward":       "reward",
    "rewards":      "rewards",
    "termination":  "termination",
    "timeout":      "timeout"
}

if __name__ == "__main__":
    for traj in load_hdf5_trajectories("dataset.hdf5"):
        print("Action shape:", traj["action"].shape)