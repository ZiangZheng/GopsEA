import argparse

import gymnasium as gym
import gops_tasks  # noqa: F401


def _select_ids(prefix: str) -> list[str]:
    return sorted([str(env_id) for env_id in gym.envs.registry.keys() if str(env_id).startswith(prefix)])


def _print_group(title: str, env_ids: list[str], show_all: bool, limit: int) -> None:
    print(f"\n[{title}] count={len(env_ids)}")
    if not env_ids:
        return
    if show_all:
        for env_id in env_ids:
            print(f"  - {env_id}")
        return
    for env_id in env_ids[:limit]:
        print(f"  - {env_id}")
    if len(env_ids) > limit:
        print(f"  ... ({len(env_ids) - limit} more)")


def main():
    parser = argparse.ArgumentParser(description="List registered environments in gymnasium registry.")
    parser.add_argument("--all", action="store_true", help="Print all env ids in each group.")
    parser.add_argument("--limit", type=int, default=20, help="Max env ids per group when not using --all.")
    args = parser.parse_args()

    groups = {
        "Legacy GOPS envs (gym_*)": _select_ids("gym_"),
        "DMControl envs (dmc-*)": _select_ids("dmc-"),
        "HumanoidBench envs (gops-hb-*)": _select_ids("gops-hb-"),
        "IsaacLab envs (GopsEA-*)": _select_ids("GopsEA-"),
    }

    print("Registered environment summary")
    total = sum(len(v) for v in groups.values())
    print(f"Total (tracked groups): {total}")

    for title, env_ids in groups.items():
        _print_group(title, env_ids, show_all=args.all, limit=args.limit)


if __name__ == "__main__":
    main()
