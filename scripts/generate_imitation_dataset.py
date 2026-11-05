"""Generate A*-labeled full-map behavior-cloning examples."""

from __future__ import annotations

import argparse

from navigation.imitation_dataset import generate_imitation_dataset, save_imitation_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--layouts-path", required=True)
    parser.add_argument("--task-seed", type=int, required=True)
    parser.add_argument("--tasks-per-layout", type=int, default=20)
    parser.add_argument("--min-path-length", type=int, default=2)
    parser.add_argument("--max-path-length", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.tasks_per_layout <= 0:
        parser.error("--tasks-per-layout must be positive.")
    if args.min_path_length <= 0:
        parser.error("--min-path-length must be positive.")
    if args.max_path_length is not None and args.max_path_length < args.min_path_length:
        parser.error("--max-path-length must be at least --min-path-length.")

    observations, actions, task_count = generate_imitation_dataset(
        args.layouts_path,
        height=args.height,
        width=args.width,
        task_seed=args.task_seed,
        tasks_per_layout=args.tasks_per_layout,
        min_path_length=args.min_path_length,
        max_path_length=args.max_path_length,
    )
    save_imitation_dataset(
        args.output,
        observations,
        actions,
        height=args.height,
        width=args.width,
        task_count=task_count,
        task_seed=args.task_seed,
        source_layouts_path=args.layouts_path,
    )
    print(
        f"Generated {task_count} tasks and {len(actions)} supervised samples "
        f"from {args.layouts_path}; saved to {args.output}"
    )


if __name__ == "__main__":
    main()
