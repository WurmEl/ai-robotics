"""Generate and save a set of same-sized static maps."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from navigation.env.map_generator import generate_grid
from navigation.visualization.plot_grid import plot_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--maps", type=int, default=10)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--min-path-length", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/train/maps.pkl"))
    parser.add_argument("--display", action="store_true")
    args = parser.parse_args()
    if args.maps <= 0:
        parser.error("--maps must be positive.")

    rng = np.random.default_rng(args.seed)
    dataset = [
        generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            min_path_length=args.min_path_length,
            rng=rng,
        )
        for _ in range(args.maps)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output_file:
        pickle.dump(dataset, output_file)
    print(f"Saved {len(dataset)} maps to {args.output}")

    if args.display and dataset:
        grid, start, goal = dataset[0]
        from navigation.env.grid_world import GridWorldEnv

        env = GridWorldEnv(grid, start, goal)
        plot_grid(env.get_state())
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
