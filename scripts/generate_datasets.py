"""Generate separate reproducible train, validation, and test layout datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from navigation.env.map_generator import generate_layout_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--min-path-length", type=int, default=2)
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--validation", type=int, default=100)
    parser.add_argument("--test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if min(args.train, args.validation, args.test) <= 0:
        parser.error("--train, --validation, and --test must be positive.")

    configurations = (
        ("train", args.train, args.seed),
        ("validation", args.validation, args.seed + 1),
        ("test", args.test, args.seed + 2),
    )
    for name, count, seed in configurations:
        path = args.output_dir / name / "layouts.pkl"
        generate_layout_dataset(
            count,
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            min_path_length=args.min_path_length,
            save_path=path,
            seed=seed,
        )
        print(f"Saved {count} {name} layouts to {path}")


if __name__ == "__main__":
    main()
