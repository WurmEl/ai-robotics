"""Print a readable summary of an evaluation CSV (per agent and per map)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def summarize_agent(rows: list[dict[str, str]]) -> dict[str, float]:
    successes = [row for row in rows if row["success"] == "True"]
    # Path overhead is only meaningful for successful episodes.
    overheads = [float(row["path_length"]) / float(row["optimal_path_length"]) for row in successes]
    return {
        "episodes": len(rows),
        "success_rate": len(successes) / len(rows),
        "collisions": mean(float(row["collisions"]) for row in rows),
        "steps": mean(float(row["steps"]) for row in rows),
        "path_ratio": mean(overheads) if overheads else float("nan"),
        "decision_ms": 1000 * mean(float(row["mean_decision_time"]) for row in rows),
        "reward": mean(float(row["cumulative_reward"]) for row in rows),
    }


def print_agent_table(rows: list[dict[str, str]]) -> None:
    by_agent: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_agent[row["agent"]].append(row)

    print("Per-agent summary (means over episodes)")
    header = (
        f"{'agent':<8}{'episodes':>9}{'success':>9}{'collisions':>12}"
        f"{'steps':>8}{'path/opt':>10}{'decision ms':>13}{'reward':>9}"
    )
    print(header)
    print("-" * len(header))
    for agent, agent_rows in by_agent.items():
        s = summarize_agent(agent_rows)
        print(
            f"{agent:<8}{s['episodes']:>9.0f}{s['success_rate']:>9.0%}{s['collisions']:>12.1f}"
            f"{s['steps']:>8.1f}{s['path_ratio']:>10.2f}{s['decision_ms']:>13.3f}"
            f"{s['reward']:>9.2f}"
        )
    print("\npath/opt = path length / BFS optimum, successful episodes only (1.00 = optimal).")


def print_paired_table(rows: list[dict[str, str]]) -> None:
    by_map: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_map[row["map_id"]][row["agent"]] = row

    agents = sorted({row["agent"] for row in rows})
    print("\nPer-map comparison (same map for every agent)")
    print(
        f"{'map_id':<11}{'optimal':>8}"
        + "".join(f"{a + ' steps':>13}{a + ' ok':>8}" for a in agents)
    )
    for map_id, per_agent in sorted(by_map.items()):
        first = next(iter(per_agent.values()))
        line = f"{map_id:<11}{first['optimal_path_length']:>8}"
        for agent in agents:
            row = per_agent.get(agent)
            line += f"{row['steps']:>13}{row['success']:>8}" if row else f"{'-':>13}{'-':>8}"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/evaluation.csv"))
    parser.add_argument("--per-map", action="store_true", help="also print one row per map")
    args = parser.parse_args()

    rows = load_rows(args.input)
    if not rows:
        parser.error(f"{args.input} contains no rows.")
    print_agent_table(rows)
    if args.per_map:
        print_paired_table(rows)


if __name__ == "__main__":
    main()
