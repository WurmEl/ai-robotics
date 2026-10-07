"""Small episode-metrics record and CSV writer."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodeMetrics:
    map_id: str
    seed: int | None
    height: int
    width: int
    obstacle_probability: float
    optimal_path_length: int
    agent: str
    episode: int
    success: bool
    collisions: int
    steps: int
    path_length: float
    episode_runtime: float
    mean_decision_time: float
    cumulative_reward: float
    training_episodes: int | None = None
    training_runtime: float | None = None


def save_metrics_csv(
    metrics: Iterable[EpisodeMetrics],
    path: str | Path,
) -> None:
    """Write episode results to a CSV file, replacing any existing file."""
    rows = [asdict(item) for item in metrics]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EpisodeMetrics.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
