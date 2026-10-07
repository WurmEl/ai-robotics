import csv

from navigation.evaluation.metrics import EpisodeMetrics, save_metrics_csv


def test_metrics_csv_includes_experimental_conditions_and_episode_results(tmp_path):
    metrics = EpisodeMetrics(
        map_id="map_00001",
        seed=31,
        height=4,
        width=6,
        obstacle_probability=0.2,
        optimal_path_length=7,
        agent="astar",
        episode=1,
        success=True,
        collisions=0,
        steps=7,
        path_length=7.0,
        episode_runtime=0.01,
        mean_decision_time=0.001,
        cumulative_reward=0.93,
    )
    output_path = tmp_path / "evaluation.csv"

    save_metrics_csv([metrics], output_path)

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        row = next(reader)

    assert reader.fieldnames is not None
    assert {
        "seed",
        "map_id",
        "height",
        "width",
        "obstacle_probability",
        "optimal_path_length",
        "agent",
        "episode",
        "success",
        "collisions",
        "steps",
        "path_length",
        "episode_runtime",
        "mean_decision_time",
        "cumulative_reward",
    }.issubset(reader.fieldnames)
    assert row["map_id"] == "map_00001"
    assert row["optimal_path_length"] == "7"
