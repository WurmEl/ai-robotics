"""Train a DQN on static maps with reproducible training and validation logs."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from navigation.agents.dqn_agent import DQNAgent
from navigation.env.grid_world import GridWorldEnv
from navigation.env.map_generator import (
    MapData,
    bfs_distance,
    generate_grid,
    generate_layout_dataset,
    load_layout_dataset,
    random_start_goal,
)
from navigation.rl.shaping import potential_shaping_reward


def _make_task(
    grid: np.ndarray,
    rng: np.random.Generator,
    min_path_length: int,
    max_path_length: int | None = None,
) -> MapData:
    start, goal = random_start_goal(
        grid,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        rng=rng,
    )
    return grid.copy(), start, goal


def _load_or_generate_layouts(
    path: str | None,
    count: int,
    height: int,
    width: int,
    obstacle_probability: float,
    min_path_length: int,
    seed: int,
) -> list[np.ndarray]:
    if path is not None and Path(path).exists():
        layouts = load_layout_dataset(path)
        if len(layouts) != count:
            raise ValueError(f"{path} contains {len(layouts)} layouts; expected {count}.")
        if any(grid.shape != (height, width) for grid in layouts):
            raise ValueError(f"{path} contains a layout with the wrong shape.")
        return layouts
    return generate_layout_dataset(
        count,
        height,
        width,
        obstacle_probability,
        min_path_length=min_path_length,
        save_path=path,
        seed=seed,
    )


def _make_fixed_tasks(
    layouts: list[np.ndarray],
    rng: np.random.Generator,
    min_path_length: int,
    max_path_length: int | None,
    count: int,
) -> list[MapData]:
    tasks: list[MapData] = []
    attempts = 0
    while len(tasks) < count and attempts < count * 20:
        attempts += 1
        try:
            tasks.append(
                _make_task(
                    layouts[int(rng.integers(len(layouts)))],
                    rng,
                    min_path_length,
                    max_path_length,
                )
            )
        except ValueError:
            continue
    if len(tasks) < count:
        raise ValueError("Could not create the requested fixed diagnostic tasks.")
    return tasks


def _run_validation(
    agent: DQNAgent,
    stage1_seen: list[MapData],
    stage1_unseen: list[MapData],
    current_tasks: list[MapData],
    final_unseen: list[MapData],
    max_steps: int,
) -> dict[str, float]:
    def evaluate(tasks: list[MapData], prefix: str) -> dict[str, float]:
        successes = 0
        collisions = 0
        timeouts = 0
        path_ratios: list[float] = []
        for grid, start, goal in tasks:
            env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
            state = env.reset()
            while not env.done:
                action = agent.select_action(state, explore=False)
                state, _, _, _ = env.step(action)
            successes += int(state.position == goal)
            collisions += env.collisions
            timeouts += int(state.position != goal)
            optimal = bfs_distance(grid, start, goal)
            if state.position == goal:
                path_ratios.append(state.steps / max(1, optimal or 1))
        count = max(1, len(tasks))
        return {
            f"{prefix}_success_rate": successes / count,
            f"{prefix}_mean_collisions": collisions / count,
            f"{prefix}_timeout_rate": timeouts / count,
            f"{prefix}_mean_path_ratio": float(np.mean(path_ratios)) if path_ratios else 0.0,
        }

    stage1_seen_metrics = evaluate(stage1_seen, "stage1_seen")
    stage1_unseen_metrics = evaluate(stage1_unseen, "stage1_unseen")
    current_metrics = evaluate(current_tasks, "current_stage")
    final_metrics = evaluate(final_unseen, "final_unseen")
    return {
        **stage1_seen_metrics,
        **stage1_unseen_metrics,
        **current_metrics,
        **final_metrics,
        "validation_success_rate": final_metrics["final_unseen_success_rate"],
        "validation_mean_collisions": final_metrics["final_unseen_mean_collisions"],
        "validation_timeout_rate": final_metrics["final_unseen_timeout_rate"],
        "validation_mean_path_ratio": final_metrics["final_unseen_mean_path_ratio"],
    }


def _q_value_diagnostics(agent: DQNAgent, tasks: list[MapData]) -> dict[str, float]:
    values = np.asarray(
        [agent.q_values(GridWorldEnv(grid, start, goal).get_state()) for grid, start, goal in tasks]
    )
    return {
        "q_mean": float(values.mean()),
        "q_min": float(values.min()),
        "q_max": float(values.max()),
        "q_mean_abs": float(np.abs(values).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--data-seed", type=int, default=None)
    parser.add_argument("--agent-seed", type=int, default=None)
    parser.add_argument("--task-seed", type=int, default=None)
    parser.add_argument("--model-path", default="models/dqn.pt")
    parser.add_argument("--network", choices=("mlp", "cnn"), default="mlp")
    parser.add_argument("--double-dqn", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--epsilon-decay-steps",
        type=int,
        default=None,
        help="default: 100000 for pool mode, 10000 otherwise",
    )
    parser.add_argument(
        "--learning-starts",
        type=int,
        default=None,
        help="default: 2000 for pool mode, 0 otherwise",
    )
    parser.add_argument("--learn-every", type=int, default=1)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--shaping-alpha", type=float, default=0.1)
    parser.add_argument("--shaping-gamma", type=float, default=0.99)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument(
        "--stage-episodes",
        default="250,250,250,250",
        help="episodes per curriculum stage",
    )
    parser.add_argument(
        "--training-mode",
        choices=("random", "fixed", "pool"),
        default="random",
        help="new map, one reused map, or a fixed pool of layouts",
    )
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--min-path-length", type=int, default=2)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--log-path", default="results/training.csv")
    parser.add_argument("--train-layouts-path", default=None)
    parser.add_argument("--validation-layouts-path", default=None)
    args = parser.parse_args()
    if args.episodes <= 0 or args.pool_size <= 0 or args.validation_size <= 0:
        parser.error("episodes, pool-size, and validation-size must be positive.")
    if args.validation_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("validation-interval and checkpoint-interval must be positive.")
    epsilon_decay_steps = args.epsilon_decay_steps or (
        100_000 if args.training_mode == "pool" else 10_000
    )
    learning_starts = args.learning_starts
    if learning_starts is None:
        learning_starts = 2_000 if args.training_mode == "pool" else 0
    if epsilon_decay_steps <= 0:
        parser.error("epsilon-decay-steps must be positive.")
    if learning_starts < 0:
        parser.error("learning-starts must be non-negative.")
    if args.learn_every <= 0:
        parser.error("learn-every must be positive.")
    if args.replay_capacity <= 0:
        parser.error("replay-capacity must be positive.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")
    if args.shaping_alpha < 0 or not 0 < args.shaping_gamma <= 1:
        parser.error("shaping-alpha must be non-negative and shaping-gamma must be in (0, 1].")
    try:
        stage_episodes = [int(value) for value in args.stage_episodes.split(",")]
    except ValueError as error:
        parser.error("--stage-episodes must be comma-separated positive integers.")
        raise error
    if not stage_episodes or any(value <= 0 for value in stage_episodes):
        parser.error("--stage-episodes must contain positive integers.")
    if args.curriculum and sum(stage_episodes) != args.episodes:
        parser.error("curriculum stage episodes must sum to --episodes.")

    data_seed = args.data_seed if args.data_seed is not None else args.seed
    agent_seed = args.agent_seed if args.agent_seed is not None else args.seed
    task_seed = args.task_seed if args.task_seed is not None else args.seed
    data_generation_seed = 0 if data_seed is None else data_seed
    task_generation_seed = 0 if task_seed is None else task_seed
    data_rng = np.random.default_rng(data_generation_seed)
    task_rng = np.random.default_rng(task_generation_seed)
    agent = DQNAgent(
        (args.height, args.width),
        epsilon_decay_steps=epsilon_decay_steps,
        seed=agent_seed,
        network=args.network,
        learning_rate=args.learning_rate,
        replay_capacity=args.replay_capacity,
        double_dqn=args.double_dqn,
        dueling=args.dueling,
    )
    max_steps = args.max_steps if args.max_steps is not None else args.height * args.width * 4

    fixed_map: MapData | None = None
    layouts: list[np.ndarray] = []
    curriculum_stages: list[tuple[float, int, int]] = []
    stage1_seen: list[MapData]
    stage1_unseen: list[MapData]
    current_stage_tasks: list[list[MapData]] = []
    final_unseen: list[MapData]
    if args.training_mode == "fixed":
        fixed_map = generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            min_path_length=args.min_path_length,
            rng=data_rng,
        )
        stage1_seen = [fixed_map]
        stage1_unseen = [fixed_map]
        current_stage_tasks = [[fixed_map]]
        final_unseen = [fixed_map]
    elif args.training_mode == "pool":
        layouts = _load_or_generate_layouts(
            args.train_layouts_path,
            args.pool_size,
            args.height,
            args.width,
            args.obstacle_probability,
            args.min_path_length,
            data_generation_seed,
        )
        if args.curriculum:
            curriculum_stages = [
                (args.obstacle_probability, 2, 4),
                (args.obstacle_probability, 3, 7),
                (args.obstacle_probability, 5, 10),
                (args.obstacle_probability, 5, 15),
            ]
        validation_layouts = _load_or_generate_layouts(
            args.validation_layouts_path,
            args.validation_size,
            args.height,
            args.width,
            args.obstacle_probability,
            args.min_path_length,
            data_generation_seed + 1,
        )
        validation_rng = np.random.default_rng(task_generation_seed)
        validation_tasks = [
            _make_task(grid, validation_rng, args.min_path_length) for grid in validation_layouts
        ]
        seen_rng = np.random.default_rng(task_generation_seed + 1)
        stage1_seen = _make_fixed_tasks(layouts, seen_rng, 2, 4, min(args.pool_size, 20))
        stage1_unseen = _make_fixed_tasks(
            validation_layouts, seen_rng, 2, 4, min(args.validation_size, 20)
        )
        final_unseen = validation_tasks
        current_stage_tasks = (
            [
                _make_fixed_tasks(
                    layouts,
                    seen_rng,
                    stage_min,
                    stage_max,
                    min(args.pool_size, 20),
                )
                for _, stage_min, stage_max in curriculum_stages
            ]
            if args.curriculum
            else [stage1_seen]
        )
    else:
        validation_rng = np.random.default_rng(task_generation_seed)
        validation_tasks = [
            generate_grid(
                args.height,
                args.width,
                obstacle_probability=args.obstacle_probability,
                min_path_length=args.min_path_length,
                rng=validation_rng,
            )
            for _ in range(args.validation_size)
        ]
        stage1_seen = validation_tasks
        stage1_unseen = validation_tasks
        current_stage_tasks = [validation_tasks]
        final_unseen = validation_tasks

    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "episode",
        "curriculum_stage",
        "environment_steps",
        "elapsed_seconds",
        "epsilon",
        "reward",
        "base_reward",
        "shaped_training_reward",
        "success",
        "steps",
        "collisions",
        "rolling_training_success",
        "training_loss",
        "training_elapsed_seconds",
        "environment_steps_per_second",
        "validation_success_rate",
        "validation_mean_collisions",
        "validation_timeout_rate",
        "validation_mean_path_ratio",
        "seen_success_rate",
        "seen_mean_collisions",
        "seen_timeout_rate",
        "seen_mean_path_ratio",
        "unseen_success_rate",
        "unseen_mean_collisions",
        "unseen_timeout_rate",
        "unseen_mean_path_ratio",
        "stage1_seen_success_rate",
        "stage1_seen_mean_collisions",
        "stage1_seen_timeout_rate",
        "stage1_seen_mean_path_ratio",
        "stage1_unseen_success_rate",
        "stage1_unseen_mean_collisions",
        "stage1_unseen_timeout_rate",
        "stage1_unseen_mean_path_ratio",
        "current_stage_success_rate",
        "current_stage_mean_collisions",
        "current_stage_timeout_rate",
        "current_stage_mean_path_ratio",
        "final_unseen_success_rate",
        "final_unseen_mean_collisions",
        "final_unseen_timeout_rate",
        "final_unseen_mean_path_ratio",
        "validation_runtime_seconds",
        "q_mean",
        "q_min",
        "q_max",
        "q_mean_abs",
    ]
    latest_path = Path(args.model_path).with_name(Path(args.model_path).stem + "-latest.pt")
    best_path = Path(args.model_path).with_name(Path(args.model_path).stem + "-best.pt")
    best_validation = -1.0
    environment_steps = 0
    training_started = time.perf_counter()
    training_elapsed = 0.0
    rolling_successes = 0
    rolling_rewards: list[float] = []
    print(
        "Experiment configuration | "
        f"data_seed={data_seed} | agent_seed={agent_seed} | task_seed={task_seed} | "
        f"replay_capacity={args.replay_capacity} | learning_rate={args.learning_rate} | "
        f"epsilon_decay_steps={epsilon_decay_steps} | pool_size={args.pool_size} | "
        f"stage_episodes={stage_episodes} | double_dqn={args.double_dqn} | "
        f"dueling={args.dueling}"
    )
    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fields)
        writer.writeheader()
        for episode in range(1, args.episodes + 1):
            stage_index = 0
            if args.curriculum:
                cumulative = 0
                for index, count in enumerate(stage_episodes):
                    cumulative += count
                    if episode <= cumulative:
                        stage_index = index
                        break
                stage_min = curriculum_stages[stage_index][1]
            if fixed_map is not None:
                task = fixed_map
            elif layouts:
                if args.curriculum:
                    requested_stage = stage_index
                    if stage_index > 0 and task_rng.random() < 0.25:
                        requested_stage = int(task_rng.integers(stage_index))
                    stage_min = curriculum_stages[requested_stage][1]
                    stage_max = curriculum_stages[requested_stage][2]
                else:
                    stage_min = args.min_path_length
                    stage_max = None
                for _ in range(20):
                    try:
                        task = _make_task(
                            layouts[int(task_rng.integers(len(layouts)))],
                            task_rng,
                            stage_min,
                            stage_max,
                        )
                        break
                    except ValueError:
                        continue
                else:
                    raise ValueError("Could not sample a training task from the layout pool.")
            else:
                task = generate_grid(
                    args.height,
                    args.width,
                    obstacle_probability=args.obstacle_probability,
                    min_path_length=args.min_path_length,
                    rng=data_rng,
                )
            grid, start, goal = task
            env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
            state = env.reset()
            reward_total = 0.0
            shaped_reward_total = 0.0
            losses: list[float] = []
            episode_started = time.perf_counter()
            while not env.done:
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                shaping = (
                    potential_shaping_reward(
                        state,
                        next_state,
                        alpha=args.shaping_alpha,
                        gamma=args.shaping_gamma,
                    )
                    if args.reward_shaping
                    else 0.0
                )
                training_reward = reward + shaping
                agent.observe(
                    state,
                    action,
                    training_reward,
                    next_state,
                    done,
                    terminated=bool(info["terminated"]),
                )
                loss = (
                    agent.learn()
                    if len(agent.replay_buffer) >= learning_starts
                    and environment_steps % args.learn_every == 0
                    else None
                )
                if loss is not None:
                    losses.append(loss)
                reward_total += reward
                shaped_reward_total += training_reward
                environment_steps += 1
                state = next_state
            training_elapsed += time.perf_counter() - episode_started
            rolling_successes += int(state.position == goal)
            rolling_rewards.append(reward_total)

            validation_started = time.perf_counter()
            validation = (
                _run_validation(
                    agent,
                    stage1_seen,
                    stage1_unseen,
                    current_stage_tasks[stage_index],
                    final_unseen,
                    max_steps,
                )
                if episode % args.validation_interval == 0 or episode == args.episodes
                else {}
            )
            q_diagnostics = _q_value_diagnostics(agent, final_unseen) if validation else {}
            validation_runtime = time.perf_counter() - validation_started if validation else 0.0
            row: dict[str, float | int | bool] = {
                "episode": episode,
                "curriculum_stage": stage_index + 1 if args.curriculum else 0,
                "environment_steps": environment_steps,
                "elapsed_seconds": time.perf_counter() - training_started,
                "epsilon": agent.epsilon,
                "reward": reward_total,
                "base_reward": reward_total,
                "shaped_training_reward": shaped_reward_total,
                "success": state.position == goal,
                "steps": state.steps,
                "collisions": env.collisions,
                "rolling_training_success": rolling_successes / len(rolling_rewards),
                "training_loss": float(np.mean(losses)) if losses else 0.0,
                "training_elapsed_seconds": training_elapsed,
                "environment_steps_per_second": environment_steps / max(training_elapsed, 1e-9),
                **validation,
                **q_diagnostics,
                "validation_runtime_seconds": validation_runtime,
            }
            writer.writerow(row)
            log_file.flush()

            if validation and validation["validation_success_rate"] > best_validation:
                best_validation = validation["validation_success_rate"]
                agent.save(best_path)
            if episode % args.checkpoint_interval == 0 or episode == args.episodes:
                agent.save(latest_path)
            if validation:
                print(
                    f"Episode {episode}/{args.episodes} | "
                    f"train success (rolling): {row['rolling_training_success']:.2f} | "
                    f"seen greedy: {validation['stage1_seen_success_rate']:.2f} | "
                    f"unseen greedy: {validation['final_unseen_success_rate']:.2f} | "
                    f"epsilon: {agent.epsilon:.3f} | "
                    f"steps/s: {row['environment_steps_per_second']:.1f} | "
                    f"Q mean: {row['q_mean']:.3f}"
                )
            if episode % 100 == 0 or episode == args.episodes:
                rolling_successes = 0
                rolling_rewards.clear()

    agent.save(args.model_path)
    print(f"Saved final model to {args.model_path}; log written to {log_path}")
    print(f"Latest checkpoint: {latest_path}; best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
