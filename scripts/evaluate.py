"""Evaluate A* and/or DQN on the same generated static maps."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from functools import partial

import numpy as np

from navigation.agents.astar_agent import AStarAgent
from navigation.agents.dqn_agent import DQNAgent
from navigation.env.grid_world import Action, GridWorldEnv
from navigation.env.map_generator import MapData, bfs_distance, generate_grid
from navigation.evaluation.metrics import EpisodeMetrics, save_metrics_csv


def _choose_astar_action(agent: AStarAgent, env: GridWorldEnv) -> Action | None:
    return agent.select_action(env.grid, env.position, env.goal)


def _choose_dqn_action(agent: DQNAgent, env: GridWorldEnv) -> Action:
    return agent.select_action(env.get_state(), explore=False)


def evaluate_episode(
    agent_name: str,
    episode: int,
    map_id: str,
    seed: int | None,
    map_data: MapData,
    obstacle_probability: float,
    choose_action: Callable[[GridWorldEnv], Action | None],
    max_steps: int,
) -> EpisodeMetrics:
    grid, start, goal = map_data
    optimal_path_length = bfs_distance(grid, start, goal)
    if optimal_path_length is None:
        raise ValueError(f"Evaluation map {map_id} has no valid path from start to goal.")

    env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
    state = env.reset()
    decision_times: list[float] = []
    path_length = 0.0
    cumulative_reward = 0.0
    episode_started = time.perf_counter()

    while not env.done:
        decision_started = time.perf_counter()
        action = choose_action(env)
        decision_times.append(time.perf_counter() - decision_started)
        if action is None:
            break
        previous_position = state.position
        state, reward, _, _ = env.step(action)
        path_length += abs(state.position[0] - previous_position[0])
        path_length += abs(state.position[1] - previous_position[1])
        cumulative_reward += reward

    episode_runtime = time.perf_counter() - episode_started
    return EpisodeMetrics(
        map_id=map_id,
        seed=seed,
        height=int(grid.shape[0]),
        width=int(grid.shape[1]),
        obstacle_probability=obstacle_probability,
        optimal_path_length=optimal_path_length,
        agent=agent_name,
        episode=episode,
        success=state.position == goal,
        collisions=env.collisions,
        steps=state.steps,
        path_length=path_length,
        episode_runtime=episode_runtime,
        mean_decision_time=(sum(decision_times) / len(decision_times) if decision_times else 0.0),
        cumulative_reward=cumulative_reward,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("astar", "dqn", "both"), default="both")
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", default="models/dqn.pt")
    parser.add_argument("--output", default="results/evaluation.csv")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")

    rng = np.random.default_rng(args.seed)
    maps = [
        generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            rng=rng,
        )
        for _ in range(args.episodes)
    ]
    selected_agents = ("astar", "dqn") if args.agent == "both" else (args.agent,)
    dqn: DQNAgent | None = None
    if "dqn" in selected_agents:
        dqn = DQNAgent((args.height, args.width), seed=args.seed)
        dqn.load(args.model_path)

    results: list[EpisodeMetrics] = []
    for agent_name in selected_agents:
        astar = AStarAgent() if agent_name == "astar" else None
        if astar is not None:
            choose_action = partial(_choose_astar_action, astar)
        else:
            assert dqn is not None
            choose_action = partial(_choose_dqn_action, dqn)

        for episode, map_data in enumerate(maps, start=1):
            result = evaluate_episode(
                agent_name=agent_name,
                episode=episode,
                map_id=f"map_{episode:05d}",
                seed=args.seed,
                map_data=map_data,
                obstacle_probability=args.obstacle_probability,
                choose_action=choose_action,
                max_steps=args.height * args.width * 4,
            )
            results.append(result)
            print(
                f"{agent_name} episode {episode}: success={result.success}, "
                f"steps={result.steps}, collisions={result.collisions}"
            )

    save_metrics_csv(results, args.output)
    print(f"Saved evaluation metrics to {args.output}")


if __name__ == "__main__":
    main()
