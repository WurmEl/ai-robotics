"""Run repeated online A* planning on one generated static map."""

from __future__ import annotations

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

from navigation.agents.astar_agent import AStarAgent
from navigation.env.grid_world import GridWorldEnv
from navigation.env.map_generator import generate_grid
from navigation.visualization.plot_grid import plot_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=15)
    parser.add_argument("--width", type=int, default=15)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    grid, start, goal = generate_grid(
        args.height,
        args.width,
        obstacle_probability=args.obstacle_probability,
        rng=np.random.default_rng(args.seed),
    )
    env = GridWorldEnv(grid, start, goal, max_steps=grid.size * 4)
    agent = AStarAgent()
    visualizer = plot_grid(env.get_state()) if args.visualize else None
    total_decision_time = 0.0

    while not env.done:
        state = env.get_state()
        started = time.perf_counter()
        action = agent.select_action(state.grid, state.position, state.goal)
        total_decision_time += time.perf_counter() - started
        if action is None:
            print("A* could not find a path.")
            break
        next_state, _, _, _ = env.step(action)
        if visualizer is not None:
            path = agent.plan_path(next_state.grid, next_state.position, next_state.goal)
            visualizer.update(next_state, path)
            plt.pause(0.15)

    final_state = env.get_state()
    print(
        f"Success: {final_state.position == goal}; steps: {final_state.steps}; "
        f"collisions: {final_state.collisions}; "
        f"mean decision time: {total_decision_time / max(final_state.steps, 1):.6f}s"
    )
    if visualizer is not None:
        plt.show()


if __name__ == "__main__":
    main()
