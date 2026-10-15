"""Run a trained DQN greedily on one evaluation map and visualize its trajectory.

Use the same --height/--width/--seed/--obstacle-probability as scripts/evaluate.py and pick
the evaluation episode with --map-index to replay exactly that map.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from navigation.agents.dqn_agent import DQNAgent
from navigation.env.grid_world import GridWorldEnv
from navigation.env.map_generator import generate_grid
from navigation.visualization.plot_grid import plot_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--map-index", type=int, default=1, help="1-based evaluation episode")
    parser.add_argument("--model-path", default="models/dqn.pt")
    parser.add_argument("--pause", type=float, default=0.05, help="seconds between frames")
    parser.add_argument(
        "--save-path", default=None, help="save the final frame instead of animating"
    )
    args = parser.parse_args()
    if args.map_index <= 0:
        parser.error("--map-index must be positive.")

    # Regenerate maps in the same order as scripts/evaluate.py.
    rng = np.random.default_rng(args.seed)
    for _ in range(args.map_index):
        grid, start, goal = generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            rng=rng,
        )

    agent = DQNAgent((args.height, args.width), seed=args.seed)
    agent.load(args.model_path)
    env = GridWorldEnv(grid, start, goal, max_steps=args.height * args.width * 4)
    state = env.reset()
    trail = [state.position]
    animate = args.save_path is None
    visualizer = plot_grid(state, trail)

    while not env.done:
        action = agent.select_action(env.get_state(), explore=False)
        state, _, _, _ = env.step(action)
        if state.position != trail[-1]:
            trail.append(state.position)
        if animate:
            visualizer.update(state, trail)
            plt.pause(args.pause)

    visualizer.update(state, trail)
    success = state.position == goal
    visualizer.ax.set_title(
        f"DQN {'reached goal' if success else 'FAILED'} | steps: {state.steps} | "
        f"collisions: {state.collisions} | yellow = trail"
    )
    print(f"Success: {success}; steps: {state.steps}; collisions: {state.collisions}")
    if animate:
        plt.show()
    else:
        visualizer.figure.savefig(args.save_path, dpi=150)
        print(f"Saved {args.save_path}")


if __name__ == "__main__":
    main()
