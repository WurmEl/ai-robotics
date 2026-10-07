"""Train a DQN on randomly generated static maps of one fixed size."""

from __future__ import annotations

import argparse
import time

import numpy as np

from navigation.agents.dqn_agent import DQNAgent
from navigation.env.grid_world import GridWorldEnv
from navigation.env.map_generator import generate_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=10)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--obstacle-probability", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", default="models/dqn.pt")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive.")

    rng = np.random.default_rng(args.seed)
    agent = DQNAgent((args.height, args.width), seed=args.seed)
    max_steps = args.max_steps if args.max_steps is not None else args.height * args.width * 4
    started = time.perf_counter()

    for episode in range(1, args.episodes + 1):
        grid, start, goal = generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            rng=rng,
        )
        env = GridWorldEnv(grid, start, goal, max_steps=max_steps)
        state = env.reset()
        cumulative_reward = 0.0

        while not env.done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.observe(state, action, reward, next_state, done)
            agent.learn()
            cumulative_reward += reward
            state = next_state

        if episode % 50 == 0 or episode == 1:
            print(
                f"Episode {episode}/{args.episodes} | "
                f"reward {cumulative_reward:.2f} | epsilon {agent.epsilon:.3f}"
            )

    training_runtime = time.perf_counter() - started
    agent.save(args.model_path)
    print(
        f"Saved model to {args.model_path}; training runtime "
        f"{training_runtime:.2f}s over {args.episodes} episodes."
    )


if __name__ == "__main__":
    main()
