"""Train a DQN on static maps of one fixed size."""

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
    parser.add_argument(
        "--training-mode",
        choices=("random", "fixed"),
        default="random",
        help="use a new map each episode or reuse one map for every episode",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive.")

    rng = np.random.default_rng(args.seed)
    agent = DQNAgent((args.height, args.width), seed=args.seed)
    max_steps = args.max_steps if args.max_steps is not None else args.height * args.width * 4
    fixed_map = (
        generate_grid(
            args.height,
            args.width,
            obstacle_probability=args.obstacle_probability,
            rng=rng,
        )
        if args.training_mode == "fixed"
        else None
    )
    started = time.perf_counter()
    recent_successes = 0
    recent_rewards: list[float] = []
    recent_steps: list[int] = []

    for episode in range(1, args.episodes + 1):
        if fixed_map is None:
            map_data = generate_grid(
                args.height,
                args.width,
                obstacle_probability=args.obstacle_probability,
                rng=rng,
            )
        else:
            map_data = fixed_map
        grid, start, goal = map_data
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

        recent_successes += int(state.position == goal)
        recent_rewards.append(cumulative_reward)
        recent_steps.append(state.steps)
        if episode % 50 == 0 or episode == 1:
            print(
                f"Episode {episode}/{args.episodes} | "
                f"success(last {len(recent_rewards)}): "
                f"{recent_successes / len(recent_rewards):.2f} | "
                f"avg reward: {np.mean(recent_rewards):.2f} | "
                f"avg steps: {np.mean(recent_steps):.1f} | "
                f"epsilon: {agent.epsilon:.3f}"
            )
            recent_successes = 0
            recent_rewards.clear()
            recent_steps.clear()

    training_runtime = time.perf_counter() - started
    agent.save(args.model_path)
    print(
        f"Saved model to {args.model_path}; training runtime "
        f"{training_runtime:.2f}s over {args.episodes} episodes."
    )


if __name__ == "__main__":
    main()
