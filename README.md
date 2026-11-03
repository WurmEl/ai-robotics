# AI Robotics Navigation Project

This project compares classical A* search with Deep Q-Learning (DQN) for
mobile-robot navigation in 2D grid-world environments.

## Research question

> How do classical global search and deep reinforcement learning compare as
> environment size and obstacle dynamics increase?

The current milestone focuses on reproducible **static** environments. Both
agents use the same `GridWorldEnv`; A* has direct access to the global grid,
while DQN receives an encoded observation of that same current state.

## Current status

Implemented:

- reproducible random map generation with valid start and goal positions;
- shared static grid-world environment;
- repeated online A* baseline;
- DQN with a full-map observation;
- replay buffer, target network, epsilon-greedy action selection, and
  checkpoint save/load;
- fixed-map DQN training mode for debugging;
- evaluation metrics and CSV output;
- simple step-by-step matplotlib visualizations;
- automated tests and Ruff/mypy configuration.

The current DQN sanity check learns a fixed 5x5 map and can reach the goal
greedily after training. This is an overfitting/debugging milestone, not
evidence that DQN generalizes to unseen maps.

Not implemented yet:

- dynamic or moving obstacles;
- systematic larger-map experiments;
- multi-map generalization experiments;
- CNN-based DQN;
- local-observation DQN;
- final A* versus DQN experiments under changing obstacle dynamics.

## Project structure

```text
src/navigation/
├── env/
│   ├── grid_world.py
│   └── map_generator.py
├── agents/
│   ├── astar_agent.py
│   ├── dqn_agent.py
│   └── observation.py
├── rl/
│   ├── network.py
│   └── replay_buffer.py
├── evaluation/
│   └── metrics.py
└── visualization/
    └── plot_grid.py

scripts/
├── generate_maps.py
├── generate_datasets.py
├── run_astar.py
├── train_dqn.py
├── run_dqn.py
├── evaluate.py
└── summarize_results.py

tests/
├── test_astar.py
├── test_dqn.py
├── test_grid_world.py
├── test_map_generator.py
├── test_metrics.py
└── test_visualization.py
```

## Setup

The project requires Python 3.10–3.14 and
[uv](https://docs.astral.sh/uv/). From the repository root:

```sh
uv sync
```

Run the tests and optional development checks:

```sh
uv run pytest
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run mypy src
```

Standard editable installation is also supported:

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

## Quick start

### Generate maps

```sh
uv run python scripts/generate_maps.py --height 10 --width 12 --maps 20 --seed 7
```

Generate separate, non-overlapping layout datasets for experiments:

```sh
uv run python scripts/generate_datasets.py --height 10 --width 10 --train 50 --validation 100 --test 300 --seed 7
```

The command writes obstacle-only layouts to
`data/train/layouts.pkl`, `data/validation/layouts.pkl`, and
`data/test/layouts.pkl`. The three sets use separate deterministic seeds.

### Run repeated A*

Use `--visualize` to display the map while A* replans after each environment
step:

```sh
uv run python scripts/run_astar.py --height 15 --width 15 --seed 7 --visualize
```

### Train DQN on one fixed map

Start with fixed-map mode to verify that the DQN can learn a simple task:

```sh
uv run python scripts/train_dqn.py --height 5 --width 5 --episodes 300 --obstacle-probability 0.1 --seed 7 --training-mode fixed --max-steps 50 --model-path models/fixed.pt
```

The trainer reports rolling:

- success rate;
- average reward;
- average episode length;
- epsilon.

The generated model is ignored by Git.

### Train DQN on a layout pool

Pool mode samples one layout from a fixed training set and samples a new valid
start/goal pair on that layout for each episode. It also creates a separate
validation layout set and evaluates greedily at each validation interval:

```sh
uv run python scripts/train_dqn.py --height 10 --width 10 --episodes 500 --pool-size 50 --validation-size 100 --seed 7 --training-mode pool --model-path models/pool.pt --log-path results/pool-training.csv --train-layouts-path data/train/pool-layouts.pkl --validation-layouts-path data/validation/pool-layouts.pkl
```

Training writes an episode-level CSV and saves:

- the final model;
- a periodic `-latest.pt` checkpoint;
- the best-validation `-best.pt` checkpoint.

The training CSV includes episode reward, success, steps, collisions, epsilon,
environment steps, elapsed time, and validation success/collision/path-ratio
columns at validation intervals.

For multi-map development runs, make the training controls explicit:

```sh
uv run python scripts/train_dqn.py --training-mode pool --height 10 --width 10 --episodes 500 --pool-size 50 --validation-size 100 --max-steps 120 --epsilon-decay-steps 100000 --learning-starts 2000 --learn-every 4 --seed 7 --model-path models/pool.pt --log-path results/pool-training.csv
```

`--learning-starts` delays gradient updates until enough transitions are in
the replay buffer. `--learn-every` controls how often updates occur after the
warm-up period. The defaults preserve the earlier behavior; the command above
uses more conservative settings for 10x10 pool experiments.

### Visualize a trained DQN

Replay the first map generated by the same seed:

```sh
uv run python scripts/run_dqn.py --height 5 --width 5 --obstacle-probability 0.1 --seed 7 --map-index 1 --model-path models/fixed.pt
```

To save a final frame instead of opening an animation:

```sh
uv run python scripts/run_dqn.py --height 5 --width 5 --obstacle-probability 0.1 --seed 7 --map-index 1 --model-path models/fixed.pt --save-path results/fixed-map.png
```

### Evaluate a trained DQN

Evaluation uses greedy DQN actions (`explore=False`):

```sh
uv run python scripts/evaluate.py --agent dqn --height 5 --width 5 --episodes 1 --obstacle-probability 0.1 --seed 7 --model-path models/fixed.pt --output results/fixed.csv
```

To compare A* and DQN on the same generated maps:

```sh
uv run python scripts/evaluate.py --agent both --height 10 --width 10 --episodes 20 --seed 11 --model-path models/fixed.pt --output results/evaluation.csv
```

Summarize an evaluation CSV:

```sh
uv run python scripts/summarize_results.py --input results/evaluation.csv
uv run python scripts/summarize_results.py --input results/evaluation.csv --per-map
```

## Environment and observations

The environment is a 2D NumPy grid:

- `0` = free cell;
- `1` = obstacle.

Actions are:

- `0` = up;
- `1` = down;
- `2` = left;
- `3` = right.

The environment handles movement, collisions, rewards, episode termination,
and the maximum episode length. It does not contain agent-specific or
PyTorch logic.

### A*

A* uses:

- the full current grid;
- the current robot position;
- the goal position;
- Manhattan distance as its heuristic.

The baseline recomputes a path at every environment step. This is redundant
for a static map, but preserves the repeated-planning structure needed for
later dynamic-environment experiments.

### DQN

The current DQN observation is a full-map tensor with shape
`3 x height x width`:

1. obstacle channel;
2. robot-position channel;
3. goal-position channel.

The observation is encoded separately from the environment. A small MLP
flattens this tensor and outputs four Q-values:

```text
Q(up), Q(down), Q(left), Q(right)
```

The current model supports one fixed map shape per checkpoint.

Training and evaluation accept `--network mlp` or `--network cnn`. The default
is `mlp`. The CNN is intentionally small:

```text
Conv2d(3, 32, 3, padding=1)
ReLU
Conv2d(32, 64, 3, padding=1)
ReLU
Flatten
Linear
ReLU
Linear(4)
```

Both networks use the same observation encoder, replay buffer, reward,
epsilon schedule, target-network update, and checkpoint interface. A
checkpoint records its network type and rejects loading into the other
architecture.

## Training modes

### `fixed`

One map is generated once and reused for every episode:

```sh
uv run python scripts/train_dqn.py --training-mode fixed --height 5 --width 5 --episodes 300 --seed 7 --model-path models/fixed.pt
```

Use this mode first to verify that the DQN implementation can learn at all.

### `random`

A new valid map is generated for every episode. This remains the default and
preserves the original training behavior:

```sh
uv run python scripts/train_dqn.py --training-mode random --height 10 --width 10 --episodes 500 --seed 7 --model-path models/random.pt
```

This is substantially harder because the agent must learn navigation and
generalization at the same time.

### Comparing MLP and CNN on the same experiment

The controlled CNN comparison keeps the MLP baseline settings unchanged:

```sh
uv run python scripts/train_dqn.py --height 10 --width 10 --episodes 1000 --pool-size 50 --validation-size 100 --obstacle-probability 0.1 --min-path-length 2 --seed 7 --network cnn --training-mode pool --max-steps 120 --epsilon-decay-steps 100000 --learning-starts 2000 --learn-every 4 --validation-interval 100 --checkpoint-interval 100 --model-path models/cnn-pool-seed7.pt --log-path results/cnn-pool-seed7-training.csv --train-layouts-path data/train/cnn-pool-seed7-layouts.pkl --validation-layouts-path data/validation/cnn-pool-seed7-layouts.pkl
```

Use the same command with `--network mlp` and different output filenames for
the MLP baseline. This comparison asks whether spatial convolutions improve
unseen 10x10-map generalization; it is not intended to make DQN outperform
A*.

### Optional curriculum and potential-based shaping

The sparse reward remains the default. For a controlled curriculum experiment,
enable both options explicitly:

```sh
uv run python scripts/train_dqn.py --height 10 --width 10 --episodes 1000 --pool-size 50 --validation-size 100 --seed 7 --network cnn --training-mode pool --curriculum --reward-shaping --shaping-alpha 0.1 --shaping-gamma 0.99 --max-steps 120 --epsilon-decay-steps 100000 --learning-starts 2000 --learn-every 4 --model-path models/cnn-curriculum.pt --log-path results/cnn-curriculum-training.csv
```

Curriculum stages use progressively longer start/goal paths and obstacle
probabilities. The shaping term is the potential-based Manhattan-distance
signal
`alpha * (gamma * Phi(next_state) - Phi(state))`, where
`Phi(state) = -ManhattanDistance(robot, goal)`. Evaluation continues to use
the normal environment reward. Training logs include base reward, shaped
training reward, curriculum stage, and fixed greedy seen/unseen diagnostics.

## Evaluation metrics

Evaluation CSV files include:

- stable `map_id`;
- seed and map dimensions;
- obstacle probability;
- BFS optimal path length;
- agent;
- success;
- collisions;
- steps;
- path length;
- episode runtime;
- mean decision time;
- cumulative reward.

The same generated maps are used for A* and DQN when `--agent both` is
selected. Training episodes and training runtime are reported by the training
command separately rather than being mixed with per-episode inference time.

To evaluate a saved held-out layout set, use `--layouts-path`. Set
`--task-seed` to reproduce the same start/goal tasks used by a training
validation run:

```sh
uv run python scripts/evaluate.py --agent both --height 10 --width 10 --episodes 100 --seed 8 --task-seed 9 --max-steps 120 --model-path models/pool-best.pt --layouts-path data/validation/layouts.pkl --output results/validation.csv
```

## Development roadmap

1. Fixed map, fixed start and goal — current sanity-check milestone.
2. Fixed map with varying starts.
3. Fixed map with varying starts and goals.
4. Small pool of static maps.
5. Unseen held-out static maps.
6. Larger map experiments.
7. Moving obstacles and repeated replanning under map updates.
8. Final A* versus DQN experiments.

Do not interpret success on one fixed map as generalization. The fixed-map
experiment only establishes that the current DQN pipeline can learn a simple
static navigation task.
