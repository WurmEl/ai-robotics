# AI Robotics Grid-World Navigation

This project compares classical global search with learned reactive navigation:

> How do classical global search and learned reactive navigation compare as environment size and obstacle dynamics increase?

The initial milestone uses **static** maps. Both agents interact with the same
`GridWorldEnv`, and both receive the full current map, robot position, and goal:

- **A\*** plans online from the global grid at every environment step. It does
  not require training.
- **DQN** receives a full-map observation with separate obstacle, robot, and
  goal channels (`3 x height x width`). It learns a policy; training is more
  expensive, while action inference is inexpensive. Its MLP currently expects
  one fixed map size per model.

Neither agent receives information about future obstacle movement. The
observation encoder is separate from the environment and network, leaving room
for later local-window, dynamic-obstacle-channel, or CNN-based observations.
Moving obstacles and large-scale experiments are not implemented yet.

## Setup

Python 3.10 or newer is required. Install the package and development test
dependency from the repository root:

```powershell
python -m pip install -e ".[dev]"
```

## Examples

Generate and save 20 reproducible 10-by-12 maps:

```powershell
python scripts/generate_maps.py --height 10 --width 12 --maps 20 --seed 7
```

Show a generated map with repeated A* replanning:

```powershell
python scripts/run_astar.py --height 15 --width 15 --seed 7 --visualize
```

Train a DQN on fixed-size maps, then evaluate both agents on the same test
maps:

```powershell
python scripts/train_dqn.py --height 10 --width 10 --episodes 500 --seed 7 --model-path models/dqn.pt
python scripts/evaluate.py --agent both --height 10 --width 10 --episodes 20 --seed 11 --model-path models/dqn.pt
```

Evaluation writes per-episode success, collisions, steps, path length, runtime,
mean decision time, and cumulative reward to `results/evaluation.csv`.
Generated datasets, models, and results are not committed by default.

Run the tests with:

```powershell
python -m pytest
```
