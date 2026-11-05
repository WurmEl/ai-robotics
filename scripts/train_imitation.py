"""Train a CNN policy from A* behavior-cloning examples."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from navigation.rl.network import CNNPolicyNetwork


def _load_dataset(path: str | Path, height: int, width: int) -> TensorDataset:
    with np.load(path) as data:
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.int64)
    expected_shape = (3, height, width)
    if observations.ndim != 4 or tuple(observations.shape[1:]) != expected_shape:
        raise ValueError(
            f"Expected observations with shape (N, {expected_shape}), got {observations.shape}."
        )
    if actions.ndim != 1 or len(actions) != len(observations):
        raise ValueError("Dataset actions must be a one-dimensional array matching observations.")
    if len(actions) == 0 or np.any((actions < 0) | (actions >= 4)):
        raise ValueError("Dataset must contain valid non-empty action labels.")
    return TensorDataset(torch.from_numpy(observations), torch.from_numpy(actions))


def _evaluate(
    model: CNNPolicyNetwork,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for observations, actions in loader:
            observations = observations.to(device)
            actions = actions.to(device)
            logits = model(observations)
            total_loss += float(loss_function(logits, actions).item()) * len(actions)
            correct += int((logits.argmax(dim=1) == actions).sum().item())
            total += len(actions)
    return total_loss / total, correct / total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--validation-data", default=None)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive.")
    if args.height <= 0 or args.width <= 0:
        parser.error("height and width must be positive.")

    torch.manual_seed(args.seed)
    rng = torch.Generator().manual_seed(args.seed)
    train_dataset = _load_dataset(args.train_data, args.height, args.width)
    validation_dataset = (
        _load_dataset(args.validation_data, args.height, args.width)
        if args.validation_data is not None
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=rng,
    )
    validation_loader = (
        DataLoader(validation_dataset, batch_size=args.batch_size)
        if validation_dataset is not None
        else None
    )
    device = torch.device("cpu")
    model = CNNPolicyNetwork((3, args.height, args.width)).to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model_path = Path(args.model_path)
    latest_path = model_path.with_name(model_path.stem + "-latest" + model_path.suffix)
    best_path = model_path.with_name(model_path.stem + "-best" + model_path.suffix)
    log_path = Path(args.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "elapsed_seconds",
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    ]
    best_accuracy = -1.0
    started = time.perf_counter()

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            for observations, actions in train_loader:
                logits = model(observations.to(device))
                loss = loss_function(logits, actions.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(actions)
                correct += int((logits.argmax(dim=1) == actions).sum().item())
                total += len(actions)
            train_loss = total_loss / total
            train_accuracy = correct / total
            if validation_loader is None:
                validation_loss = float("nan")
                validation_accuracy = float("nan")
                score = train_accuracy
            else:
                validation_loss, validation_accuracy = _evaluate(
                    model, validation_loader, loss_function, device
                )
                score = validation_accuracy
            checkpoint = {
                "map_shape": (args.height, args.width),
                "policy": model.state_dict(),
                "network": "cnn_policy",
                "epoch": epoch,
                "seed": args.seed,
            }
            torch.save(checkpoint, latest_path)
            if score > best_accuracy:
                best_accuracy = score
                torch.save(checkpoint, best_path)
            writer.writerow(
                {
                    "epoch": epoch,
                    "elapsed_seconds": time.perf_counter() - started,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "validation_loss": validation_loss,
                    "validation_accuracy": validation_accuracy,
                }
            )
            log_file.flush()
            print(
                f"Epoch {epoch}/{args.epochs} | train loss: {train_loss:.4f} | "
                f"train accuracy: {train_accuracy:.3f} | validation accuracy: "
                f"{validation_accuracy:.3f}"
            )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "map_shape": (args.height, args.width),
            "policy": model.state_dict(),
            "network": "cnn_policy",
            "epoch": args.epochs,
            "seed": args.seed,
        },
        model_path,
    )
    print(f"Saved final model to {model_path}; best model to {best_path}")


if __name__ == "__main__":
    main()
