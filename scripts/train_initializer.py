#!/usr/bin/env python3
"""Train the geometry-conditioned initializer from an NPZ label file."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from geometry_vqe import GeometryConditionedInitializer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/initializer.pt"))
    parser.add_argument("--atomic-numbers", type=int, nargs="+")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--message-layers", type=int, default=6)
    parser.add_argument("--radial-basis", type=int, default=50)
    parser.add_argument("--radial-start", type=float, default=0.0)
    parser.add_argument("--radial-stop", type=float, default=5.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_labels(
    path: Path,
    atomic_numbers: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    missing = {"R", "theta"} - set(archive.files)
    if missing:
        raise KeyError(f"{path} is missing required arrays: {sorted(missing)}")

    positions = np.asarray(archive["R"], dtype=np.float32)
    angles = np.asarray(archive["theta"], dtype=np.float32)
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("R must have shape (samples, atoms, 3)")
    if angles.ndim != 2 or angles.shape[0] != positions.shape[0]:
        raise ValueError("theta must have shape (samples, parameters)")

    if "Z" in archive.files:
        numbers = np.asarray(archive["Z"], dtype=np.int64)
        if numbers.ndim == 1:
            numbers = np.repeat(numbers[None, :], len(positions), axis=0)
    elif atomic_numbers is not None:
        if len(atomic_numbers) != positions.shape[1]:
            raise ValueError("--atomic-numbers must contain one value per atom")
        numbers = np.repeat(
            np.asarray(atomic_numbers, dtype=np.int64)[None, :],
            len(positions),
            axis=0,
        )
    else:
        raise ValueError("provide an array Z in the archive or use --atomic-numbers")

    if numbers.shape != positions.shape[:2]:
        raise ValueError("Z must have shape (atoms,) or (samples, atoms)")
    return positions, angles, numbers


def wrapped_angle_mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    difference = torch.atan2(
        torch.sin(prediction - target),
        torch.cos(prediction - target),
    )
    return difference.abs().mean()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.test_fraction < 1.0:
        raise ValueError("--test-fraction must lie between zero and one")
    set_seed(args.seed)

    positions, angles, atomic_numbers = load_labels(args.data, args.atomic_numbers)
    order = np.random.permutation(len(positions))
    test_size = max(1, int(round(args.test_fraction * len(order))))
    test_indices = order[:test_size]
    train_indices = order[test_size:]
    if not len(train_indices):
        raise ValueError("the training split is empty")

    target_circular = np.stack([np.sin(angles), np.cos(angles)], axis=-1)
    train_data = TensorDataset(
        torch.from_numpy(atomic_numbers[train_indices]),
        torch.from_numpy(positions[train_indices]),
        torch.from_numpy(target_circular[train_indices]),
    )
    loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeometryConditionedInitializer(
        angles.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.message_layers,
        num_radial_basis=args.radial_basis,
        radial_start=args.radial_start,
        radial_stop=args.radial_stop,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_samples = 0
        for numbers, coordinates, target in loader:
            numbers = numbers.to(device)
            coordinates = coordinates.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            prediction = model(numbers, coordinates)
            loss = criterion(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(numbers)
            total_samples += len(numbers)

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} loss={total_loss / total_samples:.8f}")

    model.eval()
    with torch.no_grad():
        numbers_test = torch.from_numpy(atomic_numbers[test_indices]).to(device)
        positions_test = torch.from_numpy(positions[test_indices]).to(device)
        angles_test = torch.from_numpy(angles[test_indices]).to(device)
        prediction = model.predict_angles(numbers_test, positions_test)
        test_mae = float(wrapped_angle_mae(prediction, angles_test).item())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_configuration": model.configuration(),
            "source_data": str(args.data),
            "split_seed": args.seed,
            "test_indices": test_indices,
            "test_wrapped_angle_mae_rad": test_mae,
        },
        args.output,
    )
    print(f"test_wrapped_angle_mae_rad={test_mae:.8f}")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
