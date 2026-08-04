import math

import torch

from geometry_vqe import GeometryConditionedInitializer


def test_output_shapes() -> None:
    model = GeometryConditionedInitializer(
        num_parameters=12,
        hidden_dim=24,
        num_layers=2,
        num_radial_basis=8,
    )
    atomic_numbers = torch.tensor([[7, 7], [1, 8]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]],
            [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        ]
    )

    circular = model(atomic_numbers, positions)
    angles = model.predict_angles(atomic_numbers, positions)

    assert circular.shape == (2, 12, 2)
    assert angles.shape == (2, 12)
    assert torch.all(angles <= math.pi)
    assert torch.all(angles >= -math.pi)


def test_rigid_motion_invariance() -> None:
    torch.manual_seed(7)
    model = GeometryConditionedInitializer(
        num_parameters=8,
        hidden_dim=20,
        num_layers=2,
        num_radial_basis=10,
    )
    model.eval()

    atomic_numbers = torch.tensor([[8, 1, 1]])
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.2, 0.7, 0.1]]],
        dtype=torch.float32,
    )
    angle = torch.tensor(0.73)
    rotation = torch.tensor(
        [
            [torch.cos(angle), -torch.sin(angle), 0.0],
            [torch.sin(angle), torch.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translated = positions @ rotation.T + torch.tensor([2.4, -1.3, 0.8])

    with torch.no_grad():
        original = model(atomic_numbers, positions)
        transformed = model(atomic_numbers, translated)

    torch.testing.assert_close(original, transformed, rtol=1.0e-5, atol=1.0e-6)
