"""Distance-based graph model for predicting periodic circuit parameters."""

from __future__ import annotations

import torch
from torch import nn


class GaussianRadialBasis(nn.Module):
    """Expand pair distances in equally spaced Gaussian basis functions."""

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 5.0,
        num_basis: int = 50,
    ) -> None:
        super().__init__()
        if stop <= start:
            raise ValueError("stop must be greater than start")
        if num_basis < 2:
            raise ValueError("num_basis must be at least two")

        centers = torch.linspace(start, stop, num_basis)
        spacing = (stop - start) / (num_basis - 1)
        self.coefficient = -0.5 / spacing**2
        self.register_buffer("centers", centers)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        offsets = distances.unsqueeze(-1) - self.centers
        return torch.exp(self.coefficient * offsets.square())


class RadialMessageBlock(nn.Module):
    """Scalar radial message-passing block with a residual update."""

    def __init__(self, hidden_dim: int, num_basis: int) -> None:
        super().__init__()
        self.radial_projection = nn.Linear(num_basis, hidden_dim)
        self.node_projection = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()

    def forward(
        self,
        node_features: torch.Tensor,
        radial_features: torch.Tensor,
    ) -> torch.Tensor:
        filters = self.activation(self.radial_projection(radial_features))
        neighbor_features = self.node_projection(node_features).unsqueeze(1)
        messages = filters * neighbor_features
        aggregated = messages.sum(dim=2)
        update = self.output_projection(aggregated)
        return self.activation(self.normalization(node_features + update))


class GeometryConditionedInitializer(nn.Module):
    """Map molecular geometry to a vector of periodic circuit parameters.

    The input representation uses only atomic numbers and pair distances. The
    resulting map is invariant to translations and rotations of the Cartesian
    coordinates. The final two-channel representation treats every predicted
    parameter as a point on the unit circle.
    """

    def __init__(
        self,
        num_parameters: int,
        *,
        hidden_dim: int = 128,
        num_layers: int = 6,
        num_radial_basis: int = 50,
        radial_start: float = 0.0,
        radial_stop: float = 5.0,
        max_atomic_number: int = 9,
    ) -> None:
        super().__init__()
        if num_parameters < 1:
            raise ValueError("num_parameters must be positive")
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if max_atomic_number < 1:
            raise ValueError("max_atomic_number must be positive")

        self.num_parameters = int(num_parameters)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.num_radial_basis = int(num_radial_basis)
        self.radial_start = float(radial_start)
        self.radial_stop = float(radial_stop)
        self.max_atomic_number = int(max_atomic_number)

        self.atomic_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.radial_basis = GaussianRadialBasis(
            radial_start,
            radial_stop,
            num_radial_basis,
        )
        self.message_blocks = nn.ModuleList(
            RadialMessageBlock(hidden_dim, num_radial_basis)
            for _ in range(num_layers)
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * num_parameters),
        )

    def _validate_inputs(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        if atomic_numbers.ndim != 2:
            raise ValueError("atomic_numbers must have shape (batch, atoms)")
        if positions.ndim != 3 or positions.shape[-1] != 3:
            raise ValueError("positions must have shape (batch, atoms, 3)")
        if positions.shape[:2] != atomic_numbers.shape:
            raise ValueError("atomic_numbers and positions must share batch and atom dimensions")
        if atomic_numbers.numel() and (
            atomic_numbers.min() < 0 or atomic_numbers.max() > self.max_atomic_number
        ):
            raise ValueError("atomic number outside the configured embedding range")

    def encode(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-atom scalar graph features."""

        self._validate_inputs(atomic_numbers, positions)
        atomic_numbers = atomic_numbers.to(dtype=torch.long)
        positions = positions.to(dtype=self.atomic_embedding.weight.dtype)

        node_features = self.atomic_embedding(atomic_numbers)
        distances = torch.cdist(positions, positions)
        radial_features = self.radial_basis(distances)

        for block in self.message_blocks:
            node_features = block(node_features, radial_features)
        return node_features

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return sine/cosine channels with shape `(batch, parameters, 2)`."""

        graph_features = self.encode(atomic_numbers, positions).sum(dim=1)
        output = self.readout(graph_features)
        return output.reshape(-1, self.num_parameters, 2)

    def predict_angles(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Return circuit angles in the principal interval `[-pi, pi]`."""

        circular = self(atomic_numbers, positions)
        return torch.atan2(circular[..., 0], circular[..., 1])

    def configuration(self) -> dict[str, int | float]:
        """Return constructor arguments suitable for a portable checkpoint."""

        return {
            "num_parameters": self.num_parameters,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_radial_basis": self.num_radial_basis,
            "radial_start": self.radial_start,
            "radial_stop": self.radial_stop,
            "max_atomic_number": self.max_atomic_number,
        }
