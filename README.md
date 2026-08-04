# Geometry-Conditioned VQE Initialization

Reference code accompanying the manuscript
"Geometry-Conditioned Basin Initialization for Variational Quantum
Eigensolvers in Strongly Correlated Molecules."

The repository contains a compact implementation of the geometry-to-circuit
initializer and a training entry point for geometry/parameter label files.

## Repository layout

- `src/geometry_vqe/`: distance/RBF graph initializer with a circular angle readout.
- `scripts/train_initializer.py`: training entry point for geometry/parameter label files.
- `tests/`: checks for rigid-motion invariance and output shapes.

## Installation

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Run the test suite:

```bash
pytest
```

## Core model

`GeometryConditionedInitializer` maps atomic numbers and Cartesian coordinates
to periodic circuit parameters. Molecular geometry enters only through pair
distances expanded in Gaussian radial basis functions. The graph features are
therefore invariant to rigid translations and rotations. Each circuit angle is
represented by two output channels and recovered with `atan2`.
The default configuration uses 128 scalar channels, six message blocks, and
50 Gaussian radial functions on the interval `[0, 5]`.

The expected tensor shapes are:

- atomic numbers: `(batch, atoms)`
- Cartesian coordinates: `(batch, atoms, 3)`
- circular output: `(batch, parameters, 2)`
- predicted angles: `(batch, parameters)`

## Training interface

The training script expects a NumPy archive with arrays `R` and `theta`.
`R` must have shape `(samples, atoms, 3)` and use the same length unit as the
chosen radial-basis interval. `theta` must have shape `(samples, parameters)`. Atomic
numbers can be stored as `Z` or supplied on the command line.

```bash
python scripts/train_initializer.py \
  --data path/to/labels.npz \
  --atomic-numbers 7 7 \
  --epochs 300 \
  --output checkpoints/n2_initializer.pt
```

The checkpoint records the model configuration, learned weights, split seed,
and held-out wrapped-angle error.

## Citation

Citation metadata are provided in `CITATION.cff`. Please cite the associated
article and the archived repository record when using this material.

## License

The code in this repository is released under the MIT License.
