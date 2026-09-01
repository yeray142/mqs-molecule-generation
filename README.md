# MQS Molecule Generation

Latent Style-based Quantum Wasserstein GAN for Drug Design — a reproduction of [arXiv:2603.22399](https://arxiv.org/abs/2603.22399) with PennyLane + PyTorch, benchmarked via QPUBench.

## Overview

This project implements a quantum generative adversarial network (QGAN) for molecular generation, combining:

- **Quantum circuits** (PennyLane) for the generator and discriminator
- **Entanglement-based latent style encoding** for controlled molecule generation
- **Wasserstein GAN loss** for improved training stability
- **RDKit** for molecular validity, SA scores, and diversity metrics

> **Status**: Scaffold with stub training loops. No actual quantum execution in core install.

## Installation

```sh
# Core dependencies
uv sync

# With PennyLane (quantum circuit support)
uv sync --extra pennylane

# With GPU acceleration
uv sync --extra gpu

# With hardware provider support (IBM, IQM)
uv sync --extra hardware
```

### Optional Extras

| Extra | Description |
|-------|-------------|
| `pennylane` | PennyLane + Qiskit + lightning.qubit |
| `gpu` | PyTorch with CUDA support |
| `hardware` | IBM Quantum & IQM hardware access |
| `moses` | Moses molecular (benchmarking) |
| `storage` | AWS S3 + Pandas for result storage |

## Usage

### Training

```sh
# Dry run (validates pipeline, no quantum execution)
mqs-molecule-generation train --dry-run --seeds 0 --ansatz simple --n-qubits 2 --n-layers 1

# Full benchmark chain with BEL ansatz
mqs-molecule-generation train --dry-run --chain --seeds 0-4 --ansatz bel --n-qubits 4 --n-layers 2

# Training with custom parameters
mqs-molecule-generation train --ansatz simple --n-qubits 4 --n-layers 3 --max-iters 1000 --readout dual
```

### Evaluation

```sh
# Run full evaluation chain (SA score, diversity, validity)
mqs-molecule-generation eval --chain

# Dry run validation
mqs-molecule-generation eval --dry-run
```

### Manifest

Print per-run metadata (git SHA, lockfile hash, package versions):

```sh
mqs-molecule-generation manifest
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--ansatz` | Ansatz type: `simple` or `bel` | `simple` |
| `--n-qubits` | Number of qubits | `4` |
| `--n-layers` | Number of circuit layers | `2` |
| `--max-iters` | Max training iterations | `100` |
| `--seeds` | Seed range (e.g., `0-4` or `0,2,5`) | `0-4` |
| `--readout` | Discriminator readout: `single` or `dual` | `single` |
| `--dry-run` | Validate without quantum execution | `False` |
| `--chain` | Run full evaluation chain | `False` |

## Circuit Invariants

From arXiv:2603.22399:

```
simple: n_angles = n_qb * n_l
bel:    n_angles = n_qb * (5*n_l + 1)
params  = 2 * n_angles  (W and b per angle)
D_l     = n_qb (single) | 2*n_qb (dual)
⟨Z₀⟩    = -1.00 (simple) | -0.26 (BEL)
```

## Project Structure

```
src/mqs_molecule_generation/
├── circuits/          # Ansatz builders, discriminator, generator
├── adapters/          # PennyLane backend adapter (deferred imports)
├── metrics/           # SA score, diversity, validity
├── training/          # Training loop stubs (NotImplementedError)
├── cli/               # Click CLI commands
├── utils/             # Manifest, seeding
└── hydra/configs/     # Hydra configuration tree
```

## Testing

```sh
# Run all tests (except slow/hardware)
uv run pytest -m "not slow and not hardware" tests/

# Lint and type-check
uv run ruff check src/
uv run mypy src/
```

## Documentation

- [docs/paper_mapping.md](docs/paper_mapping.md) — Paper sections to modules mapping
- [CLAUDE.md](CLAUDE.md) — Developer documentation

## License

This project is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.
