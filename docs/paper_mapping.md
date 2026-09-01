# Paper Mapping: arXiv:2603.22399 -> Module Status

This document maps sections of arXiv:2603.22399 ("Latent Style-based Quantum Wasserstein GAN for Drug Design") to the implementation modules in this repository.

## Paper Sections

| Paper Section | Topic | Module | Status |
|--------------|-------|--------|--------|
| §1 | Introduction | - | - |
| §2 | Background: Quantum GANs | - | - |
| §3 | Generator Architecture | `circuits/generator.py` | STUB |
| §3.1 | Simple Ansatz | `circuits/ansatz.py` | DONE |
| §3.2 | BEL (Basic Entangling Layer) Ansatz | `circuits/ansatz.py` | DONE |
| §4 | Discriminator & Training | `training/latent_style_gan.py` | STUB |
| §4.1 | Wasserstein GAN Objective | `training/losses.py` | STUB |
| §4.2 | Gradient Penalty | `training/losses.py` | STUB |
| §5 | Molecular Generation | `training/latent_style_gan.py` | STUB |
| §6 | Experiments | - | - |
| §6.1 | Datasets | `metrics/` | PARTIAL |
| §7 | Conclusion | - | - |

## Circuit Invariants (Table from Paper)

The following invariants are verified in `tests/test_ansatz.py`:

```
simple: n_angles = n_qb * n_l
bel:    n_angles = n_qb * (5*n_l + 1)
params  = 2 * n_angles          (W and b per angle, Eq. 4)
D_l     = n_qb (single) | 2*n_qb (dual)
⟨Z₀⟩    = -1.00 (simple) | -0.26 (BEL)  (Table 14)
```

## Module Status Details

### `circuits/ansatz.py` - DONE
- `AnsatzParams`: Dataclass verifying all parameter counts
- `build_simple_ansatz_qasm()`: Simple ansatz QASM generator
- `build_bel_ansatz_qasm()`: BEL ansatz QASM generator
- `make_ansatz_circuit()`: Combined circuit builder
- Verified against all 11 rows of paper's N_params table

### `circuits/discriminator.py` - DONE
- `build_discriminator_circuit()`: Single/dual readout discriminator
- `get_discriminator_readout_dim()`: D_l dimension verification

### `circuits/generator.py` - DONE
- `build_generator_circuit()`: Generator using ansatz
- `build_full_generator_discriminator_circuit()`: Combined builder

### `adapters/pennylane_latent_style_adapter.py` - DONE
- Implements `BackendAdapter` protocol
- Deferred imports inside `run()`
- Returns `FAILED` with error_message on recoverable errors

### `metrics/sa_score.py` - DONE
- SA score from `RDConfig.RDContribDir/SA_Score/sascore.py`
- Lazy loading with graceful error handling

### `metrics/diversity.py` - DONE
- Morgan fingerprint Tanimoto diversity

### `metrics/validity.py` - DONE
- RDKit SMILES validity checks

### `training/latent_style_gan.py` - STUB
- `LatentStyleGANTraining` class with NotImplementedError stubs
- `run()` works in dry-run mode (dummy metrics)
- `_discriminator_step()`: TODO(paper §4)
- `_generator_step()`: TODO(paper §4)
- `_generate_samples()`: TODO(paper §3)

### `training/losses.py` - STUB
- `wasserstein_loss()`: TODO(paper §4)
- `gradient_penalty()`: TODO(paper §4.2)
- `generator_loss()`: TODO(paper §4)

## Test Coverage

| Test File | Coverage |
|-----------|----------|
| `tests/test_ansatz.py` | All 11 N_params rows, invariants, QASM generation |
| `tests/test_adapter.py` | Adapter validation, FAILED status, mock paths |
| `tests/test_metrics.py` | SA score, diversity, validity |
| `tests/test_cli.py` | All CLI commands, --dry-run, --chain |
| `tests/test_manifest.py` | Manifest generation |

## QPUBench Integration

The `LatentStylePennyLaneAdapter` implements the `BackendAdapter` protocol:
- `spec`: BackendSpec with lightning.qubit
- `validate()`: Returns warnings, raises ValueError on hard errors
- `run()`: Returns QuantumResult with SUCCEEDED/FAILED status

Integration with BenchmarkRunner:
```python
from qpubench import BenchmarkRunner
from mqs_molecule_generation.adapters.pennylane_latent_style_adapter import LatentStylePennyLaneAdapter

runner = BenchmarkRunner(store="results.ndjson")
runner.register(LatentStylePennyLaneAdapter(n_qubits=4), name="latent_style_pennylane")
```
