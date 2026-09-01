"""Per-run seed management. Seeds [0..4] by default."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Applies to: Python random, NumPy, PyTorch.
    Attempts PennyLane if available.
    """
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Try PennyLane if available
    try:
        import pennylane as qml

        qml.seed(seed)
    except (ImportError, AttributeError):
        pass


def default_seed_range() -> range:
    """Default 5-seed range [0..4]."""
    return range(5)
