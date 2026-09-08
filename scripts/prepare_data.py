#!/usr/bin/env python
"""Prepare the MOSES subsample, or calibrate the filter set against the paper.

Two modes:

    # Identify which filter definition reproduces the paper's 0.734 (Table 3)
    uv run python scripts/00_prepare_data.py calibrate

    # Freeze the 12,000 / 4,087 splits, from the pinned config
    uv run python scripts/00_prepare_data.py prepare --config configs/data/moses_subset.yaml

Run ``calibrate`` first if the filter set is still in question. It writes no
split files; it only reports pass rates so the winning filter set can be
pinned in the config. Any flag passed on the command line overrides the same
field in ``--config``, which in turn overrides :class:`PrepareConfig`'s
built-in defaults -- so ``prepare --config moses_subset.yaml --seed 1`` reuses
everything else from the file but freezes a different seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqs_molecule_generation.data import filters
from mqs_molecule_generation.data.moses import (
    PrepareConfig,
    canonicalize,
    download,
    load_raw,
    prepare,
    subsample,
)

#: Paper Table 3, "Filters" row for the training set. See docs/paper_mapping.md
#: for why this is currently unreproduced (xfail'd, non-blocking).
TARGET_PASS_RATE = 0.734
DEFAULT_TOLERANCE = 0.01

_PREPARE_FIELDS = {f.name for f in fields(PrepareConfig)}
_PATH_FIELDS = {"raw_dir", "out_dir"}


def _load_yaml_overrides(config_path: Path | None) -> dict[str, Any]:
    """Load a Hydra-style config file, dropping the ``_target_`` marker."""
    if config_path is None:
        return {}
    data = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{config_path} must contain a mapping, got {type(data).__name__}")
    data.pop("_target_", None)
    unknown = set(data) - _PREPARE_FIELDS
    if unknown:
        raise ValueError(f"{config_path} has unrecognised field(s): {sorted(unknown)}")
    return data


def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Only the fields explicitly passed on the command line (SUPPRESS-defaulted)."""
    overrides = {k: v for k, v in vars(args).items() if k in _PREPARE_FIELDS}
    if getattr(args, "no_verify", False):
        overrides["verify_checksum"] = False
    return overrides


def _resolve_prepare_config(args: argparse.Namespace) -> PrepareConfig:
    """Layer defaults < ``--config`` file < explicit CLI flags."""
    merged: dict[str, Any] = asdict(PrepareConfig())
    merged.update(_load_yaml_overrides(getattr(args, "config", None)))
    merged.update(_cli_overrides(args))
    for key in _PATH_FIELDS:
        merged[key] = Path(merged[key])
    return PrepareConfig(**merged)


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Sweep every registered filter set and rank by distance to 0.734."""
    cfg = _resolve_prepare_config(args)
    raw = download(cfg.raw_dir / "dataset_v1.csv", verify=cfg.verify_checksum)
    frame = load_raw(raw)
    smiles = frame.loc[frame["SPLIT"] == cfg.train_split, "SMILES"].tolist()
    canonical = [c for c in (canonicalize(s) for s in smiles) if c is not None]
    pool = list(dict.fromkeys(canonical))
    sample = subsample(pool, cfg.n_train, np.random.default_rng(cfg.seed))

    print(f"Calibrating on {len(sample)} training molecules (seed={cfg.seed})")
    print(f"Paper target: {TARGET_PASS_RATE:.3f}\n")

    rows: list[tuple[str, float, float]] = []
    for name in filters.available():
        if name == "none":
            continue
        rate = filters.pass_rate(sample, name)
        rows.append((name, rate, abs(rate - TARGET_PASS_RATE)))
    rows.sort(key=lambda r: r[2])

    native_rate = next(r for r in rows if r[0] == "moses_native")[1]
    if native_rate > 0.97:
        print(
            "Note: moses_native (the real MOSES construction filter) landed at "
            f"{native_rate:.4f}, close to 1.0 as expected -- MOSES data was already "
            "filtered through this exact function when it was built. This confirms "
            "the paper's 0.734 is NOT the MOSES construction filter; it is either a "
            "stricter catalogue (see the RDKit rows below) or a metric defined "
            "differently than assumed. Re-read paper §2.3's 'Filters' definition "
            "and consider extending CATALOG_SETS before trusting any row here.\n"
        )

    width = max(len(r[0]) for r in rows)
    print(f"{'filter set':<{width}}  {'pass rate':>9}  {'|Δ|':>7}")
    print("-" * (width + 20))
    for name, rate, delta in rows:
        flag = "  <-- match" if delta <= args.tolerance else ""
        print(f"{name:<{width}}  {rate:>9.4f}  {delta:>7.4f}{flag}")

    best, best_rate, best_delta = rows[0]
    print()
    if best_delta <= args.tolerance:
        print(f"Pin filter_set: {best} (pass rate {best_rate:.4f})")
        exit_code = 0
    else:
        print(
            f"No filter set within {args.tolerance} of {TARGET_PASS_RATE}. "
            f"Closest: {best} at {best_rate:.4f}. See docs/paper_mapping.md -- "
            "this is a known, documented discrepancy, not necessarily a bug."
        )
        exit_code = 1

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "target": TARGET_PASS_RATE,
                    "tolerance": args.tolerance,
                    "seed": cfg.seed,
                    "n_molecules": len(sample),
                    "results": {name: rate for name, rate, _ in rows},
                    "best": best,
                },
                indent=2,
            )
        )
        print(f"Wrote {args.report}")
    return exit_code


def cmd_prepare(args: argparse.Namespace) -> int:
    """Freeze the train/validation subsamples and their manifest."""
    cfg = _resolve_prepare_config(args)
    manifest = prepare(cfg)
    print(json.dumps(asdict(manifest), indent=2))

    delta = abs(manifest.train_pass_rate - TARGET_PASS_RATE)
    if delta > args.tolerance:
        print(
            f"\nNote: train pass rate {manifest.train_pass_rate:.4f} is "
            f"{delta:.4f} from the paper's {TARGET_PASS_RATE}. This is a known, "
            "documented discrepancy (see docs/paper_mapping.md) -- non-blocking "
            "if filter_set=moses_native, which is expected to land near 1.0.",
            file=sys.stderr,
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    for name, handler in (("calibrate", cmd_calibrate), ("prepare", cmd_prepare)):
        p = sub.add_parser(name, help=(handler.__doc__ or "").splitlines()[0])
        p.add_argument(
            "--config",
            type=Path,
            default=None,
            help="YAML file, e.g. configs/data/moses_subset.yaml",
        )
        p.add_argument("--raw-dir", type=Path, default=argparse.SUPPRESS)
        p.add_argument("--seed", type=int, default=argparse.SUPPRESS)
        p.add_argument("--n-train", type=int, default=argparse.SUPPRESS)
        p.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
        p.add_argument(
            "--no-verify", action="store_true", default=False, help="skip the source checksum"
        )
        p.set_defaults(func=handler)

    sub.choices["calibrate"].add_argument("--report", type=Path, default=None)
    sub.choices["prepare"].add_argument("--out-dir", type=Path, default=argparse.SUPPRESS)
    sub.choices["prepare"].add_argument("--n-valid", type=int, default=argparse.SUPPRESS)
    sub.choices["prepare"].add_argument(
        "--filter-set", dest="filter_set", default=argparse.SUPPRESS
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())