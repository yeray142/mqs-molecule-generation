#!/usr/bin/env python
"""Render generated molecules as a PNG/PDF grid for visual QC.

GAN mode (samples via a trained generator, decoded through its paired VAE):
    uv run python scripts/visualize_samples.py \\
        --generator-checkpoint results/significance_runs/nominal_gan_seed_0.generator.pt

VAE-prior mode (no generator -- samples directly from the VAE's own prior,
useful as a side-by-side reference for the GAN grids above):
    uv run python scripts/visualize_samples.py \\
        --vae-checkpoint results/vae_runs/tuned_n_ep_1000_seed_0.epoch0200.pt

In GAN mode, the paired VAE checkpoint and scenario are read automatically
from the companion `<run_name>.json` that scripts/train_gan.py writes next
to the generator checkpoint (it saves exactly this: which VAE checkpoint
and scenario were used) -- no need to re-specify them by hand, and no risk
of pairing a generator with the wrong VAE by mistake. --vae-checkpoint/
--vae-scenario override this if given explicitly, or if that JSON is
missing (e.g. an older run).

Generates in batches, oversampling as needed, until at least --n-molecules
valid ones are found (bounded by --max-attempts total samples) -- asking
for 12 molecules should reliably show 12, not fewer just because some
fraction of raw samples don't parse as valid SMILES. Each cell is labelled
with molecular weight, QED, and SA score -- the same three properties this
project has been tracking in the significance metrics throughout, so a
grid here and a metrics table elsewhere are talking about the same
quantities.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from rdkit import Chem, RDConfig, RDLogger
from rdkit.Chem import QED, Descriptors, Draw

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer
from mqs_molecule_generation.models.checkpoint_io import (
    load_tokenizer_for_checkpoint,
    load_vae_checkpoint,
)
from mqs_molecule_generation.models.classical_gan import Generator

RDLogger.DisableLog("rdApp.*")


def infer_generator_latent_dim(state_dict: dict) -> int:
    """Read latent_dim directly off the generator's own first-layer weight shape.

    model.0 is always Linear(latent_dim, 128) -- see models.classical_gan.Generator.
    Used as a fallback when no companion JSON is available to read it from.
    """
    return state_dict["model.0.weight"].shape[1]


def load_vae_flexibly(checkpoint: Path, vocab: Path | None, scenario: str | None, device):
    """Call checkpoint_io.load_vae_checkpoint correctly whichever signature is
    currently deployed (with or without a `scenario` parameter) -- this
    project's VAE loader was mid-refactor toward auto-detecting scenario
    from the checkpoint's own tensor shapes at time of writing, and this
    script shouldn't have to assume which version is installed.
    """
    tokenizer = load_tokenizer_for_checkpoint(checkpoint, vocab, Path("data/processed"))
    sig = inspect.signature(load_vae_checkpoint)
    if "scenario" in sig.parameters:
        if scenario is None:
            raise SystemExit(
                "The installed checkpoint_io.load_vae_checkpoint still requires a "
                "scenario argument, but none was given or found in a companion JSON. "
                "Pass --vae-scenario {nominal,tuned} explicitly."
            )
        return load_vae_checkpoint(checkpoint, tokenizer, scenario, device)
    return load_vae_checkpoint(checkpoint, tokenizer, device)


def mol_legend(mol) -> str:
    mw = Descriptors.MolWt(mol)
    qed = QED.qed(mol)
    sa = sascorer.calculateScore(mol)
    return f"MW={mw:.1f}  QED={qed:.2f}  SA={sa:.2f}"


def generate_valid_molecules(
    sample_fn, n_needed: int, max_attempts: int, batch_size: int
) -> tuple[list, list[str], int, int]:
    """Call sample_fn(n) -> list[str] repeatedly, oversampling until n_needed
    valid, parseable molecules are collected (or max_attempts total samples
    drawn). Returns (mols, smiles, n_total_sampled, n_valid_seen).

    n_valid_seen counts EVERY valid molecule encountered while scanning,
    not just the (possibly smaller, capped-at-n_needed) list returned for
    display -- sample_fn(batch_size) already generates the whole batch in
    one call, so scanning past the n_needed-th valid molecule costs nothing
    and gives a less noisy validity estimate than stopping the count the
    instant enough are found to fill the grid.
    """
    mols = []
    smiles_kept = []
    n_total = 0
    n_valid_seen = 0
    while len(mols) < n_needed and n_total < max_attempts:
        batch = sample_fn(batch_size)
        n_total += len(batch)
        for smi in batch:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                n_valid_seen += 1
                if len(mols) < n_needed:
                    mols.append(mol)
                    smiles_kept.append(smi)
    return mols, smiles_kept, n_total, n_valid_seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--generator-checkpoint", type=Path, default=None)
    parser.add_argument("--vae-checkpoint", type=Path, default=None)
    parser.add_argument("--vae-vocab", type=Path, default=None)
    parser.add_argument("--vae-scenario", choices=["nominal", "tuned"], default=None)
    parser.add_argument("--n-molecules", type=int, default=12)
    parser.add_argument("--molecules-per-row", type=int, default=4)
    parser.add_argument("--sub-image-size", type=int, default=300, help="pixels, square")
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--decode", choices=["stochastic", "greedy"], default="stochastic")
    parser.add_argument("--temp", type=float, default=1.0, help="--decode stochastic only")
    parser.add_argument("--batch-size", type=int, default=64, help="oversampling batch size")
    parser.add_argument("--max-attempts", type=int, default=5000, help="cap on total raw samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.generator_checkpoint is None and args.vae_checkpoint is None:
        print("Give at least one of --generator-checkpoint or --vae-checkpoint.", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    vae_checkpoint = args.vae_checkpoint
    vae_scenario = args.vae_scenario
    generator = None
    latent_dim = None
    title = ""

    if args.generator_checkpoint is not None:
        gen_state = torch.load(args.generator_checkpoint, map_location=device)
        latent_dim = infer_generator_latent_dim(gen_state)

        if vae_checkpoint is None:
            json_path = Path(str(args.generator_checkpoint).replace(".generator.pt", ".json"))
            if json_path.exists():
                run_info = json.loads(json_path.read_text())
                vae_checkpoint = Path(run_info["vae_checkpoint"])
                vae_scenario = vae_scenario or run_info.get("vae_scenario")
                print(f"Auto-detected paired VAE from {json_path}: {vae_checkpoint}")
            else:
                print(
                    f"No companion JSON found at {json_path} and --vae-checkpoint "
                    "not given -- can't determine which VAE decodes this generator's "
                    "output. Pass --vae-checkpoint explicitly.",
                    file=sys.stderr,
                )
                return 1

        generator = Generator(latent_dim).to(device)
        generator.load_state_dict(gen_state)
        generator.eval()
        print(f"Loaded generator from {args.generator_checkpoint} (latent_dim={latent_dim})")
        title = f"GAN samples: {args.generator_checkpoint.name}"
    else:
        title = f"VAE prior samples: {vae_checkpoint.name}"

    print(f"\n{title}")
    vae = load_vae_flexibly(vae_checkpoint, args.vae_vocab, vae_scenario, device)

    def sample_fn(n: int) -> list[str]:
        if generator is not None:
            with torch.no_grad():
                noise = torch.rand(n, latent_dim, device=device) * 2 - 1
                z = generator(noise)
        else:
            z = None  # VAE.sample draws from its own prior when z is None
        greedy = args.decode == "greedy"
        return vae.sample(n, max_len=args.max_len, z=z, temp=args.temp, greedy=greedy)

    print(
        f"Sampling until {args.n_molecules} valid molecules are found (max {args.max_attempts})..."
    )
    mols, smiles_kept, n_total, n_valid_seen = generate_valid_molecules(
        sample_fn, args.n_molecules, args.max_attempts, args.batch_size
    )

    if len(mols) < args.n_molecules:
        print(
            f"WARNING: only found {len(mols)}/{args.n_molecules} valid molecules "
            f"after {n_total} samples (observed validity ~{n_valid_seen / max(n_total, 1):.1%}). "
            "Showing what was found; raise --max-attempts for more.",
            file=sys.stderr,
        )
    else:
        print(
            f"Found {len(mols)} valid molecules to display; {n_valid_seen}/{n_total} "
            f"raw samples were valid overall (~{n_valid_seen / n_total:.1%} -- a quick "
            "estimate from this run's sample size, not a substitute for train_gan.py's "
            "own eps_v, which is computed over 30,000 samples)"
        )

    legends = [f"#{i + 1}  {mol_legend(mol)}" for i, mol in enumerate(mols)]

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=args.molecules_per_row,
        subImgSize=(args.sub_image_size, args.sub_image_size),
        legends=legends,
        returnPNG=False,
    )

    if args.output is None:
        stem = (args.generator_checkpoint or vae_checkpoint).stem
        args.output = Path("results") / f"{stem}_samples.{args.format}"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "pdf":
        img.save(args.output, "PDF")
    else:
        img.save(args.output)
    print(f"Wrote {args.output}")

    smiles_path = args.output.with_suffix(".smiles.txt")
    smiles_path.write_text("\n".join(smiles_kept))
    print(f"Wrote {smiles_path} (the SMILES behind each numbered cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())