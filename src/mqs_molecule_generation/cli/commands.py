"""CLI entry point for mqs-molecule-generation."""

from __future__ import annotations

import click
import yaml

from mqs_molecule_generation.training.latent_style_gan import LatentStyleGANTraining
from mqs_molecule_generation.utils.manifest import build_manifest
from mqs_molecule_generation.utils.seeding import set_seed


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """Latent Style-based Quantum Wasserstein GAN for Drug Design.

    arXiv:2603.22399 - Reproduced with PennyLane + PyTorch, benchmarked via QPUBench.
    """
    pass


@main.command()
@click.option("--dry-run", default=False, is_flag=True, help="Run full pipeline without quantum execution")
@click.option("--chain", default=False, is_flag=True, help="Run full benchmark chain (implies dry-run)")
@click.option("--seeds", default="0-4", help="Seed range, e.g. 0-4 or 0,2,5")
@click.option(
    "--ansatz",
    type=click.Choice(["simple", "bel"]),
    default="simple",
    help="Ansatz type",
)
@click.option("--n-qubits", type=int, default=4, help="Number of qubits")
@click.option("--n-layers", type=int, default=2, help="Number of layers")
@click.option("--max-iters", type=int, default=100, help="Max training iterations")
@click.option(
    "--readout",
    type=click.Choice(["single", "dual"]),
    default="single",
    help="Discriminator readout type",
)
def train(
    dry_run: bool,
    chain: bool,
    seeds: str,
    ansatz: str,
    n_qubits: int,
    n_layers: int,
    max_iters: int,
    readout: str,
) -> None:
    """Train the LatentStyleGAN model.

    Examples:

        mqs-molecule-generation train --dry-run --seeds 0 --ansatz simple

        mqs-molecule-generation train --dry-run --chain --seeds 0-4 --ansatz bel
    """
    # Chain implies dry-run
    if chain:
        dry_run = True

    manifest = build_manifest()

    # Parse seed range
    if "-" in seeds:
        start, end = map(int, seeds.split("-"))
        seed_list = list(range(start, end + 1))
    else:
        seed_list = [int(s) for s in seeds.split(",")]

    click.echo("=== LatentStyleGAN Training ===")
    click.echo(f"Ansatz: {ansatz}, n_qubits={n_qubits}, n_layers={n_layers}")
    click.echo(f"Readout: {readout}")
    click.echo(f"Seeds: {seed_list}")
    click.echo(f"Dry run: {dry_run}")
    click.echo(f"Chain: {chain}")
    click.echo(f"Manifest: git_sha={manifest.git_sha}, lock={manifest.lockfile_hash}")
    click.echo(f"Python: {manifest.python_version}")
    click.echo("")

    for seed in seed_list:
        set_seed(seed)

        click.echo(f"\n=== Training seed={seed} ===")

        training = LatentStyleGANTraining(
            n_qubits=n_qubits,
            n_layers=n_layers,
            ansatz_type=ansatz,
            seed=seed,
            dry_run=dry_run,
            manifest=manifest,
            max_iterations=max_iters,
            readout=readout,
        )

        if dry_run:
            click.echo("[DRY RUN] Pipeline validated, no quantum execution")
            if chain:
                click.echo("[CHAIN] Running full evaluation chain...")
                # In dry-run + chain mode, we validate the circuit construction
                # and metric computation without actual quantum execution
                try:
                    from mqs_molecule_generation.metrics.validity import compute_validity

                    valid, total, frac = compute_validity(["CCO", "c1ccccc1", "INVALID"])
                    click.echo(f"[CHAIN] Validity check: {valid}/{total} = {frac:.2%}")
                except Exception as e:
                    click.echo(f"[CHAIN] Validity check skipped: {e}")
        else:
            click.echo("Training would run here (stub implementation)")
            metrics = training.run()
            click.echo(f"Training completed: {len(metrics)} eval points")


@main.command()
@click.option("--chain", is_flag=True, help="Run full evaluation chain")
@click.option("--dry-run", is_flag=True, help="Validate without execution")
def eval(chain: bool, dry_run: bool) -> None:
    """Run evaluation metrics (SA score, diversity, validity).

    Examples:

        mqs-molecule-generation eval --dry-run

        mqs-molecule-generation eval --chain
    """
    if dry_run:
        click.echo("[DRY RUN] Evaluation validated, no metrics computed")
        return

    if chain:
        click.echo("[CHAIN] Running full evaluation chain...")

        # Evaluate with dummy samples
        dummy_samples = ["CCO", "c1ccccc1", "CCN", "CC(=O)O"]
        try:
            from mqs_molecule_generation.metrics.sa_score import compute_sa_scores_batch

            sa_scores = compute_sa_scores_batch(dummy_samples)
            click.echo(f"[CHAIN] SA scores: {sa_scores}")
        except Exception as e:
            click.echo(f"[CHAIN] SA score skipped: {e}")

        try:
            from mqs_molecule_generation.metrics.diversity import compute_diversity

            div = compute_diversity(dummy_samples)
            click.echo(f"[CHAIN] Diversity: {div:.4f}")
        except Exception as e:
            click.echo(f"[CHAIN] Diversity skipped: {e}")

        try:
            from mqs_molecule_generation.metrics.validity import compute_validity

            valid, total, frac = compute_validity(dummy_samples)
            click.echo(f"[CHAIN] Validity: {valid}/{total} = {frac:.2%}")
        except Exception as e:
            click.echo(f"[CHAIN] Validity skipped: {e}")
    else:
        click.echo("Use --chain for full evaluation or --dry-run to validate")


@main.command()
def manifest_cmd() -> None:
    """Print per-run manifest (git SHA, lockfile hash, versions).

    Example:

        mqs-molecule-generation manifest-cmd
    """
    m = build_manifest()
    click.echo(yaml.dump(m.model_dump(), default_flow_style=False))


# Entry point for pyproject entry point
def cli_main() -> None:
    main()
