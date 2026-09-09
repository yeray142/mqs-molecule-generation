#!/usr/bin/env python
"""Encode molecule(s) through a trained VAE and check the reconstruction.

Single molecule:
    uv run python scripts/reconstruct.py \\
        --checkpoint results/vae_runs/n_ep_100_seed_0.pt \\
        --smiles "CC(=O)Oc1ccccc1C(=O)O"

Batch, against real training/test molecules (the decisive check -- see below):
    uv run python scripts/reconstruct.py \\
        --checkpoint results/vae_runs/n_ep_100_seed_0.pt \\
        --from-train 50

Decoder-only sensitivity probe (isolates the decoder from the encoder --
see below):
    uv run python scripts/reconstruct.py \\
        --checkpoint results/vae_runs/n_ep_100_seed_0.pt \\
        --probe-latent

Encodes to the deterministic latent point (mu, the mean of q(z|x) -- no
sampling), decodes it back both greedily (the single "most likely"
reconstruction) and, in single-molecule mode, stochastically too (temp=1,
to see how much the decode varies from a fixed latent point). Reports exact
string match, canonical molecule match, and Tanimoto similarity for a
graceful "how close" measure when it doesn't match exactly.

A SINGLE molecule's mu vector having only one or two dimensions far from
zero is suggestive of posterior collapse (the encoder only using a fraction
of its latent dimensions) but isn't proof -- that molecule alone might just
be poorly represented, e.g. if it's out-of-distribution for what the model
was trained on. The decisive version of that check needs MANY different
molecules: a dimension that stays near zero across every one of them,
regardless of what's being encoded, is a dead/collapsed dimension; one that
varies meaningfully from molecule to molecule is doing real work. --from-train
and --from-valid run that check directly, encoding real molecules from your
frozen splits and reporting each latent dimension's mean and standard
deviation ACROSS the batch, not just for one input.

--from-train/--from-valid also recompute kl_loss/recon_loss on the sampled
batch using the exact same forward pass training used (model.forward), and,
if train_vae.py wrote a companion `<checkpoint_stem>.history.csv` next to
the checkpoint, print the last logged training-epoch's values alongside it.
Close agreement is a direct, independent check that the checkpoint loaded
correctly -- not inferred from reconstruction quality, but from re-running
training's own loss computation on real data after loading.

If reconstructions collapse to the same output regardless of input, that
alone doesn't distinguish "the encoder never produces an informative z" from
"the decoder ignores z entirely, even when it IS informative" -- both look
identical from the encoder side. --probe-latent isolates the decoder: it
feeds hand-built z vectors, well outside any range a (possibly collapsed)
encoder would actually produce, and checks whether the decoder's greedy
output changes at all. If it doesn't respond even to those, the issue is
worth checking on the decoder side directly, not just written off as
posterior collapse.

Vocabulary handling: a checkpoint's embedding weights only line up with the
right characters if loaded with the EXACT tokenizer used at training time.
This script looks for a `<checkpoint_stem>.vocab.json` saved alongside the
checkpoint (written automatically by scripts/train_vae.py runs from now on)
and uses that if present. For an older checkpoint saved before that existed,
it falls back to rebuilding the tokenizer from data/processed/train.smi --
which is only correct if that file is unchanged since training, and prints a
warning rather than silently assuming so.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from mqs_molecule_generation.data.moses import load_splits
from mqs_molecule_generation.data.tokenizer import SmilesTokenizer
from mqs_molecule_generation.models.vae import VAE, VAEConfig

RDLogger.DisableLog("rdApp.*")

#: A latent dimension whose mean-posterior std across a batch of DIFFERENT
#: molecules falls below this is flagged as likely collapsed/dead. A
#: standard informal diagnostic in the VAE literature, not a validated
#: statistical test -- treat the count as a signal, not a hard verdict.
ACTIVE_DIM_STD_THRESHOLD = 0.1


def load_tokenizer(
    checkpoint: Path, vocab_path: Path | None, processed_dir: Path
) -> SmilesTokenizer:
    if vocab_path is not None:
        print(f"Loading tokenizer vocabulary from {vocab_path}")
        return SmilesTokenizer.load(vocab_path)

    auto_vocab_path = checkpoint.with_suffix("").with_suffix(".vocab.json")
    if auto_vocab_path.exists():
        print(f"Loading tokenizer vocabulary from {auto_vocab_path}")
        return SmilesTokenizer.load(auto_vocab_path)

    train_path = processed_dir / "train.smi"
    print(
        f"WARNING: no vocabulary file found next to {checkpoint} "
        f"(expected {auto_vocab_path}). Falling back to rebuilding the "
        f"tokenizer from {train_path}. This is ONLY correct if that file "
        "is byte-identical to what was used when this checkpoint was "
        "trained -- if it has been regenerated since (a different seed, a "
        "different filter set), the embedding indices will silently no "
        "longer match, and results below will be meaningless without any "
        "error being raised.",
        file=sys.stderr,
    )
    return SmilesTokenizer.from_file(train_path)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    """Morgan/ECFP4 Tanimoto similarity between two SMILES, or None if either is invalid."""
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp_a, fp_b = gen.GetFingerprint(mol_a), gen.GetFingerprint(mol_b)
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def describe_reconstruction(label: str, original_canonical: str, reconstructed: str) -> None:
    exact_string_match = reconstructed == original_canonical
    mol = Chem.MolFromSmiles(reconstructed)
    valid = mol is not None
    canonical_match = valid and Chem.MolToSmiles(mol) == original_canonical
    similarity = tanimoto_similarity(original_canonical, reconstructed)

    print(f"  {label}: {reconstructed!r}")
    print(f"    valid RDKit molecule:        {valid}")
    print(f"    exact string match:          {exact_string_match}")
    print(f"    canonical molecule match:    {canonical_match}")
    print(
        f"    Tanimoto similarity:         {similarity:.3f}"
        if similarity is not None
        else "    Tanimoto similarity:         n/a (invalid)"
    )


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[VAE, SmilesTokenizer]:
    tokenizer = load_tokenizer(args.checkpoint, args.vocab, args.processed_dir)
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    config = (
        VAEConfig.nominal(d_z=args.latent_dim)
        if args.scenario == "nominal"
        else VAEConfig.tuned(d_z=args.latent_dim)
    )
    model = VAE(tokenizer, config).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)  # raises loudly on any shape mismatch
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint} ({args.scenario}, d_z={args.latent_dim})")
    return model, tokenizer


def run_single(args: argparse.Namespace, model: VAE, tokenizer: SmilesTokenizer, device) -> int:
    original_mol = Chem.MolFromSmiles(args.smiles)
    if original_mol is None:
        print(f"'{args.smiles}' is not a valid SMILES string.", file=sys.stderr)
        return 1
    original_canonical = Chem.MolToSmiles(original_mol)
    if original_canonical != args.smiles:
        print(f"Input canonicalised: {args.smiles!r} -> {original_canonical!r}")

    x = [
        torch.tensor(
            tokenizer.encode(original_canonical, add_bos=True, add_eos=True),
            dtype=torch.long,
            device=device,
        )
    ]
    mu = model.encode_mu(x)
    mu_vals = mu.squeeze(0).tolist()
    n_active = sum(1 for v in mu_vals if abs(v) > ACTIVE_DIM_STD_THRESHOLD)
    print(f"\nEncoded latent mu (d_z={mu.shape[-1]}): {mu_vals}")
    print(
        f"  {n_active}/{len(mu_vals)} dimensions exceed |mu|>{ACTIVE_DIM_STD_THRESHOLD} for "
        "this molecule alone -- suggestive of posterior collapse if low, but "
        "NOT decisive from one molecule; use --from-train/--from-valid for "
        "the real test (mu variance ACROSS many different molecules)."
    )

    print(f"\nOriginal (canonical): {original_canonical!r}\n")

    print("Deterministic reconstruction (mu, greedy decode):")
    greedy_recon = model.sample(n_batch=1, max_len=args.max_len, z=mu, greedy=True)[0]
    describe_reconstruction("greedy", original_canonical, greedy_recon)

    print(
        f"\n{args.n_stochastic_samples} stochastic reconstructions "
        f"(mu, temp={args.temp}, multinomial decode):"
    )
    z_repeated = mu.repeat(args.n_stochastic_samples, 1)
    stochastic_recons = model.sample(
        n_batch=args.n_stochastic_samples,
        max_len=args.max_len,
        z=z_repeated,
        temp=args.temp,
        greedy=False,
    )
    n_exact = int(greedy_recon == original_canonical)
    for i, recon in enumerate(stochastic_recons):
        describe_reconstruction(f"sample {i + 1}", original_canonical, recon)
        n_exact += int(recon == original_canonical)

    print(
        f"\n{n_exact}/{1 + args.n_stochastic_samples} reconstructions "
        "exactly matched the canonical original."
    )
    return 0


def load_last_training_loss(checkpoint: Path) -> dict[str, float] | None:
    """Load the last logged 'train'-mode row from a companion history CSV.

    scripts/train_vae.py writes `<checkpoint_stem>.history.csv` (one row per
    epoch, both train and eval) alongside the checkpoint. Returns None if no
    such file exists (older checkpoints, or --checkpoint-in runs that skipped
    training) -- the loss comparison in run_batch degrades gracefully to
    "not available" rather than failing.
    """
    history_path = checkpoint.with_suffix("").with_suffix(".history.csv")
    if not history_path.exists():
        return None

    last_train_row: dict[str, str] | None = None
    with history_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["mode"] == "train":
                last_train_row = row

    if last_train_row is None:
        return None
    return {
        "epoch": int(last_train_row["epoch"]),
        "kl_loss": float(last_train_row["kl_loss"]),
        "recon_loss": float(last_train_row["recon_loss"]),
        "kl_weight": float(last_train_row["kl_weight"]),
        "loss": float(last_train_row["loss"]),
    }


def run_batch(args: argparse.Namespace, model: VAE, tokenizer: SmilesTokenizer, device) -> int:
    split_name = "train" if args.from_train else "valid"
    n_requested = args.from_train or args.from_valid

    train_smiles, valid_smiles = load_splits(args.processed_dir)
    pool = train_smiles if split_name == "train" else valid_smiles

    rng = np.random.default_rng(args.seed)
    n = min(n_requested, len(pool))
    idx = rng.choice(len(pool), size=n, replace=False)
    sample_smiles = [pool[int(i)] for i in idx]
    print(f"\nSampled {n} molecules from {split_name} set (seed={args.seed})")

    x = [
        torch.tensor(
            tokenizer.encode(s, add_bos=True, add_eos=True), dtype=torch.long, device=device
        )
        for s in sample_smiles
    ]

    # Same forward pass training used (model.forward -> kl_loss, recon_loss),
    # here in eval mode with no gradient tracking. If the checkpoint loaded
    # correctly, this should land close to what was logged at the end of
    # training -- a direct check using the actual training loss computation,
    # not just downstream reconstruction quality.
    with torch.no_grad():
        eval_kl_loss, eval_recon_loss = model(x)
    eval_kl_loss = eval_kl_loss.item()
    eval_recon_loss = eval_recon_loss.item()

    print(f"\nLoss on these {n} {split_name}-set molecules (loaded checkpoint, eval mode):")
    print(f"  kl_loss:    {eval_kl_loss:.4f}")
    print(f"  recon_loss: {eval_recon_loss:.4f}")

    last_train = load_last_training_loss(args.checkpoint)
    if last_train is None:
        print(
            "  (no companion .history.csv found next to this checkpoint -- "
            "can't compare against the logged training loss; only checkpoints "
            "from a scripts/train_vae.py run that included training, not "
            "--checkpoint-in-only runs, have this file)"
        )
    else:
        kl_diff = eval_kl_loss - last_train["kl_loss"]
        recon_diff = eval_recon_loss - last_train["recon_loss"]
        print(
            f"\n  Last logged training epoch ({last_train['epoch']}, "
            f"kl_weight={last_train['kl_weight']:.4f}):"
        )
        print(f"    kl_loss:    {last_train['kl_loss']:.4f}  (this batch: {kl_diff:+.4f})")
        print(f"    recon_loss: {last_train['recon_loss']:.4f}  (this batch: {recon_diff:+.4f})")
        print(
            "\n  Close agreement here (this sample is a random subset of the "
            "full training set the epoch-level numbers were averaged over, so "
            "expect close, not bit-identical) is strong, direct evidence the "
            "checkpoint loaded correctly: it's the exact same forward-pass "
            "computation training used, re-run post-hoc on real data."
        )

    mu = model.encode_mu(x)  # (n, d_z)
    greedy_recons = model.sample(n_batch=n, max_len=args.max_len, z=mu, greedy=True)

    # The decisive posterior-collapse check: per-dimension mean/std ACROSS
    # this batch of different molecules, not within one molecule's mu.
    mu_np = mu.detach().cpu().numpy()
    dim_means = mu_np.mean(axis=0)
    dim_stds = mu_np.std(axis=0)
    n_active_dims = int((dim_stds > ACTIVE_DIM_STD_THRESHOLD).sum())

    print(f"\nPer-dimension latent statistics across {n} molecules:")
    print(f"{'dim':>4}  {'mean':>10}  {'std':>10}  {'active?':>8}")
    for i, (m, s) in enumerate(zip(dim_means, dim_stds, strict=True)):
        active = s > ACTIVE_DIM_STD_THRESHOLD
        print(f"{i:>4}  {m:>10.4f}  {s:>10.4f}  {'yes' if active else 'no':>8}")
    print(
        f"\n{n_active_dims}/{mu_np.shape[1]} dimensions show std > "
        f"{ACTIVE_DIM_STD_THRESHOLD} across different molecules. "
        f"A low count here (not just for one molecule) is the real posterior-"
        "collapse signature -- most of the latent space isn't encoding "
        "molecule-specific information regardless of what's being encoded."
    )

    n_valid = n_exact_string = n_exact_canonical = 0
    similarities: list[float] = []
    for original, recon in zip(sample_smiles, greedy_recons, strict=True):
        mol = Chem.MolFromSmiles(recon)
        if mol is not None:
            n_valid += 1
            canonical_recon = Chem.MolToSmiles(mol)
            if canonical_recon == original:
                n_exact_canonical += 1
        if recon == original:
            n_exact_string += 1
        sim = tanimoto_similarity(original, recon)
        if sim is not None:
            similarities.append(sim)

    print(f"\nGreedy reconstruction quality over {n} {split_name}-set molecules:")
    print(f"  valid RDKit molecule:      {n_valid}/{n}  ({n_valid / n:.1%})")
    print(f"  exact string match:       {n_exact_string}/{n}  ({n_exact_string / n:.1%})")
    print(f"  exact canonical match:    {n_exact_canonical}/{n}  ({n_exact_canonical / n:.1%})")
    if similarities:
        print(
            f"  mean Tanimoto similarity: {np.mean(similarities):.3f}  "
            f"(over {len(similarities)} valid reconstructions)"
        )

    print(f"\n{args.n_examples} example reconstructions:")
    for original, recon in list(zip(sample_smiles, greedy_recons, strict=True))[: args.n_examples]:
        sim = tanimoto_similarity(original, recon)
        sim_str = f"{sim:.3f}" if sim is not None else "n/a"
        print(f"  original: {original!r}")
        print(f"  greedy:   {recon!r}  (Tanimoto={sim_str})")
    return 0


def _probe_one_model(model: VAE, label: str, args: argparse.Namespace, device) -> dict[str, object]:
    """Run the weight-norm + sampling probe on a single model. Returns summary stats."""
    d_z = args.latent_dim
    print(f"\n=== {label} ===")

    weight = model.decoder_lat.weight.detach().cpu()  # (d_d_h, d_z)
    col_norms = weight.norm(dim=0)  # (d_z,) -- one norm per input (z) dimension
    order = torch.argsort(col_norms, descending=True)
    print("decoder_lat input-weight column norms (higher = more learned influence on z_0):")
    for dim in order.tolist():
        print(f"  dim {dim}: {col_norms[dim].item():.4f}")

    probes: list[tuple[str, torch.Tensor]] = [("zero baseline", torch.zeros(1, d_z, device=device))]
    for dim in range(d_z):
        for sign, sign_label in ((1.0, "+"), (-1.0, "-")):
            z = torch.zeros(1, d_z, device=device)
            z[0, dim] = sign * args.probe_magnitude
            probes.append((f"dim {dim} {sign_label}{args.probe_magnitude:g}", z))

    results: list[tuple[str, str, bool]] = []
    for probe_label, z in probes:
        decoded = model.sample(n_batch=1, max_len=args.max_len, z=z, greedy=True)[0]
        valid = Chem.MolFromSmiles(decoded) is not None
        results.append((probe_label, decoded, valid))
        print(f"  {probe_label:<16}  valid={valid!s:<5}  {decoded!r}")

    n_valid = sum(1 for _, _, v in results if v)
    distinct = len({smi for _, smi, _ in results})
    norm_ratio = (col_norms.max() / col_norms.min()).item()
    print(
        f"\n{label} summary: {distinct}/{len(results)} distinct outputs, "
        f"{n_valid}/{len(results)} valid, "
        f"weight-norm max/min ratio={norm_ratio:.2f}"
    )
    return {
        "label": label,
        "col_norms": col_norms,
        "n_valid": n_valid,
        "n_total": len(results),
        "distinct": distinct,
        "norm_ratio": norm_ratio,
    }


def run_probe(args: argparse.Namespace, model: VAE, tokenizer: SmilesTokenizer, device) -> int:
    """Decode from deliberately large, hand-constructed z vectors, greedily.

    This isolates the DECODER's sensitivity to z from the ENCODER's ability
    to produce informative z's in the first place. If a real molecule's mu
    vector is collapsed (most dimensions near zero), decoding from those
    mu's alone can't tell you whether the decoder is capable of responding
    to z at all, or whether it's ignoring z entirely regardless of magnitude
    -- the encoded z's never leave a narrow range, so that range is all
    you've tested. Here z is pushed to +/- `args.probe_magnitude` one
    dimension at a time (all others held at 0), well outside any range the
    encoder is likely to have produced, to see whether the decoder can
    respond to a strong signal even if the encoder never gave it a real one.

    Also prints decoder_lat's per-dimension input weight-column norms before
    any decoding happens -- a direct, architectural read on which z
    dimensions the decoder actually learned to respond to, independent of
    interpreting decoded SMILES at all.

    With --compare-random, the exact same probe also runs on a second,
    freshly-initialised model of identical architecture (same tokenizer, same
    config, no checkpoint loaded) so the trained checkpoint's numbers can be
    checked directly against what pure random initialisation looks like,
    rather than trusting an interpretation of what "looks trained".
    """
    checkpoint_result = _probe_one_model(model, f"CHECKPOINT ({args.checkpoint})", args, device)

    if not args.compare_random:
        distinct = checkpoint_result["distinct"]
        if distinct == 1:
            print(
                "\nThe decoder returns the SAME output even for large, deliberately "
                "different z's -- it is not just that the encoder never produces "
                "an informative z, the decoder itself appears insensitive to z "
                "across this entire probed range. Worth checking decoder_lat and "
                "the z concatenation in forward_decoder/sample directly."
            )
        else:
            print(
                "\nThe decoder responds to large z perturbations along some "
                "dimensions but not others. Re-run with --compare-random to "
                "check these numbers directly against a freshly-initialised, "
                "untrained model of identical architecture instead of relying "
                "on an interpretation of what counts as a meaningful difference."
            )
        return 0

    torch.manual_seed(args.seed + 1)  # different draw from the checkpoint's own init
    fresh_config = (
        VAEConfig.nominal(d_z=args.latent_dim)
        if args.scenario == "nominal"
        else VAEConfig.tuned(d_z=args.latent_dim)
    )
    fresh_model = VAE(tokenizer, fresh_config).to(device)
    fresh_model.eval()
    random_result = _probe_one_model(fresh_model, "FRESH, UNTRAINED (random init)", args, device)

    print("\n=== checkpoint vs. fresh random init ===")
    ck_distinct = f"{checkpoint_result['distinct']:>9}/{checkpoint_result['n_total']}"
    rand_distinct = f"{random_result['distinct']:>9}/{random_result['n_total']}"
    ck_valid = f"{checkpoint_result['n_valid']:>9}/{checkpoint_result['n_total']}"
    rand_valid = f"{random_result['n_valid']:>9}/{random_result['n_total']}"

    print(f"{'':30}  {'checkpoint':>12}  {'random init':>12}")
    print(f"{'distinct outputs':30}  {ck_distinct}  {rand_distinct}")
    print(f"{'valid SMILES':30}  {ck_valid}  {rand_valid}")
    print(
        f"{'weight-norm max/min ratio':30}  {checkpoint_result['norm_ratio']:>12.2f}"
        f"  {random_result['norm_ratio']:>12.2f}"
    )
    print(
        "\nIf the checkpoint's valid-SMILES count is far above the random "
        "model's (random weights essentially never produce parseable SMILES "
        "under greedy decoding -- there's no mechanism by which they would), "
        "that's direct evidence the checkpoint's weights ARE the trained ones, "
        "not an artifact of loading. The weight-norm ratio comparison shows "
        "whether the checkpoint's per-dimension spread is distinguishable "
        "from pure initialisation noise or not."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="path to a .pt checkpoint")
    parser.add_argument(
        "--smiles", default=None, help="single molecule to reconstruct (default: aspirin)"
    )
    parser.add_argument(
        "--from-train",
        type=int,
        default=None,
        metavar="N",
        help="batch mode: N random train-set molecules",
    )
    parser.add_argument(
        "--from-valid",
        type=int,
        default=None,
        metavar="N",
        help="batch mode: N random valid-set molecules",
    )
    parser.add_argument(
        "--probe-latent",
        action="store_true",
        help=(
            "decoder-only sensitivity probe: decode from large, hand-built z "
            "vectors (one dimension pushed to +/-magnitude at a time) instead "
            "of encoding any real molecule -- ignores --smiles/--from-train/"
            "--from-valid. See run_probe's docstring for what this isolates."
        ),
    )
    parser.add_argument("--probe-magnitude", type=float, default=3.0, help="--probe-latent only")
    parser.add_argument(
        "--compare-random",
        action="store_true",
        help=(
            "--probe-latent only: also run the identical probe on a freshly-"
            "initialised, untrained model of the same architecture, printed "
            "side by side with the checkpoint's results"
        ),
    )
    parser.add_argument("--n-examples", type=int, default=10, help="batch mode: examples to print")
    parser.add_argument(
        "--vocab",
        type=Path,
        default=None,
        help="explicit tokenizer vocab.json (overrides auto-detection)",
    )
    parser.add_argument(
        "--scenario",
        choices=["nominal", "tuned"],
        default="nominal",
        help="must match how the checkpoint was trained",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=10, help="must match how the checkpoint was trained"
    )
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--n-stochastic-samples", type=int, default=5, help="single-molecule mode only"
    )
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--temp", type=float, default=1.0, help="single-molecule mode only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    if args.from_train and args.from_valid:
        print("Use --from-train or --from-valid, not both.", file=sys.stderr)
        return 1
    if (args.from_train or args.from_valid) and args.smiles:
        print("--smiles is ignored in batch mode (--from-train/--from-valid).", file=sys.stderr)

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model, tokenizer = load_model(args, device)

    if args.probe_latent:
        return run_probe(args, model, tokenizer, device)

    if args.from_train or args.from_valid:
        return run_batch(args, model, tokenizer, device)

    args.smiles = args.smiles or "CC(=O)Oc1ccccc1C(=O)O"
    return run_single(args, model, tokenizer, device)


if __name__ == "__main__":
    raise SystemExit(main())