"""Shared VAE-checkpoint-loading logic for scripts that need a trained VAE.

Factored out once three separate scripts (reconstruct.py, train_gan.py,
eval_vae_reference.py) needed the same "find the vocabulary, build the right
config, load the checkpoint" sequence. Kept in the installed package (not a
scripts/-local helper) so it's importable and testable the normal way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from mqs_molecule_generation.data.tokenizer import SmilesTokenizer
from mqs_molecule_generation.models.vae import VAE, VAEConfig, infer_latent_dim


def load_tokenizer_for_checkpoint(
    checkpoint: Path, vocab_path: Path | None, processed_dir: Path
) -> SmilesTokenizer:
    """Load the exact vocabulary a checkpoint was trained with.

    A checkpoint's embedding weights only line up with the right characters
    if loaded with the SAME char_to_idx mapping used at training time. Looks
    for a `<checkpoint_stem>.vocab.json` saved alongside the checkpoint
    (written automatically by scripts/train_vae.py). For an older checkpoint
    saved before that existed, falls back to rebuilding the tokenizer from
    `data/processed/train.smi` -- only correct if that file is unchanged
    since training -- and prints a warning rather than silently assuming so.
    """
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
        "trained.",
        file=sys.stderr,
    )
    return SmilesTokenizer.from_file(train_path)


def load_vae_checkpoint(
    checkpoint: Path,
    tokenizer: SmilesTokenizer,
    scenario: str,
    device: torch.device,
) -> VAE:
    """Build a VAE matching `checkpoint`'s own architecture and load it.

    `d_z` is read directly off the checkpoint (models.vae.infer_latent_dim)
    rather than taken on trust from a CLI flag -- q_mu.weight's shape is
    unambiguous for any checkpoint this project ever saved, so there's no
    reason to let a mistyped --latent-dim silently load the wrong-shaped
    model. `scenario` still has to be supplied: nominal and tuned differ in
    q_d_h/d_n_layers, which aren't recoverable from d_z alone, and a mismatch
    there fails loudly (state_dict shape mismatch) rather than silently.
    """
    state_dict = torch.load(checkpoint, map_location=device)
    d_z = infer_latent_dim(state_dict)

    config = VAEConfig.nominal(d_z=d_z) if scenario == "nominal" else VAEConfig.tuned(d_z=d_z)
    model = VAE(tokenizer, config).to(device)
    model.load_state_dict(state_dict)  # raises loudly on any shape mismatch
    model.eval()
    print(
        f"Loaded VAE checkpoint from {checkpoint} "
        f"(scenario={scenario}, d_z={d_z}, vocab_size={tokenizer.vocab_size})"
    )
    return model