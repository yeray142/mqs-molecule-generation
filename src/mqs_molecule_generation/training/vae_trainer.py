"""VAE training loop (paper §2.4.1), ported from moses/vae/trainer.py + misc.py.

Loss is ``kl_weight(epoch) * kl_loss + recon_loss`` -- NOT a plain sum of the
two terms. ``kl_weight`` ramps linearly over training (``KLAnnealer``, in
``trainer_config.py``) and the learning rate follows a cosine schedule with
optional warm restarts (``CosineAnnealingLRWithRestart``, below), both
ported from MOSES's own ``moses/vae/misc.py`` rather than assumed.

Two non-obvious things carried over faithfully rather than "cleaned up",
because matching the paper's actual training procedure matters more than
aesthetic preference:

- The paper's "nominal" (constant lr=3e-4) and "tuned" (cosine 1e-3 to 1e-5)
  scenarios are the SAME scheduler class with different endpoints -- nominal
  emerges automatically when ``lr_start == lr_end`` (the cosine curve between
  two identical values is a flat line), not from a separate code path.
- ``CosineAnnealingLRWithRestart`` has an off-by-one: because
  ``torch.optim.lr_scheduler.LRScheduler.__init__`` calls ``step()`` once
  internally, the LR used on the very first training epoch is already
  slightly decayed from ``lr_start`` (computed at ``current_epoch=1``, not
  0), and a schedule with no restarts reaches ``lr_end`` EXACTLY on its last
  epoch. Verified by direct simulation before writing any code, not assumed
  -- see ``test_vae_trainer.py::TestCosineAnnealingLRWithRestart``.

This module needs torch importable just to be imported (subclassing
LRScheduler). KLAnnealer/TrainerConfig/EpochMetrics, which don't need torch
at all, live in trainer_config.py instead, so they stay usable without torch
installed.

``fit()`` reports every epoch's loss/kl_loss/recon_loss/kl_weight/lr two
ways: live, in the tqdm progress bar's postfix (via ``pbar.set_postfix``,
which updates in place rather than scrolling); and, every ``log_every``
epochs (default: every epoch), as a permanent printed line via
``tqdm.write`` (not plain ``print``, which would otherwise visually break
the progress bar). Pass ``csv_path`` to also persist every epoch's full
metrics as CSV rows, written incrementally as training proceeds -- so a
convergence curve is still recoverable even if a long run is interrupted,
and so it can be plotted afterward rather than only read off scrolled
terminal output.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from mqs_molecule_generation.data.tokenizer import SmilesTokenizer
from mqs_molecule_generation.models.vae import VAE
from mqs_molecule_generation.training.trainer_config import EpochMetrics, KLAnnealer, TrainerConfig

__all__ = [
    "CosineAnnealingLRWithRestart",
    "EpochMetrics",
    "KLAnnealer",
    "TrainerConfig",
    "VAETrainer",
    "build_collate_fn",
]


class CosineAnnealingLRWithRestart(LRScheduler):
    """Cosine LR decay with optional SGDR-style warm restarts (moses/vae/misc.py).

    lr_start == lr_end gives a constant learning rate (paper's "nominal"
    scenario); lr_start != lr_end gives the "tuned" cosine schedule. Call
    ``.step()`` once per completed training epoch (matching MOSES's trainer,
    which calls it at the END of each epoch, not the start).
    """

    def __init__(
        self, optimizer: optim.Optimizer, n_period: int, n_mult: int, lr_end: float
    ) -> None:
        self.n_period = n_period
        self.n_mult = n_mult
        self.lr_end = lr_end
        self.current_epoch = 0
        self.t_end = n_period
        super().__init__(optimizer, -1)

    def get_lr(self) -> list[float]:
        return [
            self.lr_end
            + (base_lr - self.lr_end)
            * (1 + math.cos(math.pi * self.current_epoch / self.t_end))
            / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: int | None = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.current_epoch += 1

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
            param_group["lr"] = lr

        if self.current_epoch == self.t_end:
            self.current_epoch = 0
            self.t_end = self.n_mult * self.t_end


def build_collate_fn(tokenizer: SmilesTokenizer, device: torch.device):
    """Build a DataLoader collate_fn that tokenizes SMILES with BOS/EOS.

    Unlike MOSES's own collate_fn, this deliberately does NOT sort the batch
    by length: the VAE's forward_encoder/forward_decoder (models/vae.py) use
    ``enforce_sorted=False`` in pack_sequence/pack_padded_sequence
    specifically so unsorted batches work correctly, making MOSES's manual
    sort-by-length step unnecessary here.
    """

    def collate(smiles_batch: list[str]) -> list[torch.Tensor]:
        return [
            torch.tensor(
                tokenizer.encode(s, add_bos=True, add_eos=True), dtype=torch.long, device=device
            )
            for s in smiles_batch
        ]

    return collate


class VAETrainer:
    """Trains a VAE (models.vae.VAE) following MOSES's training procedure exactly."""

    def __init__(self, config: TrainerConfig) -> None:
        self.config = config

    def _optim_params(self, model: VAE):
        return (p for p in model.parameters() if p.requires_grad)

    def _run_epoch(
        self,
        model: VAE,
        epoch: int,
        loader: DataLoader,
        kl_weight: float,
        optimizer: optim.Optimizer | None,
    ) -> EpochMetrics:
        """One pass over `loader`. Training if `optimizer` is given, else eval."""
        model.train(optimizer is not None)

        kl_sum = recon_sum = loss_sum = 0.0
        n_batches = 0
        lr = 0.0

        for batch in loader:
            kl_loss, recon_loss = model(batch)
            loss = kl_weight * kl_loss + recon_loss

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self._optim_params(model), self.config.clip_grad)
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]

            kl_sum += kl_loss.item()
            recon_sum += recon_loss.item()
            loss_sum += loss.item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return EpochMetrics(
            epoch=epoch,
            mode="train" if optimizer is not None else "eval",
            kl_weight=kl_weight,
            lr=lr,
            kl_loss=kl_sum / n_batches,
            recon_loss=recon_sum / n_batches,
            loss=loss_sum / n_batches,
        )

    def fit(
        self,
        model: VAE,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        checkpoint_fn=None,
        checkpoint_every: int = 0,
        log_every: int = 1,
        csv_path: Path | None = None,
    ) -> list[EpochMetrics]:
        """Train for config.n_epoch epochs. Returns per-epoch metrics history.

        Args:
            model: The VAE to train.
            train_loader: Training batches.
            val_loader: Optional validation batches, evaluated once per epoch
                (no gradient step) using that epoch's kl_weight.
            checkpoint_fn: Optional ``(model, epoch) -> None`` called every
                ``checkpoint_every`` epochs.
            checkpoint_every: See above; 0 disables checkpointing.
            log_every: Print a full metrics line via ``tqdm.write`` every
                this many epochs (default: every epoch). Set higher than
                ``config.n_epoch`` to disable printed lines entirely and
                keep only the live progress-bar postfix. The live postfix
                itself always updates every epoch regardless of this value.
            csv_path: If given, write every epoch's metrics (both train and,
                if present, eval rows) as CSV, flushed after every row so
                partial results survive an interrupted run.

        Returns:
            One EpochMetrics per epoch (two per epoch if val_loader is given:
            train then eval).
        """
        n_epoch = self.config.n_epoch
        optimizer = optim.Adam(self._optim_params(model), lr=self.config.lr_start)
        kl_annealer = KLAnnealer(
            n_epoch, self.config.kl_start, self.config.kl_w_start, self.config.kl_w_end
        )
        lr_annealer = CosineAnnealingLRWithRestart(
            optimizer, self.config.lr_n_period, self.config.lr_n_mult, self.config.lr_end
        )

        csv_file = None
        csv_writer = None
        if csv_path is not None:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=["epoch", "mode", "kl_weight", "lr", "kl_loss", "recon_loss", "loss"],
            )
            csv_writer.writeheader()

        history: list[EpochMetrics] = []
        try:
            model.zero_grad()
            pbar = tqdm(range(n_epoch))
            for epoch in pbar:
                kl_weight = kl_annealer(epoch)
                train_metrics = self._run_epoch(model, epoch, train_loader, kl_weight, optimizer)
                history.append(train_metrics)

                val_metrics = None
                if val_loader is not None:
                    val_metrics = self._run_epoch(model, epoch, val_loader, kl_weight, None)
                    history.append(val_metrics)

                postfix = {
                    "loss": f"{train_metrics.loss:.4f}",
                    "kl": f"{train_metrics.kl_loss:.4f}",
                    "recon": f"{train_metrics.recon_loss:.4f}",
                    "kl_w": f"{kl_weight:.4f}",
                    "lr": f"{train_metrics.lr:.2e}",
                }
                if val_metrics is not None:
                    postfix["val_loss"] = f"{val_metrics.loss:.4f}"
                pbar.set_postfix(postfix)

                if log_every > 0 and epoch % log_every == 0:
                    line = (
                        f"epoch {epoch:>5d}/{n_epoch}  loss={train_metrics.loss:.4f}  "
                        f"kl_loss={train_metrics.kl_loss:.4f}  "
                        f"recon_loss={train_metrics.recon_loss:.4f}  "
                        f"kl_weight={kl_weight:.4f}  lr={train_metrics.lr:.2e}"
                    )
                    if val_metrics is not None:
                        line += (
                            f"  | val_loss={val_metrics.loss:.4f}  "
                            f"val_kl={val_metrics.kl_loss:.4f}  "
                            f"val_recon={val_metrics.recon_loss:.4f}"
                        )
                    tqdm.write(line)

                if csv_writer is not None:
                    rows = (train_metrics, *((val_metrics,) if val_metrics is not None else ()))
                    for m in rows:
                        csv_writer.writerow(
                            {
                                "epoch": m.epoch,
                                "mode": m.mode,
                                "kl_weight": m.kl_weight,
                                "lr": m.lr,
                                "kl_loss": m.kl_loss,
                                "recon_loss": m.recon_loss,
                                "loss": m.loss,
                            }
                        )
                    csv_file.flush()

                if (
                    checkpoint_fn is not None
                    and checkpoint_every > 0
                    and epoch % checkpoint_every == 0
                ):
                    checkpoint_fn(model, epoch)

                lr_annealer.step()
        finally:
            if csv_file is not None:
                csv_file.close()

        return history

        return history