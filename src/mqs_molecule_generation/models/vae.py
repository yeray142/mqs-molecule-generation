"""GRU-based character VAE (paper §2.2.1), ported from MOSES's model.py.

Architecture, forward pass, and sampling are ported directly from
``moses/vae/model.py`` -- confirmed correct against the paper's own gate
(Table 2: 4,271,250 params nominal / 6,472,082 tuned) via
``models.param_count.vae_param_count``, not assumed from a generic VAE
design. See that module's docstring for how the exact hyperparameters
(including a correction to an earlier assumption that the encoder was
bidirectional -- it isn't, in either scenario) were derived.

Two details carried over from MOSES that are easy to get wrong by building
"a" GRU-VAE instead of porting this one:

- The embedding dimension equals the vocabulary size (``OneHotVocab``:
  ``d_emb = V``), initialised as an identity matrix, not a small fixed
  embedding size. It remains a fully trainable V x V matrix afterward
  unless ``freeze_embeddings=True``.
- The decoder is conditioned on z at every timestep (concatenated with the
  token embedding at each step, per MOSES's ``forward_decoder``), not just
  through the initial hidden state. The initial hidden state is instead a
  learned projection of z (``decoder_lat``), repeated across all decoder
  layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 (universal PyTorch convention)
from torch import nn

from mqs_molecule_generation.data.tokenizer import SmilesTokenizer

#: Clamp bounds for encoder logvar, applied in _encode_stats. NOT part of
#: MOSES's original code -- a deliberate numerical-stability addition.
#:
#: The KL term is 0.5*(exp(logvar) + mu^2 - 1 - logvar). With kl_weight
#: staying small for most of training (kl_w_end=0.05, reached only near the
#: final epoch), there is very little gradient pressure keeping logvar
#: bounded. Reconstruction loss actively rewards driving logvar very
#: negative (a tighter posterior makes z closer to a deterministic function
#: of x, which is "cheating" toward better reconstruction) -- but as
#: logvar -> -inf, the -logvar term in the KL formula diverges too, and nothing
#: is stopping it. Observed directly during tuned-scenario training (paper
#: hyperparameters: q_d_h=512, lr starting at 1e-3, over 3x nominal's 3e-4):
#: kl_loss climbing smoothly for ~90 epochs, then a runaway feedback loop
#: driving logvar increasingly negative, culminating in kl_loss jumping from
#: ~18 to ~5587 in a single epoch with no recovery over the following ~100
#: epochs. Clamping keeps exp(logvar) and -logvar bounded regardless of what
#: the optimizer tries, without touching the paper's own stated
#: hyperparameters (lr, kl_w_end) at all. Bounds are generous relative to
#: healthy training (observed kl_loss in the 5-33 range implies per-
#: dimension logvar values far inside this window) -- this only engages once
#: something has already gone numerically wrong.
LOGVAR_MIN = -20.0
LOGVAR_MAX = 2.0


@dataclass(frozen=True)
class VAEConfig:
    """VAE hyperparameters (paper §2.2.1 / §2.4.1, Table 2).

    Defaults match MOSES's own config defaults (moses/vae/config.py) for
    every field the paper's "nominal" scenario doesn't explicitly override.
    """

    q_d_h: int = 256  # encoder GRU hidden size (nominal: 256, tuned: 512)
    q_n_layers: int = 1  # encoder GRU layers (nominal and tuned: 1)
    q_bidir: bool = False  # encoder bidirectional (nominal and tuned: False --
    #                        MOSES's own default; NOT overridden by the paper,
    #                        despite an earlier planning note to the contrary --
    #                        see param_count.py's docstring for the derivation)
    q_dropout: float = 0.5  # only applied by torch if q_n_layers > 1 (it isn't here)

    d_z: int = 10  # latent dimension. Paper studies 10/20/30 (matches n_qb for
    #                single readout or 2*n_qb for dual); 10 is the default scenario.
    d_d_h: int = 512  # decoder GRU hidden size (nominal and tuned: 512)
    d_n_layers: int = 3  # decoder GRU layers (nominal: 3, tuned: 4)
    d_dropout: float = 0.0  # nominal: 0.0, tuned: 0.75

    freeze_embeddings: bool = False

    @classmethod
    def nominal(cls, d_z: int = 10) -> VAEConfig:
        """Paper's "nominal" VAE scenario (Table 2)."""
        return cls(
            q_d_h=256,
            q_n_layers=1,
            q_bidir=False,
            d_z=d_z,
            d_d_h=512,
            d_n_layers=3,
            d_dropout=0.0,
        )

    @classmethod
    def tuned(cls, d_z: int = 10) -> VAEConfig:
        """Paper's "tuned" VAE scenario (Table 2)."""
        return cls(
            q_d_h=512,
            q_n_layers=1,
            q_bidir=False,
            d_z=d_z,
            d_d_h=512,
            d_n_layers=4,
            d_dropout=0.75,
        )


class VAE(nn.Module):
    """GRU character VAE. Architecture ported from moses/vae/model.py."""

    def __init__(self, tokenizer: SmilesTokenizer, config: VAEConfig) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config

        self.pad = tokenizer.pad_idx
        self.bos = tokenizer.bos_idx
        self.eos = tokenizer.eos_idx

        vocab_size = tokenizer.vocab_size
        embedding_dim = vocab_size  # OneHotVocab: d_emb == vocab_size

        # Embedding, one-hot initialised (torch.eye), fully trainable unless frozen.
        self.x_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.pad)
        self.x_emb.weight.data.copy_(torch.eye(vocab_size))
        if config.freeze_embeddings:
            self.x_emb.weight.requires_grad = False

        # Encoder
        self.encoder_rnn = nn.GRU(
            embedding_dim,
            config.q_d_h,
            num_layers=config.q_n_layers,
            batch_first=True,
            dropout=config.q_dropout if config.q_n_layers > 1 else 0,
            bidirectional=config.q_bidir,
        )
        q_d_last = config.q_d_h * (2 if config.q_bidir else 1)
        self.q_mu = nn.Linear(q_d_last, config.d_z)
        self.q_logvar = nn.Linear(q_d_last, config.d_z)

        # Decoder
        self.decoder_rnn = nn.GRU(
            embedding_dim + config.d_z,
            config.d_d_h,
            num_layers=config.d_n_layers,
            batch_first=True,
            dropout=config.d_dropout if config.d_n_layers > 1 else 0,
        )
        self.decoder_lat = nn.Linear(config.d_z, config.d_d_h)
        self.decoder_fc = nn.Linear(config.d_d_h, vocab_size)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Full VAE forward step: encode, then decode.

        Args:
            x: List of 1-D long tensors, one per SMILES in the batch (already
                token-encoded, WITHOUT bos/eos -- forward_decoder adds them
                via teacher forcing against the raw ids, matching MOSES).

        Returns:
            (kl_loss, recon_loss), both scalars.
        """
        z, kl_loss = self.forward_encoder(x)
        recon_loss = self.forward_decoder(x, z)
        return kl_loss, recon_loss

    def _encode_stats(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Shared encoder forward pass, up to (mu, logvar).

        Used by both forward_encoder (adds sampling + KL) and encode_mu
        (deterministic point estimate).
        """
        embedded = [self.x_emb(i_x) for i_x in x]
        packed = nn.utils.rnn.pack_sequence(embedded, enforce_sorted=False)

        _, h = self.encoder_rnn(packed, None)

        h = h[-(1 + int(self.encoder_rnn.bidirectional)) :]
        h = torch.cat(h.split(1), dim=-1).squeeze(0)

        logvar = torch.clamp(self.q_logvar(h), min=LOGVAR_MIN, max=LOGVAR_MAX)
        return self.q_mu(h), logvar

    def forward_encoder(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode x -> z ~ q(z|x), plus the KL term against N(0, I).

        Args:
            x: List of 1-D long tensors (token ids), one per SMILES.

        Returns:
            (z, kl_loss): z has shape (batch, d_z); kl_loss is a scalar.
        """
        mu, logvar = self._encode_stats(x)
        eps = torch.randn_like(mu)
        z = mu + (logvar / 2).exp() * eps

        kl_loss = 0.5 * (logvar.exp() + mu**2 - 1 - logvar).sum(1).mean()
        return z, kl_loss

    def encode_mu(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Deterministic latent point for x: the MEAN of q(z|x), no sampling.

        Useful for reconstruction-fidelity checks and any other case where a
        single, repeatable latent point is wanted rather than a stochastic
        draw from the approximate posterior (which forward_encoder returns).

        Args:
            x: List of 1-D long tensors (token ids), one per SMILES.

        Returns:
            mu, shape (batch, d_z).
        """
        mu, _ = self._encode_stats(x)
        return mu

    def forward_decoder(self, x: list[torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Decode z (conditioned on x for teacher forcing) -> reconstruction loss.

        Args:
            x: List of 1-D long tensors (token ids, no bos/eos -- this method
                adds nothing; the caller's tokenizer should already include
                them if teacher forcing across the full sequence is desired,
                matching MOSES's own convention of pre-encoding with
                add_bos=True, add_eos=True before calling forward()).
            z: (batch, d_z) latent vectors, from forward_encoder or the prior.

        Returns:
            Scalar cross-entropy reconstruction loss (pad-masked).
        """
        lengths = [len(i_x) for i_x in x]
        x_padded = nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=self.pad)
        x_emb = self.x_emb(x_padded)

        z_0 = z.unsqueeze(1).repeat(1, x_emb.size(1), 1)
        decoder_input = torch.cat([x_emb, z_0], dim=-1)
        packed_input = nn.utils.rnn.pack_padded_sequence(
            decoder_input, lengths, batch_first=True, enforce_sorted=False
        )

        h_0 = self.decoder_lat(z)
        h_0 = h_0.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)

        output, _ = self.decoder_rnn(packed_input, h_0)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        logits = self.decoder_fc(output)

        recon_loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            x_padded[:, 1:].contiguous().view(-1),
            ignore_index=self.pad,
        )
        return recon_loss

    def sample_z_prior(self, n_batch: int) -> torch.Tensor:
        """Sample z ~ N(0, I), shape (n_batch, d_z)."""
        return torch.randn(n_batch, self.q_mu.out_features, device=self.x_emb.weight.device)

    def sample(
        self,
        n_batch: int,
        max_len: int = 100,
        z: torch.Tensor | None = None,
        temp: float = 1.0,
        greedy: bool = False,
    ) -> list[str]:
        """Autoregressively decode `n_batch` SMILES strings, ported from MOSES's VAE.sample.

        Args:
            n_batch: Number of SMILES to generate.
            max_len: Maximum sequence length before truncating.
            z: (n_batch, d_z) latent vectors, or None to sample from the prior.
            temp: Softmax temperature (lower = more greedy). Ignored if greedy=True.
            greedy: If True, take argmax at every step instead of sampling from
                the softmax (torch.multinomial) -- deterministic given z, useful
                for reconstruction-fidelity checks where "most likely decode"
                matters more than exploring the output distribution.

        Returns:
            Generated SMILES strings (BOS/EOS stripped, per tokenizer.decode's default).
        """
        with torch.no_grad():
            if z is None:
                z = self.sample_z_prior(n_batch)
            z = z.to(self.device)
            z_0 = z.unsqueeze(1)

            h = self.decoder_lat(z)
            h = h.unsqueeze(0).repeat(self.decoder_rnn.num_layers, 1, 1)
            w = torch.tensor(self.bos, device=self.device).repeat(n_batch)
            x = torch.tensor([self.pad], device=self.device).repeat(n_batch, max_len)
            x[:, 0] = self.bos
            end_pads = torch.tensor([max_len], device=self.device).repeat(n_batch)
            eos_mask = torch.zeros(n_batch, dtype=torch.bool, device=self.device)

            for i in range(1, max_len):
                x_emb = self.x_emb(w).unsqueeze(1)
                decoder_input = torch.cat([x_emb, z_0], dim=-1)

                o, h = self.decoder_rnn(decoder_input, h)
                logits = self.decoder_fc(o.squeeze(1))

                if greedy:
                    w = torch.argmax(logits, dim=-1)
                else:
                    probs = F.softmax(logits / temp, dim=-1)
                    w = torch.multinomial(probs, 1)[:, 0]

                x[~eos_mask, i] = w[~eos_mask]
                newly_finished = ~eos_mask & (w == self.eos)
                end_pads[newly_finished] = i + 1
                eos_mask = eos_mask | newly_finished

            return [self.tokenizer.decode(x[i, : end_pads[i]].tolist()) for i in range(x.size(0))]


def infer_latent_dim(state_dict: dict[str, torch.Tensor]) -> int:
    """Read d_z off a checkpoint's own weights rather than trust a CLI flag.

    q_mu.weight has shape (d_z, q_d_h * num_directions) regardless of
    scenario (nominal/tuned differ in q_d_h, not d_z) or which latent
    dimension the run used (10/20/30) -- its first dimension IS d_z,
    unambiguously, for any valid checkpoint. Avoids a class of silent
    mismatch: passing the wrong --latent-dim wouldn't necessarily crash
    immediately (a too-small d_z is still a valid, just wrong, shape for
    q_mu/q_logvar's construction), whereas reading it directly can't be
    wrong for a checkpoint that was ever successfully saved by this
    project's own VAE.
    """
    return state_dict["q_mu.weight"].shape[0]