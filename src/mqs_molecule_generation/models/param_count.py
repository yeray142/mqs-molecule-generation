"""Exact VAE parameter-count formula (paper §2.2.1, Table 2 gate).

Pure Python, no torch dependency -- lets the Table 2 gate
(4,271,250 nominal / 6,472,082 tuned) be checked without needing torch
importable, and gives an independent cross-check against the real
``torch.nn.Module`` in ``vae.py`` (see ``test_vae.py``, which verifies the
two agree exactly when torch is available).

Reproduces PyTorch's own parameter-counting conventions for ``nn.Embedding``,
``nn.GRU``, and ``nn.Linear`` exactly (verified against the PyTorch
documentation, not assumed): a GRU layer has 3 gates (reset, update, new)
each with an input-hidden weight, a hidden-hidden weight, and two biases.

Solved for by exhaustive search over vocab size V and latent dim Z (see
module docstring history in vae.py) -- V=37, Z=10 is the unique combination
in a broad search (V in [10,150), Z in [2,200), bidirectional in {T,F} for
each of nominal/tuned independently) that reproduces BOTH Table 2 targets
simultaneously with the same V, which is strong evidence it's correct rather
than a coincidental fit. Notably, this REVISES an earlier assumption that
the encoder was bidirectional for both scenarios -- the unique fit has
``q_bidir=False`` (MOSES's own default) for both nominal and tuned.
"""

from __future__ import annotations


def gru_param_count(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool) -> int:
    """Exact parameter count for torch.nn.GRU.

    Per layer, per direction: weight_ih (3H x I), weight_hh (3H x H),
    bias_ih (3H,), bias_hh (3H,) -- the "3" is the GRU's three gates
    (reset, update, new). Layers after the first take the previous layer's
    output as input, which has size hidden_size * num_directions.
    """
    num_directions = 2 if bidirectional else 1
    total = 0
    for layer in range(num_layers):
        layer_input_size = input_size if layer == 0 else hidden_size * num_directions
        per_direction = (
            3 * hidden_size * layer_input_size  # weight_ih
            + 3 * hidden_size * hidden_size  # weight_hh
            + 3 * hidden_size  # bias_ih
            + 3 * hidden_size  # bias_hh
        )
        total += num_directions * per_direction
    return total


def linear_param_count(in_features: int, out_features: int) -> int:
    """Exact parameter count for torch.nn.Linear (weight + bias)."""
    return in_features * out_features + out_features


def embedding_param_count(num_embeddings: int, embedding_dim: int) -> int:
    """Exact parameter count for torch.nn.Embedding.

    padding_idx does not reduce the count -- it only zeros that row's
    gradient during training, the row is still allocated and counted.
    """
    return num_embeddings * embedding_dim


def vae_param_count(
    vocab_size: int,
    q_d_h: int,
    q_n_layers: int,
    q_bidir: bool,
    d_z: int,
    d_d_h: int,
    d_n_layers: int,
) -> int:
    """Total VAE parameter count, matching MOSES's exact architecture (moses/vae/model.py).

    Architecture (paper §2.2.1, ported from MOSES's CharVAE):
        embedding:    nn.Embedding(V, E) where E = V  (OneHotVocab: d_emb ==
                      vocab_size -- the embedding is initialised as an
                      identity matrix, torch.eye(V), but remains a fully
                      trainable V x V matrix, not a fixed small embedding)
        encoder_rnn:  nn.GRU(E, q_d_h, num_layers=q_n_layers, bidirectional=q_bidir)
        q_mu:         nn.Linear(q_d_h * (2 if q_bidir else 1), d_z)
        q_logvar:     nn.Linear(q_d_h * (2 if q_bidir else 1), d_z)
        decoder_rnn:  nn.GRU(E + d_z, d_d_h, num_layers=d_n_layers, bidirectional=False)
        decoder_lat:  nn.Linear(d_z, d_d_h)
        decoder_fc:   nn.Linear(d_d_h, V)

    Args:
        vocab_size: Tokenizer vocabulary size (V). Also the embedding
            dimension, per OneHotVocab.
        q_d_h: Encoder GRU hidden size.
        q_n_layers: Encoder GRU layer count.
        q_bidir: Whether the encoder GRU is bidirectional.
        d_z: Latent dimension.
        d_d_h: Decoder GRU hidden size.
        d_n_layers: Decoder GRU layer count.

    Returns:
        Total trainable parameter count.
    """
    embedding_dim = vocab_size  # OneHotVocab: d_emb == vocab_size

    embedding = embedding_param_count(vocab_size, embedding_dim)

    encoder_gru = gru_param_count(embedding_dim, q_d_h, q_n_layers, q_bidir)
    q_d_last = q_d_h * (2 if q_bidir else 1)
    q_mu = linear_param_count(q_d_last, d_z)
    q_logvar = linear_param_count(q_d_last, d_z)

    decoder_gru = gru_param_count(embedding_dim + d_z, d_d_h, d_n_layers, bidirectional=False)
    decoder_lat = linear_param_count(d_z, d_d_h)
    decoder_fc = linear_param_count(d_d_h, vocab_size)

    return embedding + encoder_gru + q_mu + q_logvar + decoder_gru + decoder_lat + decoder_fc