"""Tests for the VAE architecture (paper §2.2.1, Table 2 gate)."""

from __future__ import annotations

import pytest

from mqs_molecule_generation.models.param_count import (
    embedding_param_count,
    gru_param_count,
    linear_param_count,
    vae_param_count,
)

# Solved exactly (unique fit) against Table 2 -- see param_count.py's docstring.
SOLVED_VOCAB_SIZE = 37
SOLVED_LATENT_DIM = 10
NOMINAL_TARGET = 4_271_250
TUNED_TARGET = 6_472_082


class TestGruParamCount:
    """Exact arithmetic against PyTorch's own documented GRU parameterisation."""

    def test_single_layer_unidirectional(self) -> None:
        # weight_ih (3*hidden x in_size) + weight_hh (3*hidden x hidden)
        # + bias_ih (3*hidden) + bias_hh (3*hidden)
        in_size, hidden = 10, 20
        expected = 3 * hidden * in_size + 3 * hidden * hidden + 3 * hidden + 3 * hidden
        assert gru_param_count(in_size, hidden, num_layers=1, bidirectional=False) == expected

    def test_bidirectional_doubles_single_layer(self) -> None:
        in_size, hidden = 10, 20
        uni = gru_param_count(in_size, hidden, num_layers=1, bidirectional=False)
        bi = gru_param_count(in_size, hidden, num_layers=1, bidirectional=True)
        assert bi == 2 * uni

    def test_second_layer_input_size_matches_first_layer_output(self) -> None:
        # For layer >= 1, input size is hidden_size * num_directions, not the
        # original in_size -- this test would fail if that were wrong.
        in_size, hidden = 10, 20
        one_layer = gru_param_count(in_size, hidden, num_layers=1, bidirectional=False)
        two_layer = gru_param_count(in_size, hidden, num_layers=2, bidirectional=False)
        # second layer's input is `hidden` (previous layer's output), not in_size
        second_layer_params = 3 * hidden * hidden + 3 * hidden * hidden + 3 * hidden + 3 * hidden
        assert two_layer == one_layer + second_layer_params

    def test_bidirectional_second_layer_input_is_doubled(self) -> None:
        in_size, hidden = 10, 20
        second_layer_input = hidden * 2  # bidirectional layer 0 output is 2*hidden
        one_layer_bi = gru_param_count(in_size, hidden, num_layers=1, bidirectional=True)
        two_layer_bi = gru_param_count(in_size, hidden, num_layers=2, bidirectional=True)
        second_layer_params_per_dir = (
            3 * hidden * second_layer_input + 3 * hidden * hidden + 6 * hidden
        )
        assert two_layer_bi == one_layer_bi + 2 * second_layer_params_per_dir


class TestLinearAndEmbeddingParamCount:
    def test_linear_includes_bias(self) -> None:
        assert linear_param_count(in_features=10, out_features=5) == 10 * 5 + 5

    def test_embedding_ignores_padding_idx_in_count(self) -> None:
        # padding_idx zeros a row's gradient; it doesn't remove it from the
        # parameter tensor. Count should just be num_embeddings * dim.
        assert embedding_param_count(num_embeddings=37, embedding_dim=37) == 37 * 37


class TestAgainstTable2:
    """The actual gate: exact reproduction of Table 2's two published totals."""

    def test_nominal_exact(self) -> None:
        total = vae_param_count(
            vocab_size=SOLVED_VOCAB_SIZE,
            q_d_h=256,
            q_n_layers=1,
            q_bidir=False,
            d_z=SOLVED_LATENT_DIM,
            d_d_h=512,
            d_n_layers=3,
        )
        assert total == NOMINAL_TARGET

    def test_tuned_exact(self) -> None:
        total = vae_param_count(
            vocab_size=SOLVED_VOCAB_SIZE,
            q_d_h=512,
            q_n_layers=1,
            q_bidir=False,
            d_z=SOLVED_LATENT_DIM,
            d_d_h=512,
            d_n_layers=4,
        )
        assert total == TUNED_TARGET

    def test_fit_is_unique_in_a_broad_search(self) -> None:
        # Re-derive the same conclusion the docstring claims: searching V in
        # [10,150), Z in [2,80), bidir in {T,F} independently for nominal and
        # tuned, exactly one combination reproduces both targets with a
        # shared V. A narrower range than the original search (for test
        # speed) but wide enough that a coincidental second fit nearby would
        # still show up.
        matches = []
        for z in range(2, 80):
            for v in range(10, 150):
                for q_bidir_nom in (True, False):
                    nominal = vae_param_count(v, 256, 1, q_bidir_nom, z, 512, 3)
                    if nominal != NOMINAL_TARGET:
                        continue
                    for q_bidir_tun in (True, False):
                        tuned = vae_param_count(v, 512, 1, q_bidir_tun, z, 512, 4)
                        if tuned == TUNED_TARGET:
                            matches.append((v, z, q_bidir_nom, q_bidir_tun))
        assert matches == [(SOLVED_VOCAB_SIZE, SOLVED_LATENT_DIM, False, False)]

    def test_bidirectional_encoder_does_not_match(self) -> None:
        # Documents the corrected assumption: q_bidir=True does NOT reproduce
        # the nominal target at the solved V, Z -- guards against silently
        # reintroducing the earlier (wrong) assumption.
        total = vae_param_count(SOLVED_VOCAB_SIZE, 256, 1, True, SOLVED_LATENT_DIM, 512, 3)
        assert total != NOMINAL_TARGET


class TestVAEModule:
    """Real torch.nn.Module verification -- requires torch, skipped if unavailable.

    Run these explicitly wherever torch is installed (it's already a project
    dependency) to confirm the actual nn.Module matches the formula above,
    not just that the formula matches the paper.
    """

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    @pytest.fixture
    def tokenizer(self):
        from mqs_molecule_generation.data.tokenizer import SmilesTokenizer

        # A tiny synthetic training set; vocab size here won't match the
        # paper's solved V=37 (that depends on the specific 12k-molecule
        # dataset), but the point of these tests is architectural
        # correctness for WHATEVER vocab size a real tokenizer produces.
        return SmilesTokenizer.from_data(["CCO", "c1ccccc1", "CC(=O)O", "CCN", "[13CH4]"])

    def test_param_count_matches_formula_nominal(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        config = VAEConfig.nominal(d_z=10)
        model = VAE(tokenizer, config)
        actual = sum(p.numel() for p in model.parameters())
        expected = vae_param_count(
            tokenizer.vocab_size,
            config.q_d_h,
            config.q_n_layers,
            config.q_bidir,
            config.d_z,
            config.d_d_h,
            config.d_n_layers,
        )
        assert actual == expected

    def test_param_count_matches_formula_tuned(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        config = VAEConfig.tuned(d_z=10)
        model = VAE(tokenizer, config)
        actual = sum(p.numel() for p in model.parameters())
        expected = vae_param_count(
            tokenizer.vocab_size,
            config.q_d_h,
            config.q_n_layers,
            config.q_bidir,
            config.d_z,
            config.d_d_h,
            config.d_n_layers,
        )
        assert actual == expected

    def test_embedding_initialised_as_identity(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        expected = torch.eye(tokenizer.vocab_size)
        assert torch.allclose(model.x_emb.weight.data, expected)

    def test_forward_pass_produces_finite_scalar_losses(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        x = [
            torch.tensor(tokenizer.encode(s, add_bos=True, add_eos=True), dtype=torch.long)
            for s in smiles
        ]
        kl_loss, recon_loss = model(x)
        assert kl_loss.dim() == 0
        assert recon_loss.dim() == 0
        assert torch.isfinite(kl_loss)
        assert torch.isfinite(recon_loss)

    def test_sample_z_prior_shape(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        z = model.sample_z_prior(n_batch=5)
        assert z.shape == (5, 10)

    def test_sample_produces_strings(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        model.eval()
        samples = model.sample(n_batch=3, max_len=20)
        assert len(samples) == 3
        assert all(isinstance(s, str) for s in samples)

    def test_encode_mu_is_deterministic(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        model.eval()
        x = [torch.tensor(tokenizer.encode("CCO", add_bos=True, add_eos=True), dtype=torch.long)]
        mu1 = model.encode_mu(x)
        mu2 = model.encode_mu(x)
        assert torch.equal(mu1, mu2)  # no sampling involved, must be bit-identical

    def test_encode_mu_differs_from_stochastic_z(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        model.eval()
        x = [torch.tensor(tokenizer.encode("CCO", add_bos=True, add_eos=True), dtype=torch.long)]
        mu = model.encode_mu(x)
        z, _ = model.forward_encoder(x)
        # Not a strict guarantee in general (eps could be ~0), but with a
        # freshly-initialised model logvar won't be so negative that this
        # flakes in practice.
        assert not torch.equal(mu, z)

    def test_greedy_sample_is_deterministic_given_z(self, torch, tokenizer) -> None:
        from mqs_molecule_generation.models.vae import VAE, VAEConfig

        model = VAE(tokenizer, VAEConfig.nominal(d_z=10))
        model.eval()
        z = model.sample_z_prior(n_batch=1)
        out1 = model.sample(n_batch=1, max_len=20, z=z, greedy=True)
        out2 = model.sample(n_batch=1, max_len=20, z=z, greedy=True)
        assert out1 == out2  # argmax decoding, no randomness once z is fixed