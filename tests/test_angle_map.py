"""Eq. 4 angle map (gate 11). numpy only -- no torch needed."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml
import pytest

from mqs_molecule_generation.circuits.angle_map import ANGLE_MODES, TWO_PI, eq4_angles
from mqs_molecule_generation.circuits.ansatz import AnsatzParams

TOL = 1e-6  # float32 rounds 2*pi up by ~1.7e-7; that is rounding, not a range violation


def _extreme_inputs(n_qb: int, n_angles: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-3, 6, size=(64, 1))  # spans tiny to 1e6 -> tanh saturates
    xi = rng.normal(size=(64, n_qb)) * scale
    w = rng.normal(size=n_angles) * 10.0 ** rng.uniform(-2, 3)
    b = rng.normal(size=n_angles) * 10.0 ** rng.uniform(-2, 3)
    return xi, w, b


class TestMapping:
    def test_each_angle_is_fed_by_its_own_qubits_noise(self) -> None:
        ap = AnsatzParams(5, 2, "bel")
        owner = np.array(ap.noise_qubit_per_angle)
        xi = np.arange(1.0, 6.0)[None, :]  # xi_q = q+1, all distinct
        theta = eq4_angles(xi, np.ones(ap.n_angles), np.zeros(ap.n_angles), owner)
        np.testing.assert_allclose(theta[0], TWO_PI * np.tanh(owner + 1.0))

    def test_literal_matches_equation_4(self) -> None:
        xi = np.array([[0.3, -1.2]])
        w = np.array([0.5, 2.0, -1.0])
        b = np.array([0.1, -0.3, 0.0])
        owner = np.array([0, 1, 1])
        expected = TWO_PI * np.tanh(xi[0, owner] * w + b)
        np.testing.assert_allclose(eq4_angles(xi, w, b, owner)[0], expected)

    def test_output_shape(self) -> None:
        ap = AnsatzParams(5, 2, "bel")
        xi = np.zeros((7, 5))
        out = eq4_angles(xi, np.zeros(55), np.zeros(55), np.array(ap.noise_qubit_per_angle))
        assert out.shape == (7, 55)

    def test_invalid_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="mode must be"):
            eq4_angles(np.zeros((1, 2)), np.zeros(2), np.zeros(2), np.array([0, 1]), mode="nope")


class TestRangeGate:
    """Gate: theta in [0, 2*pi] for all inputs."""

    @pytest.mark.parametrize("mode", ["shifted", "wrapped"])
    @pytest.mark.parametrize("kind", ["simple", "bel"])
    def test_in_domain_modes_stay_inside_zero_to_two_pi(self, mode, kind) -> None:
        ap = AnsatzParams(5, 2, kind)
        owner = np.array(ap.noise_qubit_per_angle)
        for seed in range(5):
            xi, w, b = _extreme_inputs(5, ap.n_angles, seed)
            theta = eq4_angles(xi, w, b, owner, mode=mode)
            assert theta.min() >= -TOL, mode
            assert theta.max() <= TWO_PI + TOL, mode

    def test_literal_eq4_as_printed_is_not_inside_zero_to_two_pi(self) -> None:
        # The recorded finding: the paper says [0, 2pi], but 2*pi*tanh(.) is odd.
        ap = AnsatzParams(5, 2, "bel")
        owner = np.array(ap.noise_qubit_per_angle)
        xi, w, b = _extreme_inputs(5, ap.n_angles)
        theta = eq4_angles(xi, w, b, owner, mode="literal")
        assert theta.min() < -1.0, "expected negative angles from literal Eq. 4"
        assert theta.min() > -TWO_PI - TOL and theta.max() < TWO_PI + TOL  # (-2pi, 2pi)

    def test_literal_is_in_domain_only_when_tanh_argument_is_nonnegative(self) -> None:
        owner = np.array([0])
        pos = eq4_angles(np.array([[2.0]]), np.array([1.0]), np.array([0.0]), owner)
        neg = eq4_angles(np.array([[-2.0]]), np.array([1.0]), np.array([0.0]), owner)
        assert pos[0, 0] > 0 > neg[0, 0]


class TestModeRelations:
    def test_shifted_is_a_sigmoid_reparameterisation(self) -> None:
        rng = np.random.default_rng(1)
        u = rng.normal(size=100)
        owner, xi = np.zeros(100, dtype=int), np.ones((1, 1))
        shifted = eq4_angles(xi, u, np.zeros(100), owner, mode="shifted")[0]
        np.testing.assert_allclose(shifted, TWO_PI / (1.0 + np.exp(-2.0 * u)))

    def test_wrapped_equals_literal_modulo_two_pi(self) -> None:
        ap = AnsatzParams(5, 2, "bel")
        owner = np.array(ap.noise_qubit_per_angle)
        xi, w, b = _extreme_inputs(5, ap.n_angles)
        lit = eq4_angles(xi, w, b, owner, mode="literal")
        wrapped = eq4_angles(xi, w, b, owner, mode="wrapped")
        diff = np.mod(lit - wrapped, TWO_PI)
        assert np.all(np.minimum(diff, TWO_PI - diff) < 1e-9)

    def test_wrapping_is_harmless_for_plain_rotations_but_not_controlled_ones(self) -> None:
        # Backs the angle_map docstring's claim numerically instead of asserting it.
        theta = 0.7
        ry, ry_wrapped = qml.matrix(qml.RY(theta, 0)), qml.matrix(qml.RY(theta + TWO_PI, 0))
        np.testing.assert_allclose(ry_wrapped, -ry, atol=1e-12)  # global phase only
        cry = qml.matrix(qml.CRY(theta, [0, 1]))
        cry_wrapped = qml.matrix(qml.CRY(theta + TWO_PI, [0, 1]))
        assert not np.allclose(cry_wrapped, cry)  # a real, physical difference
        assert not np.allclose(cry_wrapped, -cry)

    def test_all_modes_listed(self) -> None:
        assert set(ANGLE_MODES) == {"literal", "shifted", "wrapped"}
        assert math.isclose(TWO_PI, 2 * math.pi)