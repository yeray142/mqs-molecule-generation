"""Eq. 4 angle map: noise xi -> rotation angles theta.

    theta_{q,l,k} = 2*pi * tanh( xi_q * W_{qlk} + b_{qlk} )          (Eq. 4)

One noise component per qubit (xi_q); a scalar W and b per angle. Works on
anything ``pennylane.math`` dispatches over (numpy, torch, autograd, ...).

A tension in the paper itself, surfaced rather than papered over: the text says
the angles live in [0, 2*pi], but tanh is odd with range (-1, 1), so Eq. 4 as
printed gives angles in (-2*pi, 2*pi) -- negative whenever the tanh argument is.
(The parenthesisation above is my reading of the printed equation, whose
brackets were lost in the text I could access.) Three modes, because they are
different circuits:

  "literal"  2*pi * tanh(u)       Eq. 4 exactly as printed. Range (-2pi, 2pi).
  "shifted"  pi * (1 + tanh(u))   In [0, 2pi]; smooth. Equal to 2*pi*sigmoid(2u),
                                  i.e. the same family with W, b rescaled by 2.
  "wrapped"  (2*pi*tanh(u)) mod 2pi
                                  In [0, 2pi]; identical to "literal" for RY/RZ up
                                  to a global phase, but NOT for controlled
                                  rotations (CRY(theta + 2pi) != CRY(theta): it
                                  flips the sign of the control-on branch).

The default is "literal" because it is what the paper prints, but the stated
[0, 2pi] domain only holds for "shifted" / "wrapped".

Status: CONFIRMED against the paper's Eq. 4, which prints
``theta_{q,l,k} = 2 pi tanh(xi_q W_{qlk} + b_{qlk})``, one trainable W and one b per angle
(hence N_params = 2 * n_angles) and one noise component xi_q per QUBIT. The surrounding
prose says the angles lie in the domain [0, 2 pi], which that formula does not produce --
tanh is signed, so the literal range is (-2 pi, 2 pi). The formula is taken as
authoritative and is the ``literal`` default; ``shifted`` and ``wrapped`` implement the
prose instead, for testing which reading reproduces the paper's numbers.
"""

from __future__ import annotations

import math

import pennylane as qml

TWO_PI = 2.0 * math.pi
ANGLE_MODES = ("literal", "shifted", "wrapped")


def eq4_angles(xi, weights, biases, noise_qubit, mode: str = "literal"):
    """Map noise to rotation angles.

    Args:
        xi: Noise, shape (..., n_qb).
        weights: W, shape (n_angles,).
        biases: b, shape (n_angles,).
        noise_qubit: Integer index array, shape (n_angles,): which xi_q feeds each angle.
        mode: One of ANGLE_MODES.

    Returns:
        Angles, shape (..., n_angles).
    """
    if mode not in ANGLE_MODES:
        raise ValueError(f"mode must be one of {ANGLE_MODES}, got {mode!r}")
    u = xi[..., noise_qubit] * weights + biases
    t = qml.math.tanh(u)
    if mode == "literal":
        return TWO_PI * t
    if mode == "shifted":
        return math.pi * (1.0 + t)
    x = TWO_PI * t
    return qml.math.where(x < 0, x + TWO_PI, x)