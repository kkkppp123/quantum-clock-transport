# tests/test_phase_invariant_distance.py
# v1.0 (PR-1)
# Validate: min_phi || e^{i phi} U - I ||_F == sqrt(2d - 2 |Tr U|)
# for d=2. We compute the LHS by choosing phi* = -arg(Tr U).

import numpy as np
import pytest

from qct.holonomy.ptm import phase_invariant_distance, is_unitary


def _random_unitary_2x2(rng: np.random.Generator) -> np.ndarray:
    """Random U(2) via QR decomposition of a complex Ginibre matrix."""
    Z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(Z)
    # Make Q unitary by normalizing phases on diagonal of R
    D = np.diag(R.diagonal() / np.abs(R.diagonal()))
    U = Q @ D
    return U


@pytest.mark.parametrize("n_samples", [64])
def test_phase_min_distance_matches_closed_form(n_samples):
    rng = np.random.default_rng(2025)
    for _ in range(n_samples):
        U = _random_unitary_2x2(rng)
        assert is_unitary(U)

        trU = np.trace(U)
        abs_tr = abs(trU)
        # Closed form
        rhs = phase_invariant_distance(abs_tr, d=2)

        # Direct minimization via analytic phi* = -arg(Tr U)
        phi_star = -np.angle(trU) if abs_tr > 0 else 0.0
        eiph = np.exp(1j * phi_star)
        lhs = np.linalg.norm(eiph * U - np.eye(2), ord="fro")

        assert abs(lhs - rhs) < 1e-9


def test_phase_min_distance_for_su2_example():
    # Simple rotation U = exp(-i θ σ_z / 2):
    theta = 0.8
    U = np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])
    # |Tr U| = |2 cos(θ/2)|
    abs_tr = abs(2 * np.cos(theta / 2))
    rhs = phase_invariant_distance(abs_tr, d=2)
    # Brute force check
    phi_star = -np.angle(np.trace(U))
    lhs = np.linalg.norm(np.exp(1j * phi_star) * U - np.eye(2), ord="fro")
    assert abs(lhs - rhs) < 1e-12
