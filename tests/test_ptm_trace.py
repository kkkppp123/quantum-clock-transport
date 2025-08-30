# tests/test_ptm_trace.py
# v1.0 (PR-1)
# Property test: for random U ∈ SU(2),
#   |Tr U|^2 == 1 + (r_xx + r_yy + r_zz)
#
# We tolerate tiny numerical error from floating arithmetic.

import numpy as np
import pytest

from qct.holonomy.ptm import (
    ptm_diag_expectations,
    to_su2,
    su2_sanity_error,
)


def _random_su2(rng: np.random.Generator) -> np.ndarray:
    """Haar-uniform SU(2) via random quaternion."""
    # Random unit quaternion q = (a, b, c, d) / ||.||
    v = rng.normal(size=4)
    v = v / np.linalg.norm(v)
    a, b, c, d = v  # real components
    # Map to SU(2)
    U = np.array(
        [
            [a + 1j * b, c + 1j * d],
            [-c + 1j * d, a - 1j * b],
        ],
        dtype=complex,
    )
    # det(U)=a^2+b^2+c^2+d^2=1 ⇒ det=1 automatically
    return U


@pytest.mark.parametrize("n_samples", [64])
def test_ptm_trace_identity(n_samples):
    rng = np.random.default_rng(12345)
    errs = []
    for _ in range(n_samples):
        U = _random_su2(rng)
        # Sanity: SU(2)
        # (no need to project; but keep helper usage explicit)
        U = to_su2(U)

        rxx, ryy, rzz = ptm_diag_expectations(U)
        lhs = abs(np.trace(U)) ** 2
        rhs = 1.0 + (rxx + ryy + rzz)
        errs.append(abs(lhs - rhs))

    mean_err = float(np.mean(errs))
    max_err = float(np.max(errs))

    # We expect extremely small errors (~1e-15..1e-12). Be conservative.
    assert mean_err < 5e-12
    assert max_err < 1e-10


def test_su2_sanity_error():
    rng = np.random.default_rng(7)
    U = _random_su2(rng)
    e = su2_sanity_error(U)
    assert e < 1e-10
