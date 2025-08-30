# qct/holonomy/ptm.py
# v1.0 (PR-1) — PTM/holonomy core utilities for 1-qubit SU(2)
# - Gauge-invariant |Tr U| via diagonal PTM expectations
# - Phase-invariant distance: min_phi || e^{i phi} U - I ||_F
# - Sanity checks and helpers
#
# Dependencies: numpy
#
# Public API:
#   ptm_diag_expectations(U) -> (r_xx, r_yy, r_zz)
#   trace_magnitude_from_ptm(r_xx, r_yy, r_zz) -> float
#   phase_invariant_distance(abs_tr: float, d: int = 2) -> float
#   su2_sanity_error(U) -> float
#
# Notes:
# - For any U ∈ SU(2), with Pauli transfer matrix R (SO(3) rep.),
#   Tr R = r_xx + r_yy + r_zz and |Tr U|^2 = 1 + Tr R.
# - The phase-minimized Frobenius distance theorem yields:
#   min_phi || e^{i phi} U - I ||_F = sqrt(2d - 2 |Tr U|) for U ∈ U(d).

from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "ptm_diag_expectations",
    "trace_magnitude_from_ptm",
    "phase_invariant_distance",
    "su2_sanity_error",
    "is_unitary",
    "to_su2",
]

_I2 = np.eye(2, dtype=complex)
_SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
_SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = (_SIGMA_X, _SIGMA_Y, _SIGMA_Z)


def is_unitary(U: ArrayLike, tol: float = 1e-10) -> bool:
    """Check unitarity of a 2x2 matrix U with Frobenius tolerance."""
    U = np.asarray(U, dtype=complex)
    if U.shape != (2, 2):
        return False
    diff = np.linalg.norm(U.conj().T @ U - _I2, ord="fro")
    return diff <= tol


def to_su2(U: ArrayLike) -> np.ndarray:
    """Project U (2x2) to SU(2) by removing global phase so that det=1.
    If det(U)=0 (shouldn't happen for unitary), returns U unchanged.
    """
    U = np.asarray(U, dtype=complex)
    det = np.linalg.det(U)
    if np.isclose(det, 0.0):
        return U
    # Divide by phase factor sqrt(det) to ensure det=1 (choose principal branch).
    phase = np.sqrt(det)
    return U / phase


def _ptm_matrix(U: np.ndarray) -> np.ndarray:
    """Pauli Transfer Matrix R for one qubit: R_ij = (1/2) Tr[ σ_i U σ_j U† ], i,j∈{x,y,z}.
    Returns 3x3 real matrix (up to numerical noise).
    """
    R = np.zeros((3, 3), dtype=float)
    Ud = U.conj().T
    for i, Si in enumerate(_PAULI):
        for j, Sj in enumerate(_PAULI):
            val = 0.5 * np.trace(Si @ U @ Sj @ Ud)
            # R is real; clip small imaginary residues
            R[i, j] = float(np.real_if_close(val))
    return R


def ptm_diag_expectations(U: ArrayLike) -> tuple[float, float, float]:
    """Return diagonal entries (r_xx, r_yy, r_zz) of the 1-qubit PTM.
    These equal expectations of measurements in the same basis for input +1 eigenstates,
    and coincide with the diagonal of the SO(3) rotation that SU(2) implements on the Bloch sphere.
    """
    U = np.asarray(U, dtype=complex)
    R = _ptm_matrix(U)
    return float(R[0, 0]), float(R[1, 1]), float(R[2, 2])


def trace_magnitude_from_ptm(r_xx: float, r_yy: float, r_zz: float) -> float:
    """Recover |Tr U| from diagonal PTM entries for U ∈ SU(2).
    Uses identity: |Tr U|^2 = 1 + (r_xx + r_yy + r_zz).
    Clamps inside sqrt to zero for numerical safety.
    """
    s = 1.0 + (r_xx + r_yy + r_zz)
    return float(np.sqrt(max(0.0, s)))


def phase_invariant_distance(abs_tr: float, d: int = 2) -> float:
    """Compute min_phi || e^{i phi} U - I ||_F given |Tr U| and dimension d.
    Closed form: sqrt(2d - 2|Tr U|).
    """
    # Numerical safety: |Tr U| ∈ [0, d]
    abs_tr_clamped = min(max(abs_tr, 0.0), float(d))
    val = 2.0 * d - 2.0 * abs_tr_clamped
    return float(np.sqrt(max(0.0, val)))


def su2_sanity_error(U: ArrayLike) -> float:
    """Numerical sanity: return | |Tr U|^2 - (1 + r_xx + r_yy + r_zz) |.
    Should be ≈ 0 for U ∈ SU(2)."""
    U = to_su2(U)
    trU = np.trace(U)
    lhs = np.abs(trU) ** 2
    rxx, ryy, rzz = ptm_diag_expectations(U)
    rhs = 1.0 + (rxx + ryy + rzz)
    return float(abs(lhs - rhs))
