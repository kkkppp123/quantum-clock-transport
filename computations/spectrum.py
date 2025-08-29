# computations/spectrum.py
from __future__ import annotations

import numpy as np
from numpy.linalg import eigvalsh

from .operators import U_matrix, O_matrix  # Θ + U(T) => O(T)

__all__ = ["eig_min", "gap_OT", "scan_gap_T", "scan_gap_multi_scales"]


def eig_min(A: np.ndarray) -> float:
    """
    Najmniejsza wartość własna macierzy Hermitowskiej.
    Dla stabilności symetryzujemy przez (A + A.T)/2.
    """
    Ah = 0.5 * (A + A.T)
    vals = eigvalsh(Ah)
    return float(vals[0])


def gap_OT(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T: float,
    scale: float = 1.0,
) -> float:
    """
    Luka widmowa μ(T) = min σ( O(T) ), gdzie O(T) = scale*Θ + U(T).
    """
    Op = O_matrix(scale * Theta, U_matrix(v, m=m, T=T))
    return eig_min(Op)


def scan_gap_T(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T_vals: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Zwraca wektor μ(T) dla siatki T_vals przy ustalonym scale.
    """
    mus = [gap_OT(Theta, v, m, T, scale=scale) for T in T_vals]
    return np.asarray(mus, dtype=float)


def scan_gap_multi_scales(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T_vals: np.ndarray,
    scales: tuple[float, ...] = (1.0,),
) -> dict[float, np.ndarray]:
    """
    Dla kilku wartości 'scale' zwraca słownik: scale -> μ(T) na siatce T.
    """
    return {s: scan_gap_T(Theta, v, m, T_vals, scale=s) for s in scales}


if __name__ == "__main__":
    from .operators import make_volume_grid, theta_matrix

    v, w = make_volume_grid(n=60)
    Theta = theta_matrix(v, w)
    T_vals = np.linspace(-1.0, 1.0, 9)
    for s in (1.0, 1e-3, 1e-6):
        mus = scan_gap_T(Theta, v, m=1.0, T_vals=T_vals, scale=s)
        print(f"scale={s:>7}: μ(T) = {np.array2string(mus, precision=6)}")
