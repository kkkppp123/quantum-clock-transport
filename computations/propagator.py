# computations/propagator.py
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np
from numpy.linalg import norm
from scipy.linalg import expm  # dla kroku Magnus-1

Array = np.ndarray

__all__ = [
    "as_hermitian",
    "crank_nicolson_step",
    "cn_unitary",
    "time_ordered_propagator",
    "evolve_time_dependent",
    "unitarity_error",
]


# ---------- helpers ----------


def as_hermitian(H: Array) -> Array:
    """Zwróć część hermitowską (H + H†)/2 jako np.complex128."""
    Hc = np.asarray(H, dtype=np.complex128)
    return 0.5 * (Hc + Hc.conj().T)


# ---------- jednostkowy krok CN / Magnus-1 ----------


def cn_unitary(H: Array, dT: float) -> Array:
    """Krok Crank–Nicolson dla (na tym przedziale) stałego H:
    U = (I + i dT/2 H)^(-1) (I - i dT/2 H)."""
    Hh = as_hermitian(H)
    n = Hh.shape[0]
    Id = np.eye(n, dtype=np.complex128)
    A = Id - 1j * (dT / 2.0) * Hh
    B = Id + 1j * (dT / 2.0) * Hh
    # Rozwiązujemy B X = A  =>  X = B^{-1} A
    return np.linalg.solve(B, A)


def magnus1_unitary(H: Array, dT: float) -> Array:
    """Krok Magnus-1: U ≈ exp(-i H dT) dla (na przedziale) stałego H."""
    Hh = as_hermitian(H)
    return expm(-1j * dT * Hh)


def crank_nicolson_step(H: Array, psi: Array, dT: float) -> Array:
    """Aktualizacja stanu ψ_{t+dT} z ψ_t krokiem CN."""
    return cn_unitary(H, dT) @ np.asarray(psi, dtype=np.complex128)


# ---------- uporządkowany czasowo propagator ----------


def _step_unitary(H: Array, dT: float, scheme: str) -> Array:
    if scheme == "cn":
        return cn_unitary(H, dT)
    if scheme == "magnus1":
        return magnus1_unitary(H, dT)
    raise ValueError(f"unknown scheme='{scheme}', expected 'cn' or 'magnus1'")


def time_ordered_propagator(
    H_of_t: Callable[[float], Array],
    t_grid: Iterable[float],
    scheme: str = "cn",
    midpoint: bool = True,
) -> Array:
    """
    Sklejany po przedziałach propagator uporządkowany czasowo na siatce t0 < ... < tN.

    Parameters
    ----------
    H_of_t : callable
        Zwraca hermitowski H(t) (macierz n×n) dla skalarnego t.
    t_grid : 1D iterable
        Węzły czasu. Na każdym [t_k, t_{k+1}] używamy kroku CN lub Magnus-1
        z H wyznaczonym w punkcie środkowym (domyślnie) lub lewym.
    scheme : {'cn','magnus1'}
        Metoda kroku na przedziale.
    midpoint : bool
        Jeśli True, używa H( (t_k + t_{k+1})/2 ); inaczej H(t_k).

    Returns
    -------
    U : ndarray (n, n)
        Całkowity propagator ≈ T exp(-i ∫ H dt).
    """
    t = np.asarray(tuple(t_grid), dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t_grid must be 1D with at least two points")

    # Rozmiar ustalamy z pierwszego H
    H0 = as_hermitian(H_of_t(float(t[0])))
    n = H0.shape[0]
    U = np.eye(n, dtype=np.complex128)

    for k in range(t.size - 1):
        a, b = float(t[k]), float(t[k + 1])
        dT = b - a
        tk = 0.5 * (a + b) if midpoint else a
        Hk = as_hermitian(H_of_t(tk))
        Uk = _step_unitary(Hk, dT, scheme=scheme)
        # porządkowanie czasowe: najnowszy krok mnoży z LEWEJ
        U = Uk @ U

    return U


def evolve_time_dependent(
    H_of_t: Callable[[float], Array],
    t_grid: Iterable[float],
    psi0: Array,
    scheme: str = "cn",
    midpoint: bool = True,
) -> Tuple[Array, Array]:
    """
    Zwróć (psi_end, U_total) dla ewolucji zadaną metodą ('cn' lub 'magnus1').

    psi_end = U_total @ psi0
    """
    U = time_ordered_propagator(H_of_t, t_grid, scheme=scheme, midpoint=midpoint)
    psi_end = U @ np.asarray(psi0, dtype=np.complex128)
    return psi_end, U


# ---------- diagnostyka ----------


def unitarity_error(U: Array, ord: int | float = 2) -> float:
    """Zwraca ‖U†U − I‖ w zadanej normie macierzowej (np. 2 lub 'fro')."""
    Uc = np.asarray(U, dtype=np.complex128)
    n = Uc.shape[0]
    Id = np.eye(n, dtype=np.complex128)
    return float(norm(Uc.conj().T @ Uc - Id, ord=ord))
