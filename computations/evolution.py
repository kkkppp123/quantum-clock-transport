# computations/evolution.py
from __future__ import annotations

import numpy as np
from numpy.linalg import solve, norm
from scipy.linalg import eigh

from .operators import (
    make_volume_grid,
    theta_matrix,
    U_matrix,
    O_matrix,
)
from .sqrt_ops import sqrt_psd_matrix


# ---------- Konstruktory Hamiltonianów (regulatory) ----------


def H_eps_from_params(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T: float,
    eps: float,
    shift: float = 1e-10,
) -> np.ndarray:
    """
    Regulator typu 'εI': H_ε = sqrt(Op + ε I), gdzie Op = Theta + U(T).
    Używamy tego samego 'shift' co w referencji, by wyeliminować bias.
    """
    Op = O_matrix(Theta, U_matrix(v, m=m, T=T))
    Op = Op + (eps + shift) * np.eye(Op.shape[0])
    return sqrt_psd_matrix(Op)


def H_bounded_from_H(H: np.ndarray, alpha: float) -> np.ndarray:
    """
    Bounded-generator: H_α = H (I + H^2/α^2)^(-1/2) (spektralnie).
    """
    lam, Q = eigh(H)
    damp = lam / np.sqrt(1.0 + (lam / alpha) ** 2)
    return (Q * damp) @ Q.T


def _matrix_sqrt_denman_beavers(X: np.ndarray, iters: int = 16) -> np.ndarray:
    """
    Stabilna iteracja Denmana–Beaversa dla SPD/PSD:
        Y_{k+1} = 1/2 (Y_k + Z_k^{-1})
        Z_{k+1} = 1/2 (Z_k + Y_k^{-1})
    z inicjalizacją Y_0 = X_scaled, Z_0 = I oraz skalowaniem 2^{-s} (||X_scaled|| ~ 1).

    Dodatkowo: pojedynczy krok Newtona dla sqrt po cofnięciu skali (polish).
    """
    n = X.shape[0]
    # skalowanie: X = 2^s * X_scaled, z lam_max ~ ||X||_2
    lam_max = np.abs(eigh(X, eigvals_only=True)).max()
    if lam_max <= 0:
        return np.zeros_like(X)

    # wybierz s tak, aby lam_max / 2^s ~ O(1)
    s = int(max(0, np.ceil(np.log2(lam_max)) - 1))
    X_scaled = X / (2.0**s)

    Y = X_scaled.copy()
    Z = np.eye(n)
    eye_mat = np.eye(n)

    for _ in range(iters):
        # unikamy jawnych inwersji: rozwiązujemy układy
        Z_inv = solve(Z, eye_mat)
        Y_inv = solve(Y, eye_mat)
        Y = 0.5 * (Y + Z_inv)
        Z = 0.5 * (Z + Y_inv)

    # cofnij skalowanie
    Y = (2.0 ** (s / 2.0)) * Y

    # Jedna iteracja Newtona dla sqrt: Y <- 1/2 (Y + X Y^{-1})
    # (wyraźnie zmniejsza błąd bez dużego kosztu)
    invY = solve(Y, np.eye(n))
    Y = 0.5 * (Y + X @ invY)

    # symetryzacja numeryczna
    return 0.5 * (Y + Y.T)


def H_pade_from_params(  # nazwa historyczna; implementacja DB + polish
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T: float,
    shift: float = 1e-10,
) -> np.ndarray:
    """
    'Padé' regulator: skalowana iteracja Denmana–Beaversa z pojedynczym
    krokiem Newtona na koniec. Bardzo dokładna w klasie aproksymacji racjonalnych.
    """
    Op = O_matrix(Theta, U_matrix(v, m=m, T=T))
    X = Op + shift * np.eye(Op.shape[0])
    Y = _matrix_sqrt_denman_beavers(X, iters=16)
    return 0.5 * (Y + Y.T)


# ---------- Jednostopniowy krok w czasie (unitarny) ----------


def crank_nicolson_step(H: np.ndarray, psi: np.ndarray, dt: float) -> np.ndarray:
    """
    (Id + i dt/2 H) ψ^{n+1} = (Id - i dt/2 H) ψ^n    (Crank–Nicolson)
    """
    Id = np.eye(H.shape[0])
    A = Id + 0.5j * dt * H
    B = Id - 0.5j * dt * H
    return solve(A, B @ psi)


def evolve(H: np.ndarray, psi0: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """
    Ewolucja ψ_{n+1} = CN_step(H, ψ_n, dt). Zakładamy stały H (małe okno czasu).
    """
    psi = psi0.copy()
    for _ in range(steps):
        psi = crank_nicolson_step(H, psi, dt)
    return psi / norm(psi)


# ---------- Scenariusz pomocniczy (z gotowymi parametrami) ----------


def make_default_Hs(
    n: int = 120,
    m: float = 1.0,
    T: float = 0.4,
    eps: float = 1e-6,  # małe ε, by trafić w próg 1e-6
    alpha: float = 200.0,  # bounded-generator
    common_shift: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zwraca: (H_exact, H_eps, H_alpha, H_pade, psi0)
    - H_exact: sqrt(Op + common_shift I),
    - H_eps:   sqrt(Op + (ε + common_shift) I),
    - H_alpha: bounded-generator z H_exact,
    - H_pade:  DB+Newton dla Op + common_shift I.
    """
    v, w = make_volume_grid(n=n)
    Theta = theta_matrix(v, w)

    H_exact = H_eps_from_params(Theta, v, m=m, T=T, eps=0.0, shift=common_shift)
    H_eps = H_eps_from_params(Theta, v, m=m, T=T, eps=eps, shift=common_shift)
    H_alpha = H_bounded_from_H(H_exact, alpha=alpha)
    H_pade = H_pade_from_params(Theta, v, m=m, T=T, shift=common_shift)

    rng = np.random.default_rng(0)
    psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
    psi0 = psi0 / norm(psi0)

    return H_exact, H_eps, H_alpha, H_pade, psi0
