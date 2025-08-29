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

__all__ = [
    "H_eps_from_params",
    "H_bounded_from_H",
    "H_pade_from_params",
    "crank_nicolson_step",
    "evolve",
    "make_default_Hs",
]


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
    Bounded-generator: H_α = H (I + H^2/α^2)^(-1/2) — liczony spektralnie.
    """
    lam, Q = eigh(H)  # H hermitowski -> wartości własne rzeczywiste
    damp = lam / np.sqrt(1.0 + (lam / alpha) ** 2)
    # f(H) = Q diag(f(lam)) Q^T, mnożenie kolumn przez damp jest OK
    return (Q * damp) @ Q.T


def H_pade_from_params(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    T: float,
    shift: float = 1e-10,
) -> np.ndarray:
    """
    Racjonalna aproksymacja sqrt na X = Op + shift*I:
        sqrt(X) ≈ (1/√s) [ a I + b Y (I + d Y)^(-1) ],  Y = s X,  spec(Y) ⊂ [0,1].
    Bierzemy prosty Padé[1,1] stabilny w naszym zakresie.
    """
    Op = O_matrix(Theta, U_matrix(v, m=m, T=T))
    X = Op + shift * np.eye(Op.shape[0])

    # Skala s, by norm(Y) <= 1
    lam_max = np.abs(eigh(X, eigvals_only=True)).max()
    s = 1.0 if lam_max <= 0 else 1.0 / lam_max
    Y = s * X

    a = 0.0
    b = 1.0
    d = 0.5
    Id = np.eye(X.shape[0])
    sqrtX_approx = (1.0 / np.sqrt(s)) * (a * Id + b * Y @ solve(Id + d * Y, Id))
    # symetryzacja dla bezpieczeństwa numerycznego
    return 0.5 * (sqrtX_approx + sqrtX_approx.T)


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
    eps: float = 1e-6,  # mniejsze ε, by trafić w próg 1e-6
    alpha: float = 200.0,  # mocniejszy bounded-generator
    common_shift: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Zwraca: (H_exact, H_eps, H_alpha, H_pade, psi0)
    - H_exact: sqrt(Op + common_shift I),
    - H_eps:   sqrt(Op + (ε + common_shift) I),
    - H_alpha: bounded-generator z H_exact,
    - H_pade:  Padé dla Op + common_shift I.
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
