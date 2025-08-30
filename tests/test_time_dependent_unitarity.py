# tests/test_time_dependent_unitarity.py
# v1.0 – testy Etapu E: unitarność (gap), crossing-zero + H_alpha, sygnatura Höldera ~1/2

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

from computations.propagator import (
    time_ordered_propagator,
    evolve_time_dependent,
    unitarity_error,
)


def _rand_state(d: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=d) + 1j * rng.normal(size=d)
    return psi / norm(psi)


# ----------------------------
# 1) Reżim z luką – unitarność
# ----------------------------


def test_unitarity_gap():
    """
    Gładki H(T) z luką – sprawdzamy, że oba schematy dają bardzo mały błąd unitarności.
    """

    # 2x2 hermitowski z (niemal) stałą luką
    def H_gap(T: float) -> np.ndarray:
        Delta = 1.0 + 0.2 * T  # min ~ 0.8
        g = 0.3
        H = np.array([[Delta, g], [g, -Delta]], dtype=float)
        return H.astype(np.complex128)

    T = np.linspace(-1.0, 1.0, 240)  # umiarkowana siatka
    U_cn = time_ordered_propagator(H_gap, T, scheme="cn")
    U_m1 = time_ordered_propagator(H_gap, T, scheme="magnus1")

    err_cn = unitarity_error(U_cn, ord=2)
    err_m1 = unitarity_error(U_m1, ord=2)

    # CN zwykle ostrzejszy od expm-kroków; oba powinny być bardzo małe
    assert err_cn < 1e-12, f"CN unitarity too large: {err_cn:.2e}"
    assert err_m1 < 5e-12, f"Magnus-1 unitarity too large: {err_m1:.2e}"


# ----------------------------------------------------
# 2) Crossing-zero + bounded generator H_alpha (stabil)
# ----------------------------------------------------


def _bounded_generator(H: np.ndarray, alpha: float) -> np.ndarray:
    """
    H_alpha = H (I + (H/alpha)^2)^(-1/2)
    Dla hermitowskiego H realizujemy przez diagonalizację (eigh).
    Gwarantuje ||H_alpha|| <= alpha i gładkość przy dużych |H|.
    """
    Hh = 0.5 * (H + H.conj().T)
    vals, vecs = eigh(Hh)  # gwarantuje rzeczywiste wartości własne
    scal = vals / np.sqrt(1.0 + (vals / alpha) ** 2)
    return (vecs * scal) @ vecs.conj().T


def test_unitarity_crossing_bounded():
    """
    Crossing-zero: Δ(T)=kT, bez sprzężenia. Sprawdzamy:
      (a) unitarność przy CN,
      (b) porównanie trajektorii z „bounded” H_alpha – brak wysadzeń.
    """
    k = 3.0

    def H_cross(T: float) -> np.ndarray:
        Delta = k * T
        H = np.array([[Delta, 0.0], [0.0, -Delta]], dtype=float)
        return H.astype(np.complex128)

    alpha = 10.0  # > sup|λ| (~3) – bounded ale bliskie oryginałowi

    def H_cross_alpha(T: float) -> np.ndarray:
        return _bounded_generator(H_cross(T), alpha=alpha)

    T = np.linspace(-1.0, 1.0, 401)
    # Pełne propagatory (do testu unitarności)
    U_cn = time_ordered_propagator(H_cross, T, scheme="cn")
    U_cn_a = time_ordered_propagator(H_cross_alpha, T, scheme="cn")

    err_cn = unitarity_error(U_cn, ord=2)
    err_cn_a = unitarity_error(U_cn_a, ord=2)
    assert err_cn < 1e-12, f"CN unitarity (crossing) too large: {err_cn:.2e}"
    assert err_cn_a < 1e-12, f"CN unitarity (bounded) too large: {err_cn_a:.2e}"

    # Brak wysadzeń: końcowe stany są bliskie (H_alpha ~ H dla alpha >> |H|)
    psi0 = _rand_state(2, seed=123)
    psi_end_cn, _ = evolve_time_dependent(H_cross, T, psi0, scheme="cn")
    psi_end_bnd, _ = evolve_time_dependent(H_cross_alpha, T, psi0, scheme="cn")

    diff = norm(psi_end_cn - psi_end_bnd)
    # Przy alpha=10 różnice muszą być minimalne – dopuszczamy margines 1e-3
    assert (
        diff < 1e-3
    ), f"Bounded vs original diverged near crossing (||Δψ||={diff:.2e})"

    # Kontrola normy (dla pewności – dodatkowo do unitarity_error)
    assert abs(norm(psi_end_cn) - 1.0) < 1e-12
    assert abs(norm(psi_end_bnd) - 1.0) < 1e-12


# -----------------------------------------
# 3) Sygnatura Höldera 1/2 w okolicy zera
# -----------------------------------------


def test_holder_signature():
    """
    Sprawdzamy, że ||H(T+δ) - H(T)|| ~ δ^{1/2} przy T≈0.
    Wybieramy H(T) z widmem ~ ±sqrt(|T|).
    """

    def H_sqrt(T: float) -> np.ndarray:
        s = np.sqrt(abs(T))
        H = np.array([[s, 0.0], [0.0, -s]], dtype=float)
        return H.astype(np.complex128)

    T0 = 0.0
    deltas = np.logspace(-6, -2, 7)  # 1e-6 ... 1e-2
    diffs = []
    for d in deltas:
        D = H_sqrt(T0 + d) - H_sqrt(T0)
        diffs.append(norm(D, 2))
    x = np.log(deltas)
    y = np.log(np.array(diffs))
    slope, intercept = np.polyfit(x, y, 1)  # y ≈ slope * log δ + c

    # Sygnał ~0.5 z luzem ±0.1 (numeryka + mała liczba punktów)
    assert 0.4 <= slope <= 0.6, f"Hölder slope ≈ {slope:.3f}, oczekiwane ~0.5±0.1"
