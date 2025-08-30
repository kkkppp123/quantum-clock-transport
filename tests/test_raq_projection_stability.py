# tests/test_raq_projection_stability.py
# v1.0 – Etap F: stabilność projekcji RAQ (toy)

import numpy as np
from numpy.linalg import norm
from scipy.linalg import eigh

from computations.raq_toy import (
    gaussian_raq_projector,
    box_raq_projector,
    physical_spectral_projector,
    dirac_observable_from_constraint,
    projector_error,
    build_toy_constraint,
)


def _eigpairs(C: np.ndarray):
    vals, vecs = eigh(0.5 * (C + C.conj().T))
    return vals, vecs


# ---------------------------------------------------------
# 1) Prawie-projektor i zgodność z projektorem spektralnym
# ---------------------------------------------------------


def test_raq_projector_gaussian_matches_exact():
    C, _ = build_toy_constraint(seed=7)

    P_exact = physical_spectral_projector(C, tol=1e-12)
    # sigma musi tłumić wartości ~0.1 – wybieramy duży, ale nadal szybki
    P_sigma = gaussian_raq_projector(C, sigma=40.0)

    # hermitowskość
    herm_err = norm(P_sigma - P_sigma.conj().T, 2)
    assert herm_err < 1e-12

    # prawie-projektor
    idem_err = projector_error(P_sigma, ord=2)
    assert (
        idem_err < 1e-3
    ), f"RAQ Gaussian not close to projector (||P^2-P||={idem_err:.2e})"

    # zgodność z „dokładnym” projektorem na ker(C)
    diff = norm(P_sigma - P_exact, 2)
    assert diff < 1e-3, f"RAQ Gaussian differs from spectral projector: {diff:.2e}"

    # wartości własne w [0, 1+ε]
    vals, _ = _eigpairs(P_sigma)
    assert vals.min() >= -1e-9
    assert vals.max() <= 1.0 + 1e-3


# ---------------------------------------------------------
# 2) Stabilność obserwabli Diraca po projekcji (operatorowo)
# ---------------------------------------------------------


def test_dirac_observable_projection_stability():
    C, _ = build_toy_constraint(seed=11)

    P_exact = physical_spectral_projector(C, tol=1e-12)
    P_sigma = gaussian_raq_projector(C, sigma=40.0)

    F = dirac_observable_from_constraint(C)  # [F, C] = 0

    F_phys_exact = P_exact @ F @ P_exact
    F_phys_sigma = P_sigma @ F @ P_sigma

    # Operatorowe odchylenie – powinno być na poziomie „śladu” tłumienia ~exp(-8) dla λ=0.1
    diff = norm(F_phys_sigma - F_phys_exact, 2)
    assert diff < 2e-3, f"Projected observable drift too large: {diff:.2e}"

    # Dodatkowo: stabilność względem lekkiej zmiany sigma
    P_sigma2 = gaussian_raq_projector(C, sigma=60.0)
    drift = norm(P_sigma2 @ F @ P_sigma2 - F_phys_sigma, 2)
    assert drift < 2e-3, f"Observable not stable vs sigma change: {drift:.2e}"


# ---------------------------------------------------------
# 3) Okno „pudełkowe” (sinc) – sanity: identyczność na jądrze
#    i silne tłumienie dla dużych |λ|
# ---------------------------------------------------------


def test_box_window_sanity_kernel_and_far_from_zero():
    C, _ = build_toy_constraint(seed=23)
    vals, vecs = _eigpairs(C)

    # wybierz wektory z λ≈0 oraz z dużym |λ|
    ker_idx = np.where(np.abs(vals) <= 1e-12)[0]
    far_idx = np.where(np.abs(vals) >= 0.5 - 1e-12)[0]

    assert ker_idx.size >= 1 and far_idx.size >= 1

    P_box = box_raq_projector(C, L=80.0)

    # Na jądrze: ~tożsamość
    for i in ker_idx:
        v = vecs[:, i]
        err = norm(P_box @ v - v)
        assert err < 5e-3, f"Box window fails to preserve kernel vector (err={err:.2e})"

    # Dla dużych |λ|: silne tłumienie (sinc małą wartość)
    for i in far_idx:
        v = vecs[:, i]
        amp = norm(P_box @ v)
        assert (
            amp < 5e-2
        ), f"Box window insufficient damping for |λ|~{abs(vals[i]):.2f} (amp={amp:.2e})"
