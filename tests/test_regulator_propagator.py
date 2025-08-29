# tests/test_regulator_propagator.py
from __future__ import annotations

from numpy.linalg import norm

from computations.evolution import (
    make_default_Hs,
    evolve,
)


def _evolve_all(
    eps: float = 1e-8,
    alpha: float = 1200.0,  # mocniej niż 400.0 -> lepsza zbieżność O(alpha^-2)
    dt: float = 4e-4,  # mniejszy krok czasowy stabilizuje różnice
    steps: int = 320,  # nieco dłuższa ewolucja, by kompensować mniejsze dt
):
    H_ref, H_eps, H_alpha, H_pade, psi0 = make_default_Hs(
        n=140, m=1.0, T=0.4, eps=eps, alpha=alpha, common_shift=1e-10
    )

    psi_ref = evolve(H_ref, psi0, dt=dt, steps=steps)
    psi_e = evolve(H_eps, psi0, dt=dt, steps=steps)
    psi_a = evolve(H_alpha, psi0, dt=dt, steps=steps)
    psi_p = evolve(H_pade, psi0, dt=dt, steps=steps)

    return psi_ref, psi_e, psi_a, psi_p


def test_regulator_independence_small_errors():
    """
    Różnice między propagatorami (dla sensownych ε, α, małego dt) powinny być małe.
    """
    # podkręcone: alpha, mniejszy dt, lekko więcej kroków
    psi_ref, psi_e, psi_a, psi_p = _evolve_all(
        eps=1e-8, alpha=1200.0, dt=4e-4, steps=320
    )

    err_eps = norm(psi_ref - psi_e)
    err_alpha = norm(psi_ref - psi_a)
    err_pade = norm(psi_ref - psi_p)

    # rygorystycznie: wszystkie do 1e-6
    assert err_eps < 1e-6, f"ε-regulator różni się zbyt mocno: {err_eps:.3e}"
    assert err_alpha < 1e-6, f"bounded-generator różni się zbyt mocno: {err_alpha:.3e}"
    assert err_pade < 1e-6, f"Padé różni się zbyt mocno: {err_pade:.3e}"
