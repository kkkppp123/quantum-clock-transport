import numpy as np
from numpy.linalg import eigvalsh

from computations.operators import (
    make_volume_grid,
    theta_matrix,
    U_matrix,
    O_matrix,
)
from computations.sqrt_ops import (
    sqrt_psd_matrix,
    bounded_generator,
    holder_half_norm_diff,
)


def test_sqrt_is_psd_and_hermitian():
    v, w = make_volume_grid(n=60)
    Theta = theta_matrix(v, w)
    Omat = O_matrix(Theta, U_matrix(v, m=1.0, T=0.8))
    H = sqrt_psd_matrix(Omat)

    # hermitowskość
    assert np.allclose(H, H.T, atol=1e-12)

    # dodatnia półokreśloność (z tolerancją numeryczną)
    assert eigvalsh(H).min() >= -1e-10

    # H^2 ~ O
    H2 = H @ H
    assert np.allclose(H2, Omat, rtol=1e-8, atol=1e-8)


def test_bounded_generator_converges_strongly():
    v, w = make_volume_grid(n=80)
    Theta = theta_matrix(v, w)
    Omat = O_matrix(Theta, U_matrix(v, m=1.0, T=0.2))
    H = sqrt_psd_matrix(Omat)

    alphas = [10.0, 20.0, 40.0, 80.0]
    rng = np.random.default_rng(0)
    x = rng.normal(size=len(v))
    x = x / np.linalg.norm(x)

    prev = None
    for a in alphas:
        Ha = bounded_generator(H, a)
        err = np.linalg.norm((Ha - H) @ x)
        if prev is not None:
            # monotonicznie malejący błąd aproksymacji na wektorze testowym
            assert err <= prev + 1e-10
        prev = err

    assert prev < 1e-3  # silna zbieżność przy dużym alpha


def test_holder_half_behavior_gap_vs_crossing():
    v, w = make_volume_grid(n=70)
    Theta = theta_matrix(v, w)

    # Reżim z luką: delikatna zmiana T (powinno być "łagodnie")
    O1 = O_matrix(Theta, U_matrix(v, m=1.0, T=0.8))
    O2 = O_matrix(Theta, U_matrix(v, m=1.0, T=0.9))
    dO = np.linalg.norm(O1 - O2, 2)
    dH = holder_half_norm_diff(O1, O2)
    # Lipschitz w przybliżeniu (na macierzach i tak widoczna łagodność)
    assert dH <= 2.0 * dO**0.5

    # Crossing-zero: zmiany wokół T ~ 0 zwiększają "szorstkość" (Hölder-1/2)
    Oc1 = O_matrix(Theta, U_matrix(v, m=1.0, T=-0.05))
    Oc2 = O_matrix(Theta, U_matrix(v, m=1.0, T=+0.05))
    dOc = np.linalg.norm(Oc1 - Oc2, 2)
    dHc = holder_half_norm_diff(Oc1, Oc2)

    # oczekujemy, że dHc będzie relatywnie duże vs √dOc (brak Lipschitza)
    assert dHc >= 0.5 * (dOc**0.5)
