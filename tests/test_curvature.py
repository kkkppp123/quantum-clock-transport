# tests/test_curvature.py
import numpy as np

from computations.operators import make_volume_grid, theta_matrix
from computations.curvature import curvature_norm


def _fit_log_slope(xs, ys):
    # dopasuj k w prawie potęgowym: y ≈ C * x^k
    return np.polyfit(np.log(xs), np.log(ys), 1)[0]


def test_curvature_scaling_weyl_O_eps2():
    """
    Dla porządku Weyla (model anulujący składniki O(ε)) oczekujemy ~ ε^2.
    """
    v, w = make_volume_grid(n=90)
    Theta = theta_matrix(v, w)

    eps_list = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])
    vals = [
        curvature_norm(Theta, v, 1.0, phi=0.3, TY=0.4, eps=eps, ordering="weyl")
        for eps in eps_list
    ]
    slope = _fit_log_slope(eps_list, vals)
    assert 1.7 <= slope <= 2.3, f"Spodziewany ~2, wyszło {slope:.3f}; vals={vals}"


def test_curvature_scaling_born_jordan_O_eps():
    """
    Dla porządku Born–Jordan brak anulacji O(ε) => spodziewamy ~ ε^1.
    """
    v, w = make_volume_grid(n=90)
    Theta = theta_matrix(v, w)

    eps_list = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])
    vals = [
        curvature_norm(Theta, v, 1.0, phi=0.3, TY=0.4, eps=eps, ordering="born--jordan")
        for eps in eps_list
    ]
    slope = _fit_log_slope(eps_list, vals)
    assert 0.7 <= slope <= 1.3, f"Spodziewany ~1, wyszło {slope:.3f}; vals={vals}"


def test_curvature_goes_to_zero_as_eps_to_zero():
    """
    Sanity: ||F|| -> 0 dla ε -> 0 niezależnie od orderingu.
    """
    v, w = make_volume_grid(n=70)
    Theta = theta_matrix(v, w)

    for ordering in ("weyl", "born--jordan"):
        v1 = curvature_norm(Theta, v, 1.0, phi=0.2, TY=0.5, eps=1e-1, ordering=ordering)
        v2 = curvature_norm(Theta, v, 1.0, phi=0.2, TY=0.5, eps=1e-3, ordering=ordering)
        assert v2 < v1 * 0.2, f"Krzywizna nie maleje wystarczająco dla {ordering}"
