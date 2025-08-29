# tests/test_curvature_scaling.py
import numpy as np

from computations.operators import make_volume_grid, theta_matrix
from computations.curvature import curvature_norm


def _fit_log_slope(eps_list, vals):
    x = np.log(eps_list)
    y = np.log(np.maximum(vals, 1e-30))
    return np.polyfit(x, y, 1)[0]


def _measure_slope(ordering: str, n: int = 120, phi: float = 0.3, TY: float = 0.4):
    v, w = make_volume_grid(n=n)
    Theta = theta_matrix(v, w)

    # drabinka eps w proporcjach 1/2 — stabilna numerycznie
    eps_list = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3])
    vals = [
        curvature_norm(Theta, v, 1.0, phi=phi, TY=TY, eps=eps, ordering=ordering)
        for eps in eps_list
    ]
    slope = _fit_log_slope(eps_list, vals)
    return slope, eps_list, vals


def test_curvature_slope_weyl():
    slope, eps_list, vals = _measure_slope("weyl", n=120)
    # okno tolerancji delikatnie szersze, bo 2-norma i różniczkowanie centralne
    assert 1.9 <= slope <= 2.1, (
        f"Weyl powinien ~2.0; slope={slope:.3f}; "
        f"vals={np.array(vals)}; eps={eps_list}"
    )


def test_curvature_slope_bj():
    slope, eps_list, vals = _measure_slope("born--jordan", n=120)
    assert 0.9 <= slope <= 1.1, (
        f"Born–Jordan powinien ~1.0; slope={slope:.3f}; "
        f"vals={np.array(vals)}; eps={eps_list}"
    )
