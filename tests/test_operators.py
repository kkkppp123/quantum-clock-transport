# tests/test_operators.py
import numpy as np
from numpy.linalg import eigvalsh
from computations.operators import make_volume_grid, theta_matrix, U_matrix, O_matrix


def test_theta_is_hermitian_and_psd():
    v, w = make_volume_grid(n=50)
    Theta = theta_matrix(v, w)
    assert np.allclose(Theta, Theta.T, atol=1e-12)
    lam = eigvalsh(Theta)
    assert lam.min() >= -1e-10  # tolerancja numeryczna


def test_OT_is_hermitian_psd_and_monotone_in_T():
    v, w = make_volume_grid(n=60)
    Theta = theta_matrix(v, w)

    O1 = O_matrix(Theta, U_matrix(v, m=1.0, T=0.5))
    O2 = O_matrix(Theta, U_matrix(v, m=1.0, T=1.0))

    assert np.allclose(O1, O1.T, atol=1e-12)
    assert np.allclose(O2, O2.T, atol=1e-12)

    # PSD w tolerancji maszynowej
    assert eigvalsh(O1).min() >= -1e-10
    assert eigvalsh(O2).min() >= -1e-10

    # większe |T| ⇒ większe najmniejsze własne (monotoniczność U w T^2)
    assert eigvalsh(O2).min() >= eigvalsh(O1).min() - 1e-8
