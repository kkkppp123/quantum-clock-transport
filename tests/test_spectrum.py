# tests/test_spectrum.py
import numpy as np
from numpy.linalg import norm

from computations.operators import (
    make_volume_grid,
    theta_matrix,
    U_matrix,
    O_matrix,
)
from computations.sqrt_ops import sqrt_psd_matrix
from computations.spectrum import gap_OT, scan_gap_T


def test_gap_monotone_in_absT():
    """
    Dla U(T) ~ T^2 v^2, oczekujemy że μ(T) jest nienmalejące w |T|.
    (Dokładnie: μ powinno rosnąć wraz z |T|; numerycznie dopuszczamy mikrowahania.)
    """
    v, w = make_volume_grid(n=80)
    Theta = theta_matrix(v, w)

    # T tylko nieujemne (monotonia w |T|)
    T_vals = np.linspace(0.0, 1.2, 10)
    mus = scan_gap_T(Theta, v, m=1.0, T_vals=T_vals, scale=1.0)

    # różnice nieujemne (z małą tolerancją numeryczną)
    diffs = np.diff(mus)
    assert (diffs >= -1e-10).all(), f"μ(T) nie jest nienmalejąca: {diffs}"


def test_gap_closure_by_scaling():
    """
    Skaluje Θ -> s Θ i sprawdza, że dla s << 1 luka przy T=0 spada poniżej progu.
    """
    v, w = make_volume_grid(n=80)
    Theta = theta_matrix(v, w)

    T0 = 0.0
    scales = [1.0, 1e-3, 1e-6]
    mus = [gap_OT(Theta, v, m=1.0, T=T0, scale=s) for s in scales]

    # malejąca z s oraz bardzo mała dla s=1e-6
    assert mus[0] >= mus[1] >= mus[2] - 1e-12, f"μ nie maleje z scale: {mus}"
    assert mus[-1] < 1e-6, f"μ(scale=1e-6) za duża: {mus[-1]}"


def test_holder_half_under_operator_perturbation():
    """
    Kato–Heinz: || sqrt(A+Δ) - sqrt(A) || <= C ||Δ||^{1/2}.
    Testujemy wykładnik ~1/2 w log–log, perturbując operatorowo Δ = ε I.
    (To bezpieczny test Hölder-1/2 bez mieszania w parametr T.)
    """
    v, w = make_volume_grid(n=70)
    Theta = theta_matrix(v, w)

    # Bierzemy O(T) w pobliżu "małej luki" (np. T nieduże i/lub scale mniejsze)
    T = 0.2
    O_base = O_matrix(Theta, U_matrix(v, m=1.0, T=T))
    H_base = sqrt_psd_matrix(O_base)

    eps_list = np.array([1e-1, 5e-2, 2.5e-2, 1.25e-2])
    errs = []
    for eps in eps_list:
        O_pert = O_base + eps * np.eye(len(v))
        H_pert = sqrt_psd_matrix(O_pert)
        errs.append(norm(H_pert - H_base, 2))

    # dopasuj wykładnik nachylenia: err ~ const * eps^alpha
    alpha = np.polyfit(np.log(eps_list), np.log(errs), 1)[0]
    # oczekujemy okolic 0.5 (dopuszczamy pewną tolerancję numeryczną)
    assert 0.4 <= alpha <= 0.6, f"Spodziewany wykładnik ~0.5, wyszło {alpha:.3f}"
