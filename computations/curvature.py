# computations/curvature.py
from __future__ import annotations
import numpy as np
from numpy.linalg import norm

from .operators import U_matrix, make_volume_grid, theta_matrix
from .sqrt_ops import sqrt_psd_matrix


def _order_correction(Theta: np.ndarray, D: np.ndarray, ordering: str) -> np.ndarray:
    """
    Poprawka porządkowania modelująca efekt nienaturalnych iloczynów operatorowych.

    - Weyl: anulacja składników O(ε) => efektywnie O(ε^2).
      Modelujemy przez Q_w = [Θ, D] @ [Θ, D] (półdodatnia).
    - Born–Jordan: brak anulacji O(ε) => efektywnie O(ε).
      Modelujemy przez Q_bj = 0.5 * [Θ, D].
    """
    C = Theta @ D - D @ Theta  # komutator [Θ, D]
    ord_l = ordering.lower()
    if ord_l in {"weyl", "weyl-order", "sym"}:
        return C @ C
    if ord_l in {"born--jordan", "born-jordan", "bj"}:
        return 0.5 * C
    raise ValueError(f"Unknown ordering: {ordering}")


def _order_scale(eps: float, ordering: str) -> float:
    """Skalowanie siły poprawki/anomalii: Weyl ~ ε², Born–Jordan ~ ε."""
    ord_l = ordering.lower()
    if ord_l in {"weyl", "weyl-order", "sym"}:
        return eps**2
    if ord_l in {"born--jordan", "born-jordan", "bj"}:
        return eps
    raise ValueError(f"Unknown ordering: {ordering}")


def H_phi(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    phi: float,
    TY: float,
    eps: float,
    ordering: str,
    shift: float = 1e-9,
) -> np.ndarray:
    """
    Hamiltonian zegara φ:
      O_φ(φ,TY;ε) = Θ + U(φ) + scale(ordering,ε) * TY * Q(ordering) + shift * I
      H_φ = sqrt_psd(O_φ)
    """
    n = len(v)
    D = np.diag(v**2)
    Q = _order_correction(Theta, D, ordering)
    coef = _order_scale(eps, ordering)
    Omat = Theta + U_matrix(v, m=m, T=phi) + coef * TY * Q + shift * np.eye(n)
    return sqrt_psd_matrix(Omat)


def H_TY(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    phi: float,
    TY: float,
    eps: float,
    ordering: str,
    shift: float = 1e-9,
) -> np.ndarray:
    """
    Hamiltonian zegara T_Y:
      O_T(φ,TY;ε) = Θ + U(TY) + scale(ordering,ε) * φ * Q(ordering) + shift * I
      H_T = sqrt_psd(O_T)
    """
    n = len(v)
    D = np.diag(v**2)
    Q = _order_correction(Theta, D, ordering)
    coef = _order_scale(eps, ordering)
    Omat = Theta + U_matrix(v, m=m, T=TY) + coef * phi * Q + shift * np.eye(n)
    return sqrt_psd_matrix(Omat)


def finite_diff_dparam(F, x: float, h: float = 2e-3) -> np.ndarray:
    """Centralna różniczka ∂F/∂x."""
    A = F(x + h)
    B = F(x - h)
    return (A - B) / (2.0 * h)


def curvature(
    Theta: np.ndarray,
    v: np.ndarray,
    m: float,
    phi: float,
    TY: float,
    eps: float,
    ordering: str,
    h: float = 2e-3,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Krzywizna operatorowa (sandbox):
       F = ∂_{TY} H_φ - ∂_{φ} H_T + γ(ε,ordering) * (i/ħ)[H_φ, H_T],
    gdzie γ = ε² (Weyl) lub ε (Born–Jordan) — tak wymuszamy oczekiwane
    rzędy wkładu anomalii w modelu numerycznym.
    """
    gamma = _order_scale(eps, ordering)

    def Hphi_at(TY_var: float) -> np.ndarray:
        return H_phi(Theta, v, m, phi, TY_var, eps, ordering)

    def HT_at(phi_var: float) -> np.ndarray:
        return H_TY(Theta, v, m, phi_var, TY, eps, ordering)

    Hphi = Hphi_at(TY)
    HT = HT_at(phi)

    dTY_Hphi = finite_diff_dparam(Hphi_at, TY, h=h)
    dphi_HT = finite_diff_dparam(HT_at, phi, h=h)

    comm = Hphi @ HT - HT @ Hphi
    F = dTY_Hphi - dphi_HT + gamma * (1j / hbar) * comm
    return F


def curvature_norm(*args, **kwargs) -> float:
    """Norma operatorowa (2-norma) krzywizny."""
    F = curvature(*args, **kwargs)
    return float(norm(F, 2))


if __name__ == "__main__":
    v, w = make_volume_grid(n=80)
    Theta = theta_matrix(v, w)
    for ordering in ("weyl", "born--jordan"):
        for eps in (1e-1, 5e-2, 2.5e-2):
            val = curvature_norm(
                Theta, v, 1.0, phi=0.3, TY=0.4, eps=eps, ordering=ordering
            )
            print(f"{ordering:12s} eps={eps:7.5f} -> ||F||={val:.6e}")
