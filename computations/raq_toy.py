# computations/raq_toy.py
# v1.0 (Etap F – RAQ-toy)
# Minimalny „group averaging” dla G = (R, +) z regulatorami:
#  - Gaussian (analityczny): P_sigma = exp( - (sigma^2/2) * C^2 )
#  - Box (sinc):            P_L     = sinc(L * C)  = sin(L C)/(L C)
# oraz "dokładny" projektor spektralny na ker(C) do weryfikacji.
#
# Uwaga: to zabawka – dla sigma >> 0 (lub L >> 1) dostajemy prawie-projektor:
# ||P^2 - P|| jest małe, a P ~ P_exact na podprzestrzeni fizycznej.

from __future__ import annotations
from typing import Callable, Tuple
import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm
from scipy.linalg import eigh

Array = NDArray[np.complex128]


# ---------- pomocnicze ----------


def _hermitize(A: Array) -> Array:
    A = np.asarray(A, dtype=np.complex128)
    return 0.5 * (A + A.conj().T)


def _spectral_function_of(C: Array, f: Callable[[np.ndarray], np.ndarray]) -> Array:
    """Zwraca f(C) dla hermitowskiego C via rachunek spektralny."""
    C = _hermitize(C)
    vals, vecs = eigh(C)
    diag = f(vals.astype(np.float64))
    return (vecs * diag) @ vecs.conj().T


def projector_error(P: Array, ord: int | float = 2) -> float:
    """||P^2 - P|| (domyślnie norma spektralna)."""
    P = np.asarray(P, dtype=np.complex128)
    return float(norm(P @ P - P, ord=ord))


def random_unitary(d: int, seed: int = 42) -> Array:
    """Losowa jednostkowa (Haar przez QR z korekcją faz)."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    diagR = np.diag(R)
    phases = np.ones(d, dtype=np.complex128)
    nz = np.abs(diagR) > 1e-15
    phases[nz] = diagR[nz] / np.abs(diagR[nz])
    return Q @ np.diag(phases.conj())


# ---------- RAQ: projektory regularyzowane ----------


def gaussian_raq_projector(C: Array, sigma: float = 40.0) -> Array:
    """
    P_sigma = exp( - (sigma^2/2) * C^2 )
    - Dodatnia, hermitowska „filtracja ciepła” – dla sigma >> 1 zbiega do projektora na ker(C).
    - Szybka i stabilna (analityczna).
    """
    s2 = 0.5 * (sigma**2)

    def f(lam: np.ndarray) -> np.ndarray:
        return np.exp(-s2 * (lam**2))

    P = _spectral_function_of(C, f)
    return _hermitize(P)


def box_raq_projector(C: Array, L: float = 80.0) -> Array:
    """
    P_L = sinc(L C) = sin(L C) / (L C)  (operacyjnie przez rachunek spektralny).
    - To uśrednianie po oknie pudełkowym [-L, L] (Fejér/Dirichlet-like).
    - Nie gwarantuje dodatniości, ale działa jako przybliżenie δ(C).
    """
    # numpy.sinc(x) = sin(pi x)/(pi x), więc sin(L x)/(L x) = sinc((L/π) x)
    scale = L / np.pi

    def f(lam: np.ndarray) -> np.ndarray:
        x = scale * lam
        return np.sinc(x)

    P = _spectral_function_of(C, f)
    return _hermitize(P)


def physical_spectral_projector(C: Array, tol: float = 1e-12) -> Array:
    """
    Dokładny projektor spektralny na ker(C): suma |v_i><v_i| dla |λ_i| <= tol.
    Jeśli jądro jest puste (w tej zabawce nie jest), zwraca macierz zerową.
    """
    C = _hermitize(C)
    vals, vecs = eigh(C)
    mask = np.abs(vals) <= tol
    if not np.any(mask):
        return np.zeros_like(C, dtype=np.complex128)
    V = vecs[:, mask]
    return V @ V.conj().T


def dirac_observable_from_constraint(C: Array) -> Array:
    """
    Budujemy obserwablę Diraca F = g(C), która z definicji komutuje z C.
    Prosty wybór: g(λ) = λ^2 + 2.0 (dodatnia i niesingularna).
    """

    def g(lam: np.ndarray) -> np.ndarray:
        return lam**2 + 2.0

    return _spectral_function_of(C, g)


# ---------- wygodny "toy" budulec ----------


def build_toy_constraint(seed: int = 0) -> Tuple[Array, np.ndarray]:
    """
    Zwraca (C, true_eigs) – hermitowski ogranicznik z widmem:
      [-2.0, -1.0, 0.0, 0.0, 0.1, 0.5]  w losowej bazie.
    Dwa ścisłe zera zapewniają niepustą przestrzeń fizyczną.
    """
    spec = np.array([-2.0, -1.0, 0.0, 0.0, 0.1, 0.5], dtype=float)
    U = random_unitary(spec.size, seed=seed)
    C = U @ np.diag(spec.astype(np.complex128)) @ U.conj().T
    return _hermitize(C), spec
