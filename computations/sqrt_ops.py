import numpy as np
from numpy.linalg import eigh


def sqrt_psd_matrix(A: np.ndarray) -> np.ndarray:
    """
    Zwraca macierzowy pierwiastek z dodatnio półokreślonej macierzy A (PSD).
    Implementacja przez diagonalizację hermitowską: A = V diag(w) V^T,
    następnie sqrt(A) = V diag(sqrt(w_+)) V^T, gdzie w_+ = max(w, 0).
    """
    # bezpieczeństwo numeryczne: ucinamy ewentualne ujemne wartości własne ~1e-15
    w, V = eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def bounded_generator(H: np.ndarray, alpha: float) -> np.ndarray:
    """
    Zwraca ograniczony generator:
        H_alpha = H (I + H^2/alpha^2)^(-1/2)
    poprzez diagonalizację hermitowską H.
    Dla alpha -> ∞ mamy H_alpha -> H w sensie silnym.
    """
    w, V = eigh(H)
    damp = w / np.sqrt(1.0 + (w / alpha) ** 2)
    return (V * damp) @ V.T


def holder_half_norm_diff(A: np.ndarray, B: np.ndarray) -> float:
    """
    Oblicza ‖sqrt(A) - sqrt(B)‖_2 (norma spektralna) dla PSD A, B.
    Dla macierzy symetrycznych norma 2 = max(|wartości własne|).
    """
    SA = sqrt_psd_matrix(A)
    SB = sqrt_psd_matrix(B)
    w, _ = eigh(SA - SB)
    return float(np.max(np.abs(w)))
