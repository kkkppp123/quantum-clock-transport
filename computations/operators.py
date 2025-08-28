# computations/operators.py
import numpy as np


def make_volume_grid(v_min: float = 1e-2, v_max: float = 50.0, n: int = 200):
    """Jednorodna siatka objętości v oraz dodatnia waga w ~ v^{2/3}."""
    v = np.linspace(v_min, v_max, n)
    w = v ** (2.0 / 3.0)
    return v, w


def theta_matrix(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Dyskretny operator typu -d/dv( w d/dv ) z warunkami Dirichleta (0 na brzegach).
    Zwraca macierz symetryczną, dodatnio półokreśloną.
    """
    n = len(v)
    H = np.zeros((n, n), dtype=float)
    h = float(v[1] - v[0])  # zakładamy równomierną siatkę

    # w_{i+1/2} średnia na krawędzi
    wp = (w[1:] + w[:-1]) / 2.0

    # Wewnętrzne węzły i=1..n-2
    for i in range(1, n - 1):
        H[i, i - 1] = -wp[i - 1] / h**2
        H[i, i] = (wp[i - 1] + wp[i]) / h**2
        H[i, i + 1] = -wp[i] / h**2

    # Warunki brzegowe Dirichleta: f(0)=f(N-1)=0 (ustaw 1 na diag. krańcach)
    H[0, 0] = 1.0
    H[-1, -1] = 1.0
    return H


def U_matrix(v: np.ndarray, m: float, T: float) -> np.ndarray:
    """Operator mnożenia przez m^2 T^2 v^2 (diag)."""
    return (m**2) * (T**2) * np.diag(v**2)


def O_matrix(theta: np.ndarray, U: np.ndarray) -> np.ndarray:
    return theta + U
