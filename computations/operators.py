import numpy as np


def make_volume_grid(
    n: int = 100, vmin: float = 0.0, vmax: float = 8.0, k: float = 1.0
):
    """
    Siatka objętości v i wagi w ~ v^k dv (trapezowo).
    Zwraca:
        v:  (n,)  — węzły
        w:  (n,)  — wagi dodatnie
    """
    v = np.linspace(vmin, vmax, n)
    dv = v[1] - v[0]
    w = v**k
    # trapez: połowa wagi na końcach
    if n >= 1:
        w[0] *= 0.5
        w[-1] *= 0.5
    w *= dv
    # gwarancja dodatniości (jeśli vmin=0 i k>0, pierwszy węzeł może być 0)
    w = np.maximum(w, 1e-16)
    return v, w


def theta_matrix(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Symetryczna wersja operatora 'Θ' jako ważony laplasjan:
        T — trójdiagonalna macierz z warunkami Dirichleta:
             T[0,0] = T[-1,-1] = 1, a dla 1..n-2: diag=2, pod/naddiagonale = -1
        Θ = W^{1/2} T W^{1/2}, gdzie W = diag(w)

    Dzięki temu Θ jest hermitowska i dodatnio półokreślona.
    """
    n = len(v)
    T = np.zeros((n, n))
    if n == 1:
        T[0, 0] = 1.0
    else:
        T[0, 0] = 1.0
        T[-1, -1] = 1.0
        for i in range(1, n - 1):
            T[i, i] = 2.0
            T[i, i - 1] = -1.0
            T[i - 1, i] = -1.0
    Wsqrt = np.diag(np.sqrt(w))
    Theta = Wsqrt @ T @ Wsqrt
    return Theta


def U_matrix(v: np.ndarray, m: float = 1.0, T: float = 1.0) -> np.ndarray:
    """
    Potencjał diagonalny: U(T) = m^2 T^2 v^2 (diag).
    """
    return np.diag((m * T * v) ** 2)


def O_matrix(Theta: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Operator całkowity:  O_T = Θ + U(T).
    Hermitowski i PSD, jeśli Θ, U są hermitowskie oraz U >= 0.
    """
    return Theta + U
