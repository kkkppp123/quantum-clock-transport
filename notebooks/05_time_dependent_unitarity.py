# %%
# =============================================================================
# 05_time_dependent_unitarity.py
#  — unitarność ewolucji zależnej od czasu (CN vs Magnus-1) + Hölder 1/2 (Etap G)
#
# Cel:
#   1) Zmierzyć błąd unitarności ‖U†U−I‖₂ dla propagatora porządkowanego w czasie
#      przy dwóch schematach krokowych:
#        • Crank–Nicolson ("cn") – praktycznie unitarny dla hermitowskich H,
#        • Magnus-1 ("magnus1")  – U_k = exp(-i H(T_k) ΔT).
#      Porównujemy błędy w zależności od liczby kroków N (siatki T).
#   2) Pokazać sygnaturę Höldera ~1/2 w okolicach crossing-zero na przykładzie
#      H(T) o zachowaniu ~sqrt(|T|) → ||H(T+δ)−H(T)||₂ ~ δ^{1/2}.
#
# Wejście:
#   Preferujemy implementację repo:
#     computations.propagator.time_ordered_propagator
#     computations.propagator.unitarity_error
#   W razie braku – fallback samowystarczalny (CN + expm).
#
# Wyjście (ryciny do LaTeX):
#   docs/fig/05_time_dependent_unitarity.png
#   docs/fig/05_time_dependent_unitarity.pdf
#
# Jak uruchomić:
#   python notebooks/05_time_dependent_unitarity.py
# =============================================================================
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# --- erzatz print odporny na Windows-1250 ---
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")  # py3.7+
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


def safe_print(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    try:
        print(s, **kwargs)
    except UnicodeEncodeError:
        print(s.encode("utf-8", "ignore").decode("ascii", "ignore"), **kwargs)


# --- katalog wyjściowy ---
FIG_DIR = Path("docs/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# 1) Preferowany import z repo; w razie czego fallback (samowystarczalny)
# -----------------------------------------------------------------------------
try:
    from computations.propagator import (  # type: ignore
        time_ordered_propagator,
        unitarity_error,
    )

    impl = "repo"
except Exception:
    impl = "fallback"
    from numpy.linalg import norm
    from scipy.linalg import expm, solve

    def _her(A: np.ndarray) -> np.ndarray:
        A = np.asarray(A, dtype=np.complex128)
        return 0.5 * (A + A.conj().T)

    def _cn_step(H: np.ndarray, dT: float) -> np.ndarray:
        Hh = _her(H)
        eye = np.eye(Hh.shape[0], dtype=np.complex128)
        A = eye - 1j * (dT / 2.0) * Hh
        B = eye + 1j * (dT / 2.0) * Hh
        return solve(A, B, assume_a="gen")

    def time_ordered_propagator(H_of_T, T_grid, scheme="cn"):
        """Prosty GS dla T-zależnego H: CN lub Magnus-1 (expm)."""
        U = np.eye(H_of_T(float(T_grid[0])).shape[0], dtype=np.complex128)
        for k in range(len(T_grid) - 1):
            t = float(T_grid[k])
            dT = float(T_grid[k + 1] - T_grid[k])
            H = H_of_T(t)
            if scheme == "cn":
                U_step = _cn_step(H, dT)
            elif scheme == "magnus1":
                U_step = expm(-1j * _her(H) * dT)
            else:
                raise ValueError("scheme must be 'cn' or 'magnus1'")
            U = U_step @ U
        return U

    def unitarity_error(U: np.ndarray, ord: int | float = 2) -> float:
        Uc = np.asarray(U, dtype=np.complex128)
        eye = np.eye(Uc.shape[0], dtype=np.complex128)
        return float(norm(Uc.conj().T @ Uc - eye, ord=ord))


# -----------------------------------------------------------------------------
# 2) Model H(T) z crossing-zero (ze sprzężeniem „g”)
# -----------------------------------------------------------------------------
def H_cross(T: float) -> np.ndarray:
    Delta = 3.0 * T
    g = 0.30
    return np.array([[Delta, g], [g, -Delta]], dtype=np.complex128)


# -----------------------------------------------------------------------------
# 3) Unitarność vs liczba kroków N (mniejszy krok → mniejszy błąd)
# -----------------------------------------------------------------------------
N_list = np.array([50, 100, 200, 400, 800, 1600], dtype=int)
errs_cn: list[float] = []
errs_m1: list[float] = []

for N in N_list:
    T = np.linspace(-1.0, 1.0, int(N))
    Ucn = time_ordered_propagator(H_cross, T, scheme="cn")
    Um1 = time_ordered_propagator(H_cross, T, scheme="magnus1")
    errs_cn.append(unitarity_error(Ucn, ord=2))
    errs_m1.append(unitarity_error(Um1, ord=2))

errs_cn = np.asarray(errs_cn)
errs_m1 = np.asarray(errs_m1)


# -----------------------------------------------------------------------------
# 4) Sygnatura Höldera 1/2: ||H(T+δ)−H(T)||₂ ~ δ^{1/2}
# -----------------------------------------------------------------------------
def H_sqrt(T: float) -> np.ndarray:
    s = np.sqrt(abs(T))
    return np.array([[s, 0.0], [0.0, -s]], dtype=np.complex128)


deltas = np.logspace(-6, -2, 9)
diffs = np.array([np.linalg.norm(H_sqrt(d) - H_sqrt(0.0), 2) for d in deltas])

# policz nachylenie na środkowych punktach (pomijamy 2 skrajne z każdej strony)
lo = 2
hi = -2 if deltas.size > 6 else deltas.size
slope = np.polyfit(np.log(deltas[lo:hi]), np.log(diffs[lo:hi]), 1)[0]

# -----------------------------------------------------------------------------
# 5) Wykresy i zapis
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (a) Unitarność vs liczba kroków
ax[0].loglog(N_list, errs_cn, marker="o", lw=1.8, label="CN")
ax[0].loglog(N_list, errs_m1, marker="s", lw=1.8, label="Magnus-1")
ax[0].invert_xaxis()  # więcej kroków → mniejszy błąd (intuicyjnie „w prawo”)
ax[0].set_xlabel("liczba kroków N (więcej → drobniejsza siatka)")
ax[0].set_ylabel("‖U†U − I‖₂")
ax[0].set_title(
    f"Unitarność vs rozmiar kroku ({'repo' if impl=='repo' else 'fallback'})"
)
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

# (b) Hölder ~1/2
ax[1].loglog(deltas, diffs, marker="o", lw=1.8)
ax[1].set_xlabel("δ")
ax[1].set_ylabel("‖H(δ) − H(0)‖₂")
ax[1].set_title(f"Hölder ~1/2 w okolicy 0 (slope≈{slope:.2f})")
ax[1].grid(True, which="both", alpha=0.3)

fig.suptitle("Time-dependent unitarity + Hölder 1/2", y=1.03)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(
        FIG_DIR / f"05_time_dependent_unitarity.{ext}", bbox_inches="tight", dpi=200
    )

safe_print("[OK] Zapisano:", FIG_DIR / "05_time_dependent_unitarity.png", "oraz .pdf")
safe_print(
    f"[INFO] Unitarność: CN median={np.median(errs_cn):.2e}, "
    f"Magnus-1 median={np.median(errs_m1):.2e} | źródło: {impl}"
)
safe_print(f"[INFO] Hölder slope≈{slope:.3f} (oczekiwane ~0.5)")
