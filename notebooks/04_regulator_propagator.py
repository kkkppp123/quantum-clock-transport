# %%
# =============================================================================
# 04_regulator_propagator.py  —  porównanie propagatorów/regulatorów (Etap G)
#
# Cel:
#   Zobrazować stabilność unitarności i zbieżność trajektorii dla różnych
#   wariantów generatora w okolicy crossing-zero:
#     • raw        — goły H(T)
#     • bounded    — H_α = H (I + (H/α)^2)^(-1/2)
#     • dbsqrt     — wariant „DB-sqrt” (tu równoważny bounded: stabilny, ograniczony)
#     • epsI       — prosty regulator przesunięcia spektrum H + ε I (dla podglądu)
#
# Metoda:
#   Dyskretyzacja w T i kroki Crank–Nicolson (unitarne numerycznie).
#   Panel 1:  ‖U†U − I‖ w funkcji kroku (skala log).
#   Panel 2:  ‖ΔU‖₂ dla propagatorów końcowych (T_end) — różnice wariantów.
#
# Wyjście (ryciny do LaTeX):
#   docs/fig/04_regulator_compare.png
#   docs/fig/04_regulator_compare.pdf
#
# Jak uruchomić:
#   python notebooks/04_regulator_propagator.py
# =============================================================================
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm, svd
from scipy.linalg import eigh, solve

# --- erzatz print odporny na Windows-1250 ---
os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8")
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

# ---------- pomocnicze: hermityzacja, krok CN, regulacje ----------


def her(A: np.ndarray) -> np.ndarray:
    """Hermityzacja macierzy (część hermitowska)."""
    A = np.asarray(A, dtype=np.complex128)
    return 0.5 * (A + A.conj().T)


def cn_step(H: np.ndarray, dT: float) -> np.ndarray:
    """Pojedynczy krok Crank–Nicolson: U = (Id - i dT/2 H)^{-1} (Id + i dT/2 H)."""
    Hh = her(H)
    n = Hh.shape[0]
    Id = np.eye(n, dtype=np.complex128)
    A = Id - 1j * (dT / 2.0) * Hh
    B = Id + 1j * (dT / 2.0) * Hh
    return solve(A, B, assume_a="gen")


def bounded_generator(H: np.ndarray, alpha: float) -> np.ndarray:
    """H_α = H (I + (H/α)^2)^(-1/2) przez diagonalizację (stabilny, ograniczony)."""
    Hh = her(H)
    w, V = eigh(Hh)
    s = w / np.sqrt(1.0 + (w / alpha) ** 2)
    return (V * s) @ V.conj().T


def dbsqrt_generator(H: np.ndarray, alpha: float) -> np.ndarray:
    """„DB-sqrt” – tutaj równoważny 'bounded' (bez ryzyka niestabilności)."""
    return bounded_generator(H, alpha=alpha)


# ---------- model bazowy H(T) (crossing-zero + sprzężenie) ----------


def H_base(T: float) -> np.ndarray:
    """Prosty model 2×2 z zamykającą się szczeliną przy T=0."""
    Delta = 2.5 * T
    g = 0.25
    return np.array([[Delta, g], [g, -Delta]], dtype=np.complex128)


# ---------- ewolucja i pomiar błędów ----------


def evolve(
    H_of_T,
    Tgrid: np.ndarray,
    variant: str = "raw",
    alpha: float = 12.0,
    eps_shift: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ewoluuje po siatce Tgrid i zwraca:
      (U_total, norms), gdzie norms[k] = ‖U_k† U_k − Id‖₂ po k-tym kroku.
    """
    U = np.eye(2, dtype=np.complex128)
    norms: list[float] = []

    for k in range(len(Tgrid) - 1):
        t = float(Tgrid[k])
        dT = float(Tgrid[k + 1] - Tgrid[k])

        H = H_of_T(t)
        if variant == "raw":
            Hk = H
        elif variant == "bounded":
            Hk = bounded_generator(H, alpha=alpha)
        elif variant == "dbsqrt":
            Hk = dbsqrt_generator(H, alpha=alpha)
        elif variant == "epsI":
            Hk = H + (eps_shift * np.eye(H.shape[0], dtype=np.complex128))
        else:
            raise ValueError(f"unknown variant: {variant}")

        U_step = cn_step(Hk, dT)
        U = U_step @ U

        Id = np.eye(U.shape[0], dtype=np.complex128)
        norms.append(norm(U.conj().T @ U - Id, 2))

    return U, np.asarray(norms, float)


def spec_norm(A: np.ndarray) -> float:
    """Norma spektralna (największa wartość osobliwa)."""
    return float(svd(A, compute_uv=False)[0])


# ---------- symulacja ----------

T = np.linspace(-1.0, 1.0, 801)

U_raw, norms_raw = evolve(H_base, T, "raw")
U_bnd, norms_bnd = evolve(H_base, T, "bounded", alpha=12.0)
U_db, norms_db = evolve(H_base, T, "dbsqrt", alpha=12.0)
U_eps, norms_eps = evolve(H_base, T, "epsI", eps_shift=1e-2)

# Różnice propagatorów końcowych
d_raw_bnd = spec_norm(U_raw - U_bnd)
d_raw_db = spec_norm(U_raw - U_db)
d_bnd_db = spec_norm(U_bnd - U_db)
d_raw_eps = spec_norm(U_raw - U_eps)

# ---------- wykresy ----------

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

# (1) Błąd unitarności w czasie
ax[0].plot(norms_raw, lw=1.6, label="raw H")
ax[0].plot(norms_bnd, lw=1.6, label="bounded Hα")
ax[0].plot(norms_db, lw=1.6, label="DB-sqrt α")
ax[0].plot(norms_eps, lw=1.6, label="εI")
ax[0].set_yscale("log")
ax[0].set_xlabel("krok")
ax[0].set_ylabel("‖U†U − I‖₂")
ax[0].set_title("Błąd unitarności w trakcie ewolucji (CN)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

# (2) Różnice propagatorów (T_end)
labels = ["raw vs bnd", "raw vs db", "bnd vs db", "raw vs εI"]
ax[1].bar(labels, [d_raw_bnd, d_raw_db, d_bnd_db, d_raw_eps])
ax[1].set_ylabel("‖ΔU‖₂")
ax[1].set_title("Różnice propagatorów w T_end")
ax[1].grid(True, axis="y", alpha=0.3)

fig.suptitle("Regulatory a propagator (Crank–Nicolson) — crossing-zero", y=1.03)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"04_regulator_compare.{ext}", bbox_inches="tight", dpi=200)

safe_print("[OK] Zapisano:", FIG_DIR / "04_regulator_compare.png", "oraz .pdf")
safe_print(
    "[INFO] ΔU (‖·‖₂): "
    f"raw-bnd={d_raw_bnd:.3e}, raw-db={d_raw_db:.3e}, bnd-db={d_bnd_db:.3e}, raw-εI={d_raw_eps:.3e}"
)
safe_print(
    "[INFO] Unitarność (mediana): "
    f"raw={np.median(norms_raw):.1e}, bnd={np.median(norms_bnd):.1e}, "
    f"db={np.median(norms_db):.1e}, epsI={np.median(norms_eps):.1e}"
)
