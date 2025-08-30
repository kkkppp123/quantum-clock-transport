# %%
# =============================================================================
# 01_spectrum_gap.py  —  μ(T) + crossing-zero   [Etap G]
# =============================================================================
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

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


# --- katalog wyjściowy na rysunki --------------------------------------------------------------
FIG_DIR = Path("docs/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- próba użycia implementacji repozytoryjnej -------------------------------------------------
use_repo_impl = False
try:
    from computations.spectrum import scan_gap_T  # type: ignore

    def compute_mu_T_repo(T: np.ndarray) -> np.ndarray:
        """
        Oblicz minimalną lukę μ(T) przy użyciu funkcji z repozytorium.
        """
        out = scan_gap_T(T)
        if isinstance(out, dict) and "mu" in out:
            mu = np.asarray(out["mu"], dtype=float)
        else:
            mu = np.asarray(out, dtype=float)
        if mu.shape != T.shape:
            mu = mu.reshape(T.shape)
        return mu

    use_repo_impl = True
except Exception:
    use_repo_impl = False


# --- fallback: prosty Hamiltonian 2×2 z crossingiem w okolicy T≈0 -------------------------------
def H_gap_toy(T: float) -> np.ndarray:
    """
    Toy Hamiltonian 2×2:
        H(T) = [[Δ(T), g], [g, -Δ(T)]],
    gdzie Δ(T) ma część liniową i nieliniową, co prowadzi do crossing-zero.
    """
    Delta = 0.6 + 0.2 * np.tanh(2.0 * T) + 1.0 * T
    g = 0.35
    return np.array([[Delta, g], [g, -Delta]], dtype=float)


def compute_mu_T_toy(T: np.ndarray) -> np.ndarray:
    """
    Oblicz minimalną lukę μ(T) dla siatki T przy użyciu toy Hamiltonianu.
    """
    mu = np.empty_like(T, dtype=float)
    for i, t in enumerate(T):
        w, _ = eigh(H_gap_toy(float(t)))
        mu[i] = np.min(np.abs(w))
    return mu


# --- siatka T i obliczenia ---------------------------------------------------------------------
T = np.linspace(-1.5, 1.5, 601)
if use_repo_impl:
    mu = compute_mu_T_repo(T)
    source_label = "repo: computations.spectrum"
else:
    mu = compute_mu_T_toy(T)
    source_label = "toy 2×2 (fallback)"

# crossing-zero (minimum μ)
imin = int(np.argmin(mu))
T_min = float(T[imin])
mu_min = float(mu[imin])

# --- rysunek: główny + zoom --------------------------------------------------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel główny
ax0.plot(T, mu, lw=1.8)
ax0.set_xlabel("T")
ax0.set_ylabel("μ(T)")
ax0.set_title(f"Minimalna luka μ(T) — {source_label}")
ax0.grid(True, alpha=0.3)

ax0.axvline(T_min, lw=0.9, color="k", alpha=0.6)
ax0.annotate(
    f"min @ T≈{T_min:.3f}\nμ≈{mu_min:.2e}",
    xy=(T_min, mu_min),
    xytext=(T_min + 0.25, mu_min * 1.8 if mu_min > 0 else 0.02),
    arrowprops=dict(arrowstyle="->", lw=0.8),
)

# Panel zoom
window = 0.25
mask = (T >= T_min - window) & (T <= T_min + window)
ax1.plot(T[mask], mu[mask], lw=1.8)
ax1.set_xlabel("T (zoom)")
ax1.set_ylabel("μ(T)")
ax1.set_title("Powiększenie okolicy crossing-zero")
ax1.grid(True, alpha=0.3)
ax1.axvline(T_min, lw=0.9, color="k", alpha=0.6)

fig.suptitle("μ(T) i crossing-zero", y=1.03)
fig.tight_layout()

# --- zapis plików ------------------------------------------------------------------------------
for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"01_mu_T.{ext}", bbox_inches="tight", dpi=200)

safe_print("[OK] Zapisano:", FIG_DIR / "01_mu_T.png", "oraz 01_mu_T.pdf")
safe_print(f"[INFO] Źródło: {source_label}; min: T≈{T_min:.6f}, μ≈{mu_min:.3e}")
