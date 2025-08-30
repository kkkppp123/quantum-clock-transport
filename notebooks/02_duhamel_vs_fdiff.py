# %%
# =============================================================================
# 02_duhamel_vs_fdiff.py — Fréchet (Duhamel) vs różnice skończone dla sqrt
# =============================================================================
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
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


# --- katalog na rysunki ---
FIG_DIR = Path("docs/fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------- pomocnicze: hermityzacja, sqrt(PSD), Fréchet dla sqrt ----------
def her(A: np.ndarray) -> np.ndarray:
    """Hermityzacja macierzy."""
    A = np.asarray(A, dtype=np.complex128)
    return 0.5 * (A + A.conj().T)


def sqrtm_psd(X: np.ndarray) -> np.ndarray:
    """Macierzowy pierwiastek dla PSD z obcięciem ujemnych wartości własnych."""
    lam, V = eigh(her(X))
    lam = np.clip(lam, 0.0, None)
    return (V * np.sqrt(lam)) @ V.conj().T


def frechet_sqrt(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """
    L_f(X)[E] dla f(x)=sqrt(x), w bazie własnej X (podzielone różnice).
    Dla i==j: f'(λ_i) = 1/(2√λ_i); dla i!=j: (√λ_i−√λ_j)/(λ_i−λ_j).
    """
    Xh = her(X)
    lam, V = eigh(Xh)
    lam = np.clip(lam, 0.0, None)
    Ein = V.conj().T @ E @ V
    G = np.empty_like(Ein)
    for i in range(len(lam)):
        for j in range(len(lam)):
            if i == j:
                denom = max(lam[i], 1e-32)
                G[i, j] = (0.5 / np.sqrt(denom)) * Ein[i, j]
            else:
                denom = lam[i] - lam[j]
                num = np.sqrt(lam[i]) - np.sqrt(lam[j])
                G[i, j] = (num / denom) * Ein[i, j]
    return V @ G @ V.conj().T


# ---------- scenariusz testowy: X(T) = A + T B ----------
rng = np.random.default_rng(0)
d = 6
Q = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
B = her(Q)
Bw, _ = eigh(B)
B /= max(abs(Bw).max(), 1.0)
B *= 0.15
A = np.diag(np.linspace(0.8, 2.0, d)).astype(np.complex128)
T0 = 0.1
X0 = A + T0 * B
E = B  # dX/dT

# ---------- porównanie Frécheta vs różnice centralne ----------
hs = np.logspace(-8, -2, 13)
rel_err, scaled_err = [], []
D_true = frechet_sqrt(X0, E)

for h in hs:
    D_fd = (sqrtm_psd(A + (T0 + h) * B) - sqrtm_psd(A + (T0 - h) * B)) / (2.0 * h)
    e = norm(D_fd - D_true, "fro") / max(norm(D_true, "fro"), 1e-16)
    rel_err.append(e)
    scaled_err.append(e / (h**2))

rel_err = np.asarray(rel_err)
scaled_err = np.asarray(scaled_err)

# nachylenie w reżimie asymptotycznym
lo, hi = 2, -2 if len(hs) > 6 else len(hs)
slope = np.polyfit(np.log(hs[lo:hi]), np.log(rel_err[lo:hi]), 1)[0]

# ---------- wykresy ----------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2))

ax0.loglog(hs, rel_err, marker="o", lw=1.8, label=f"błąd względny (slope≈{slope:.2f})")
ax0.loglog(
    hs,
    (rel_err[lo] / (hs[lo] ** 2)) * (hs**2),
    ls="--",
    lw=1.0,
    label="~h² (odniesienie)",
)
ax0.set_xlabel("krok h")
ax0.set_ylabel("‖D_fd − D_Fréchet‖_F / ‖D_Fréchet‖_F")
ax0.set_title("Pochodna √X — Fréchet vs różnice centralne")
ax0.grid(True, which="both", alpha=0.3)
ax0.legend()

ax1.semilogx(hs, scaled_err, marker="s", lw=1.8)
ax1.set_xlabel("krok h")
ax1.set_ylabel("błąd / h²")
ax1.set_title("Reżim O(h²) centralnej różnicy")
ax1.grid(True, which="both", alpha=0.3)

fig.suptitle("Duhamel/Fréchet vs różnice skończone dla √X", y=1.03)
fig.tight_layout()

for ext in ("png", "pdf"):
    fig.savefig(FIG_DIR / f"02_duhamel_vs_fdiff.{ext}", bbox_inches="tight", dpi=200)

safe_print("[OK] Zapisano:", FIG_DIR / "02_duhamel_vs_fdiff.png", "oraz .pdf")
safe_print(
    f"[INFO] Szacowany wykładnik zbieżności (obszar środkowy): slope≈{slope:.3f}"
)
