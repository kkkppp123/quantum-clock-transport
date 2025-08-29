# computations/curvature_scaling.py
from __future__ import annotations
import numpy as np

from .operators import make_volume_grid, theta_matrix
from .curvature import curvature_norm

try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    _HAVE_MPL = False


def fit_log_slope(eps_list, vals):
    x = np.log(eps_list)
    y = np.log(np.maximum(vals, 1e-30))
    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def measure_series(ordering: str, n: int = 160, phi: float = 0.3, TY: float = 0.4):
    v, w = make_volume_grid(n=n)
    Theta = theta_matrix(v, w)

    eps_list = np.array([2e-1, 1e-1, 5e-2, 2.5e-2, 1.25e-2])
    vals = [
        curvature_norm(Theta, v, 1.0, phi=phi, TY=TY, eps=eps, ordering=ordering)
        for eps in eps_list
    ]
    slope, intercept = fit_log_slope(eps_list, vals)
    return eps_list, vals, slope, intercept


def print_table(ordering: str, n: int = 160):
    eps, vals, slope, _ = measure_series(ordering, n=n)
    print(f"Ordering = {ordering} (n={n})")
    for e, v in zip(eps, vals):
        print(f"  eps={e:8.5f}   ||F||={v:.6e}")
    print(f"  slope ~ {slope:.4f}")
    return slope


def plot_loglog(out_png: str | None = None, n: int = 160):
    ords = ["weyl", "born--jordan"]
    data = {}
    for ordg in ords:
        eps, vals, slope, _ = measure_series(ordg, n=n)
        data[ordg] = (eps, vals, slope)

    if not _HAVE_MPL:
        print("matplotlib not available — skipping plot")
        for ordg in ords:
            _, _, slope = data[ordg]
            print(f"{ordg:12s}: slope ~ {slope:.4f}")
        return

    plt.figure()
    for ordg, (eps, vals, slope) in data.items():
        plt.plot(np.log(eps), np.log(vals), "o-", label=f"{ordg} (slope~{slope:.2f})")
    plt.xlabel("log ε")
    plt.ylabel("log ||F||")
    plt.legend()
    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=160)
        print(f"Saved plot to {out_png}")
    else:
        plt.show()


if __name__ == "__main__":
    # Tabela
    print_table("weyl", n=160)
    print_table("born--jordan", n=160)
    # Wykres
    plot_loglog(out_png="curvature_scaling.png", n=160)
