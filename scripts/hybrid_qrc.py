"""Experiment: HARX and the quantum reservoir in one model.

Two ways to combine them, both in log space with an intercept, rolling window of WIN months,
245 out-of-sample months, exact (noiseless) QR1 features:

  augmented  one ridge readout on HARX's 13 inputs plus QR1's 10 <Z> values (23 weights + intercept).
             Ridge penalty on the 10 reservoir columns only, chosen on the 120 months before the
             test period from a fixed grid; the HARX columns stay unpenalised as in HARX itself.
  residual   HARX fitted first on the window; a ridge readout on the 10 <Z> values predicts HARX's
             in-window residuals; the forecast is the sum.

Each is scored on 20 coupling seeds against HARX alone, with a Diebold-Mariano test (HAC, 3 lags)
on the seed-averaged forecast. Writes data/qr1_hybrid.csv.

With --noise the augmented hybrid is re-scored on the cached noisy outcome probabilities from
``sweep_qrc.py simulate`` (2 Trotter steps; no circuits run here): readout trained and tested on
features sampled under the same gate noise and shot count. Writes data/qr1_hybrid_noise.csv.

Usage: .venv/bin/python scripts/hybrid_qrc.py [--seeds 20] [--noise] [--reps 5]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from math import erf, sqrt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.harness import WIN, iter_folds, load_panel
from src.qrc.reference import couplings, evolution, reservoir_features
from sweep_qrc import N_VALIDATION, RIDGES, cache_path, load_inputs, sample

Z_COLS = [f"z{q}" for q in range(10)]


def ridge_fit(X: np.ndarray, y: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """Least squares with a per-column penalty (0 = unpenalised). Column 0 is the intercept."""
    return np.linalg.solve(X.T @ X + np.diag(penalty), X.T @ y)


def rolling(F: np.ndarray, y: np.ndarray, positions: np.ndarray, penalty: np.ndarray,
            base_cols: slice | None = None) -> np.ndarray:
    """One forecast per position. With ``base_cols``, HARX on those columns is fitted first and the
    penalised readout on the remaining columns predicts its residual (the residual protocol)."""
    out = np.empty(len(positions))
    for i, j in enumerate(positions):
        lo = max(0, j - WIN)
        A, b, x = F[lo:j], y[lo:j], F[j]
        if base_cols is None:
            out[i] = x @ ridge_fit(A, b, penalty)
        else:
            beta = np.linalg.lstsq(A[:, base_cols], b, rcond=None)[0]
            rest = np.r_[0, np.arange(base_cols.stop, F.shape[1])]
            gamma = ridge_fit(A[:, rest], b - A[:, base_cols] @ beta, penalty[rest])
            out[i] = x[base_cols] @ beta + x[rest] @ gamma
    return out


def dm_hac(e1: np.ndarray, e2: np.ndarray, lag: int = 3) -> tuple[float, float]:
    d = e1**2 - e2**2
    x = d - d.mean()
    n = len(d)
    v = x @ x / n + 2 * sum((1 - k / (lag + 1)) * (x[k:] @ x[:-k]) / n for k in range(1, lag + 1))
    t = d.mean() / np.sqrt(v / n)
    return float(t), float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def noise_check(reps: int) -> None:
    panel, _, _ = load_inputs()
    full, feature_sets, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y = full["y"].to_numpy()
    H = np.column_stack([np.ones(len(full)), full[feature_sets["HARX"]].to_numpy()])
    n_h = H.shape[1]
    val = np.arange(oos[0] - N_VALIDATION, oos[0])
    mse = lambda p: float(np.mean((p - y[oos]) ** 2))
    harx = mse(rolling(H, y, oos, np.zeros(n_h)))
    pen = lambda k: np.r_[np.zeros(n_h), np.full(10, k)]
    records = []
    for seed in range(3):
        for level in (0.0, 0.004, 0.01, 0.016, 0.03):
            path = cache_path(seed, "gate", level, 2)
            if not path.exists():
                continue
            P = np.load(path)[rows]
            rng = np.random.default_rng(1000 * seed + 29)
            for shots in (256, 1024, 4096, None):
                scores = []
                for _ in range(1 if shots is None else reps):
                    F = np.column_stack([H, np.nan_to_num(sample(P, shots, rng))])
                    best = min(RIDGES, key=lambda k: np.mean((rolling(F, y, val, pen(k)) - y[val]) ** 2))
                    scores.append(mse(rolling(F, y, oos, pen(best))))
                records.append({"seed": seed, "level": level, "shots": "exact" if shots is None else shots,
                                "mse_mean": np.mean(scores), "mse_sd": np.std(scores), "n_rep": len(scores),
                                "draws_beating_harx": int(np.sum(np.array(scores) < harx)), "harx": harx})
            print(f"seed {seed} gate {level:g} done", flush=True)
    out = pd.DataFrame(records)
    out.to_csv(ROOT / "data/qr1_hybrid_noise.csv", index=False, float_format="%.6f")
    print(f"\nHARX alone {harx:.4f}. Augmented hybrid, mean over seeds and draws:")
    print(out.groupby(["level", "shots"])["mse_mean"].mean().unstack("shots")[[256, 1024, 4096, "exact"]].round(4).to_string())
    print("\nseeds per level:", out.groupby("level")["seed"].nunique().to_dict())
    print("draws beating HARX / draws:")
    g = out.groupby(["level", "shots"])[["draws_beating_harx", "n_rep"]].sum()
    print((g.draws_beating_harx.astype(str) + "/" + g.n_rep.astype(str)).unstack("shots")[[256, 1024, 4096, "exact"]].to_string())
    print("saved data/qr1_hybrid_noise.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--noise", action="store_true", help="re-score the augmented hybrid on cached noisy features")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    if args.noise:
        return noise_check(args.reps)

    panel, X, _ = load_inputs()
    full, feature_sets, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y = full["y"].to_numpy()
    harx_cols = feature_sets["HARX"]
    H = np.column_stack([np.ones(len(full)), full[harx_cols].to_numpy()])  # intercept + 13 inputs
    n_h = H.shape[1]
    base = slice(0, n_h)

    harx = rolling(H, y, oos, np.zeros(n_h))
    mse = lambda p: float(np.mean((p - y[oos]) ** 2))
    print(f"HARX alone: MSE {mse(harx):.4f}")
    val = np.arange(oos[0] - N_VALIDATION, oos[0])

    out = pd.DataFrame({"y_log": y[oos], "HARX": harx}, index=pd.Index(oos_index, name="date"))
    table = []
    for seed in range(args.seeds):
        Z = reservoir_features(X, evolution(couplings(seed)))[rows]
        ok = np.isfinite(Z).all(axis=1)
        F = np.column_stack([H, np.nan_to_num(Z)])
        assert ok[val].all() and ok[oos].all()
        res = {"seed": seed}
        for name, kw in (("augmented", {}), ("residual", {"base_cols": base})):
            best = min(RIDGES, key=lambda k: np.mean((rolling(F, y, val, np.r_[np.zeros(n_h), np.full(10, k)], **kw) - y[val]) ** 2))
            pred = rolling(F, y, oos, np.r_[np.zeros(n_h), np.full(10, best)], **kw)
            out[f"{name}_s{seed}"] = pred
            res[name] = mse(pred); res[f"{name}_ridge"] = best
        table.append(res)
        print(f"seed {seed:2d}  augmented {res['augmented']:.4f} (ridge {res['augmented_ridge']:g})   "
              f"residual {res['residual']:.4f} (ridge {res['residual_ridge']:g})", flush=True)

    t = pd.DataFrame(table).set_index("seed")
    print()
    for name in ("augmented", "residual"):
        ens = out[[c for c in out if c.startswith(f"{name}_s")]].mean(axis=1).to_numpy()
        stat, p = dm_hac(ens - y[oos], harx - y[oos])
        print(f"{name:9s} mean {t[name].mean():.4f}  sd {t[name].std(ddof=0):.4f}  best {t[name].min():.4f}  worst {t[name].max():.4f}  "
              f"beats HARX {mse(harx):.4f} on {(t[name] < mse(harx)).sum()}/{len(t)} seeds | seed-averaged forecast {mse(ens):.4f}, "
              f"DM vs HARX t={stat:+.2f} p={p:.3f}")
    t.to_csv(ROOT / "data/qr1_hybrid.csv", float_format="%.6f")
    out.to_csv(ROOT / "data/oos_qr1_hybrid.csv", float_format="%.6f")
    print("saved data/qr1_hybrid.csv and data/oos_qr1_hybrid.csv")


if __name__ == "__main__":
    main()
