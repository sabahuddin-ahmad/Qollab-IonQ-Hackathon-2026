"""HARX and ARMAX cut down to 10 trained weights, for a like-for-like comparison with QR1.

Both models have 14 weights: intercept, three volatility terms and ten lagged macro variables.
Ten weights means keeping six of the ten macro variables (210 possible subsets). Reported here:

  selected   the subset with the lowest rolling error on the 120 months before the test period
  all        the spread of test error over all 210 subsets (how much the choice matters)
  QR1 set    only the variables QR1 itself is given (MKT, STR, DP, IP, DEF): 9 weights

Same protocol as the benchmark: OLS, rolling window of WIN months, 245 test months.
Writes data/reduced_models.csv.

Usage: .venv/bin/python scripts/bench_reduced.py
"""

from __future__ import annotations

import itertools
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from hybrid_qrc import dm_hac, rolling
from src.harness import iter_folds, load_panel
from sweep_qrc import N_VALIDATION

QR1_EXOG = ["MKT", "STR", "DP", "IP", "DEF"]


def main() -> None:
    full, fs, exog = load_panel()
    _, oos_index = iter_folds(full)
    oos = full.index.get_indexer(oos_index)
    val = np.arange(oos[0] - N_VALIDATION, oos[0])
    y = full["y"].to_numpy()
    qr1 = pd.read_csv(ROOT / "data/oos_qr1.csv")
    seeds = qr1[[c for c in qr1 if c.startswith("QR1_s")]]
    qr1_avg = seeds.mean(axis=1).to_numpy()
    qr1_mean = float(((seeds.sub(y[oos], axis=0)) ** 2).mean().mean())

    def fit(cols, positions):
        F = np.column_stack([np.ones(len(full)), full[cols].to_numpy()])
        return rolling(F, y, positions, np.zeros(F.shape[1]))

    mse = lambda p, pos=oos: float(np.mean((p - y[pos]) ** 2))
    records = []
    print(f"QR1: 10 weights, mean of 20 seeds {qr1_mean:.4f}; macro variables available: {exog}\n")
    for model in ("HARX", "ARMAX"):
        core = [c for c in fs[model] if c not in exog]
        full_pred = fit(fs[model], oos)
        subsets = list(itertools.combinations(exog, 6))
        val_err = np.array([mse(fit(core + list(s), val), val) for s in subsets])
        test = [fit(core + list(s), oos) for s in subsets]
        test_err = np.array([mse(p) for p in test])
        best = int(val_err.argmin())
        chosen = test[best]
        qset = fit(core + [c for c in exog if c.split("_")[0] in QR1_EXOG], oos)
        t, p = dm_hac(qr1_avg - y[oos], chosen - y[oos])
        rows = [
            (f"{model}, all 14 weights", 14, mse(full_pred), ""),
            (f"{model}, 10 weights, subset chosen before the test period", 10, mse(chosen), ", ".join(subsets[best])),
            (f"{model}, 10 weights, median of all 210 subsets", 10, float(np.median(test_err)), ""),
            (f"{model}, 10 weights, best of all 210 subsets (chosen on the test set)", 10, float(test_err.min()), ", ".join(subsets[int(test_err.argmin())])),
            (f"{model}, 10 weights, worst of all 210 subsets", 10, float(test_err.max()), ", ".join(subsets[int(test_err.argmax())])),
            (f"{model}, QR1's own variables only", 1 + len(core) + len(QR1_EXOG), mse(qset), ", ".join(QR1_EXOG)),
            (f"{model}, no macro variables", 1 + len(core), mse(fit(core, oos)), ""),
        ]
        for name, n, m, note in rows:
            records.append({"model": name, "weights": n, "mse": m, "macro_variables": note})
            print(f"{name:66s} {n:3d}  {m:.4f}  {note}")
        print(f"  10-weight subsets beating QR1's mean {qr1_mean:.4f}: {(test_err < qr1_mean).sum()}/210; "
              f"beating QR1's best seed {((seeds.sub(y[oos], axis=0)) ** 2).mean().min():.4f}: {(test_err < ((seeds.sub(y[oos], axis=0)) ** 2).mean().min()).sum()}/210")
        print(f"  QR1 seed-averaged forecast {mse(qr1_avg):.4f} vs chosen 10-weight {model} {mse(chosen):.4f}: DM t={t:+.2f} p={p:.3f}\n")
    pd.DataFrame(records).to_csv(ROOT / "data/reduced_models.csv", index=False, float_format="%.6f")
    print("saved data/reduced_models.csv")


if __name__ == "__main__":
    main()
