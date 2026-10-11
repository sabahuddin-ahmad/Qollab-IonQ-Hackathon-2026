"""Does the ranking depend on the training window? The main models at 120, 240 and 570 months.

Same 245 test months and one-month-ahead protocol as the benchmark; only the number of months each
model is fitted on changes. 570 is the benchmark setting and is included as a check.

  linear         AR1, AR3, HAR, ARMAX, HARX (OLS), HARX on QR1's five macro variables, HARX + cos(MKT, STR);
                 HARX, ARMAX and HAR again with a ridge penalty tuned the same way as QR1's (inputs
                 standardized on the months before the test period, intercept unpenalised)
  quantum        QR1 with the benchmark readout (ridge 1e-8, no intercept) and with the ridge penalty chosen
                 on the 120 months before the test period; QR1 with entangling gates off; the volatility-only
                 reservoir (1 input + 3 memory qubits, 12 months). 20 coupling seeds, exact simulation.
  classical RC   RC and RCX as in scripts/bench_classical.py, forecast averaged over 8 seeds. Their warm-up
                 of 100 months is scaled with the window (100 * window / 570).

Writes data/windows.csv, or data/windows_<name>.csv when --windows and --name are given.

Usage: .venv/bin/python scripts/bench_windows.py [--seeds 20] [--windows 24 36 48 60 90 120 --name short]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from bench_classical import esn_forecast
from bench_univariate import reservoir, unitary
from src.harness import iter_folds, load_panel
from src.qrc.features import FEATURES
from src.qrc.reference import N_QUBITS, couplings, evolution, reservoir_features
from sweep_qrc import N_VALIDATION, RIDGES, load_inputs

WINDOWS = (120, 240, 570)


def roll(F: np.ndarray, y: np.ndarray, positions: np.ndarray, win: int, penalty: np.ndarray) -> np.ndarray:
    out = np.empty(len(positions))
    for i, j in enumerate(positions):
        A, b = F[max(0, j - win) : j], y[max(0, j - win) : j]
        out[i] = F[j] @ np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ b)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--windows", type=int, nargs="+", default=list(WINDOWS))
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    panel, X, target = load_inputs()
    full, fs, exog = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    val = np.arange(oos[0] - N_VALIDATION, oos[0])
    y = full["y"].to_numpy()
    y_scaled = target.scale(y)
    mse = lambda p: float(np.mean((p - y[oos]) ** 2))
    one = np.ones((len(full), 1))
    last = np.full_like(X, np.nan)
    last[1:] = X[:-1]
    last = last[rows]

    def ols(cols, win, extra=None):
        F = np.column_stack([one, full[cols].to_numpy()] + ([extra] if extra is not None else []))
        return mse(roll(F, y, oos, win, np.zeros(F.shape[1])))

    def ridge_linear(cols, win):
        raw = full[cols].to_numpy()
        Zs = (raw - raw[: oos[0]].mean(axis=0)) / raw[: oos[0]].std(axis=0)
        S = np.column_stack([Zs, one])
        pen = lambda k: np.r_[np.full(len(cols), k), 0.0]
        grid = RIDGES + [10.0, 100.0, 1000.0]
        k = min(grid, key=lambda r: np.mean((roll(S, y_scaled, val, win, pen(r)) - y_scaled[val]) ** 2))
        return mse(target.unscale(roll(S, y_scaled, oos, win, pen(k)))), k

    def readout(Z, win, intercept, tuned):
        S = np.nan_to_num(np.column_stack([Z, one]) if intercept else Z)
        pen = lambda k: np.r_[np.full(Z.shape[1], k), np.zeros(S.shape[1] - Z.shape[1])]
        k = 1e-8
        if tuned:
            k = min(RIDGES, key=lambda r: np.mean((roll(S, y_scaled, val, win, pen(r)) - y_scaled[val]) ** 2))
        return target.unscale(roll(S, y_scaled, oos, win, pen(k)))

    Z_qr1 = [reservoir_features(X, evolution(couplings(s)))[rows] for s in range(args.seeds)]
    Z_off = reservoir_features(X, evolution(np.zeros((N_QUBITS, N_QUBITS))))[rows][:, :7]
    rv = panel["RV_log"]
    x_uni = ((rv - rv.min()) / (rv.max() - rv.min()) - 1.0).to_numpy()[:, None]
    Z_uni = [reservoir(x_uni, unitary(couplings(s, 4)), 1, 12)[rows] for s in range(args.seeds)]
    two = np.cos(np.pi * last[:, [FEATURES.index("MKT"), FEATURES.index("STR")]])
    qr1_exog = [c for c in exog if c.split("_")[0] in ("MKT", "STR", "DP", "IP", "DEF")]
    har_core = [c for c in fs["HARX"] if c not in exog]

    records = []
    for win in args.windows:
        add = lambda model, group, weights, value, **kw: records.append({"window": win, "model": model, "inputs": group, "weights": weights, "mse": value, **kw})
        harx = ols(fs["HARX"], win)
        for m in ("HARX", "ARMAX"):
            add(m, "volatility + macro", len(fs[m]) + 1, ols(fs[m], win))
        for m in ("HARX", "ARMAX", "HAR"):
            value, k = ridge_linear(fs[m], win)
            add(f"{m}, tuned ridge", "volatility only" if m == "HAR" else "volatility + macro", len(fs[m]) + 1, value, ridge=k)
        value, k = ridge_linear(har_core + qr1_exog, win)
        add("HARX, QR1's macro variables, tuned ridge", "volatility + macro", len(har_core) + len(qr1_exog) + 1, value, ridge=k)
        add("HARX, QR1's macro variables", "volatility + macro", len(har_core) + len(qr1_exog) + 1, ols(har_core + qr1_exog, win))
        add("HARX + cos(MKT, STR)", "volatility + macro", len(fs["HARX"]) + 3, ols(fs["HARX"], win, two))
        for label, intercept, tuned in (("QR1, benchmark readout", False, False), ("QR1, tuned ridge + intercept", True, True)):
            preds = [readout(Z, win, intercept, tuned) for Z in Z_qr1]
            per = np.array([mse(p) for p in preds])
            add(label, "volatility + macro", 10 + intercept, per.mean(), mse_sd=per.std(), mse_best=per.min(), mse_worst=per.max(),
                mse_seed_averaged=mse(np.mean(preds, axis=0)), seeds_beating_harx=int((per < harx).sum()),
                seeds_beating_ridge_harx=int((per < ridge_linear(fs["HARX"], win)[0]).sum()))
        add("QR1, entangling gates off", "volatility + macro", 8, mse(readout(Z_off, win, True, True)))
        for m in ("HAR", "AR3", "AR1"):
            add(m, "volatility only", len(fs[m]) + 1, ols(fs[m], win))
        preds = [readout(Z, win, True, True) for Z in Z_uni]
        per = np.array([mse(p) for p in preds])
        add("QR-uni (1 input + 3 memory, 12 months)", "volatility only", 5, per.mean(), mse_sd=per.std(), mse_best=per.min(),
            mse_worst=per.max(), mse_seed_averaged=mse(np.mean(preds, axis=0)))
        washout = round(100 * win / 570)
        for m, cols, n, std in (("RC", ["y"], 50, False), ("RCX", ["y"] + exog, 20, True)):
            seqs = full[cols].to_numpy()
            p = np.array([np.mean([esn_forecast(seqs[max(0, j - win) : j], n, s, std_in=std, washout=washout) for s in range(8)]) for j in oos])
            add(m, "volatility only" if m == "RC" else "volatility + macro", n + 1, mse(p))
        print(f"window {win} done", flush=True)

    res = pd.DataFrame(records)
    out = ROOT / "data" / (f"windows_{args.name}.csv" if args.name else "windows.csv")
    res.to_csv(out, index=False, float_format="%.6f")
    table = res.pivot(index="model", columns="window", values="mse")
    meta = res.drop_duplicates("model").set_index("model")[["inputs", "weights"]]
    order = res[res.window == max(args.windows)].sort_values(["inputs", "mse"], ascending=[False, True]).model
    pd.set_option("display.width", 200)
    print("\n" + meta.join(table).loc[order].round(4).to_string())
    q = res[res.model.str.startswith("QR1,") & res.mse_sd.notna()][["window", "model", "mse", "mse_sd", "mse_best", "mse_worst", "mse_seed_averaged", "seeds_beating_harx", "seeds_beating_ridge_harx"]]
    print("\n" + q.round(4).to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
