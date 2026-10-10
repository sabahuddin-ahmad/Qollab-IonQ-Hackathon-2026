"""Classical benchmarks from Li et al. (2505.13933v2), S=1 one-step-ahead.

Models: AR1, AR3, ARMAX, HAR, HARX, RC, RCX.
(LSTM/LSTMX skipped: torch + 245 rolling refits x 100 epochs, noted below.)

Protocol (paper Sec. IV.2):
  target            log monthly RV
  rolling window    fixed 570 obs, step 1, refit every month
  OOS               Aug 1997 - Dec 2017 (245 forecasts)
  RC hyperparams    lr=0.6, sr=0.9, input_scale=0.1;
                    RC 50 neurons / RCX 20 neurons (best per Fig. 10)

Scoring (--scoring):
  corrected         each forecast vs the actual of the same month (245 obs)
  paper             authors' released code, which scores each forecast one
                    month late: pred[:-1] vs actual[1:] (244 obs)
  both              print both side by side (default)

Paper Table 2 S=1 targets:
  HAR 0.1476/2.0431 | HARX 0.1508/2.2436 | AR1 0.1304/1.7279 | AR3 0.1178/1.5893
  ARMAX 0.1145/1.6196 | RC 0.1441/2.1011 | RCX 0.1089/1.6480

Usage: .venv/bin/python scripts/bench_classical.py [--seeds 8] [--scoring both]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.harness import (OLSModel, PAPER, dm_test, iter_folds, load_panel,
                         score_both)


# ------------------------------------------------------------- echo state net
def esn_forecast(seq: np.ndarray, n_neurons: int, seed: int,
                 lr=0.6, sr=0.9, in_scale=0.1, washout=100,
                 ridge_alpha=1.0, std_in: bool = False) -> float:
    """Teacher-forced ESN over `seq` (T x d); forecast next step after last row.

    std_in standardizes the reservoir *inputs* only (dynamics); the ridge
    readout target stays in raw units. Required for RCX: macro features
    (std ~4) otherwise swamp log-RV (std ~0.5) through the shared Win scale.
    """
    seq = seq.copy()
    raw_target = seq[:, 0].copy()  # readout target stays raw even if inputs are standardized
    if std_in:
        mu, sd = seq.mean(axis=0), seq.std(axis=0) + 1e-12
        seq = (seq - mu) / sd
    rng = np.random.default_rng(seed)
    d = seq.shape[1]
    W = rng.standard_normal((n_neurons, n_neurons))
    W *= sr / max(abs(np.linalg.eigvals(W)))
    Win = rng.uniform(-in_scale, in_scale, (n_neurons, d))
    b = rng.uniform(-in_scale, in_scale, n_neurons)
    h = np.zeros(n_neurons)
    states = np.empty((len(seq), n_neurons))
    for t in range(len(seq)):
        h = (1 - lr) * h + lr * np.tanh(W @ h + Win @ seq[t] + b)
        states[t] = h
    H, Y = states[washout:-1], raw_target[washout + 1:]  # predict target col (log RV)
    Wout = Ridge(alpha=ridge_alpha).fit(H, Y)
    h = (1 - lr) * h + lr * np.tanh(W @ h + Win @ seq[-1] + b)
    return float(Wout.predict(h.reshape(1, -1))[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--exog-lags", type=int, default=1, help="lags of X for *MAX models")
    ap.add_argument("--scoring", choices=["corrected", "paper", "both"], default="both")
    args = ap.parse_args()

    full, feature_sets, XL_cols = load_panel(exog_lags=args.exog_lags)
    folds, oos_idx = iter_folds(full)

    models = {m: OLSModel(cols) for m, cols in feature_sets.items()}
    preds: dict[str, np.ndarray] = {m: np.zeros(len(folds)) for m in models}
    rc_preds: dict[str, list] = {"RC": [], "RCX": []}
    rc_specs = {"RC": (["y"], 50, False), "RCX": (["y"] + XL_cols, 20, True)}
    rc_time = {m: 0.0 for m in rc_specs}
    rc_n_params = {"RC": 50 + 1, "RCX": 20 + 1}  # ridge readout + intercept only

    for i, f in enumerate(folds):
        for m, model in models.items():
            model.fit(f.X_window, f.y_window)
            preds[m][i] = model.predict(f.x_next)
        for name, (cols, n, std) in rc_specs.items():
            seq = f.X_window[cols].to_numpy()
            t0 = time.perf_counter()
            rc_preds[name].append(
                [esn_forecast(seq, n, s, std_in=std) for s in range(args.seeds)])
            rc_time[name] += time.perf_counter() - t0
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(folds)} windows", flush=True)

    y_true_log = np.array([f.y_next for f in folds])
    names = list(models) + ["RC", "RCX"]
    for m in ("RC", "RCX"):
        preds[m] = np.array(rc_preds[m]).mean(axis=1)
    per_seed = {m: np.mean((y_true_log[:, None] - np.array(rc_preds[m])) ** 2, axis=0)
                for m in ("RC", "RCX")}

    print(f"\nOOS {oos_idx[0].date()} -> {oos_idx[-1].date()} ({len(oos_idx)} obs), "
          f"exog lags 1..{args.exog_lags}\n")
    want = ("corrected", "paper") if args.scoring == "both" else (args.scoring,)
    print(f"{'model':6s} " + "".join(f"{'MSE_'+s:>12s} {'QL_'+s:>10s}" for s in want) +
          f" {'MSE_paper':>9s} {'QL_paper':>8s}")
    errs = {}
    for m in names:
        scores = score_both(y_true_log, preds[m])
        errs[m] = y_true_log - preds[m]
        row = (f"{m:6s} " + "".join(
            f"{scores[s]['mse']:12.4f} {scores[s]['qlike']:10.4f}" for s in want)
            + f" {PAPER[m][0]:9.4f} {PAPER[m][1]:8.4f}")
        if m in per_seed:
            ps = per_seed[m]
            row += (f"  [seeds={args.seeds}: per-seed MSE min {ps.min():.4f} / "
                    f"mean {ps.mean():.4f} / max {ps.max():.4f}]")
        print(row)

    print("\nDM test, corrected scoring (squared log-error; sign flipped to paper "
          "convention: positive = row better)")
    hdr = "       " + "".join(f"{n:>8s}" for n in names)
    print(hdr)
    for a in names:
        print(f"{a:6s} " + "".join(f"{-dm_test(errs[a], errs[b])[0]:8.2f}" for b in names))

    eff = pd.DataFrame([
        {"model": m, "n_params": models[m].n_params,
         "refit_seconds_total": models[m].fit_seconds,
         "refit_seconds_mean": models[m].fit_seconds / len(folds),
         "mse_corrected": score_both(y_true_log, preds[m])["corrected"]["mse"]}
        for m in models
    ] + [
        {"model": m, "n_params": rc_n_params[m],
         "refit_seconds_total": rc_time[m],
         "refit_seconds_mean": rc_time[m] / len(folds),
         "mse_corrected": score_both(y_true_log, preds[m])["corrected"]["mse"]}
        for m in ("RC", "RCX")
    ])
    eff.to_csv("data/efficiency.csv", index=False)
    print("\nsaved data/efficiency.csv (grows a QR1/LSTM row in Phase B)")

    out = pd.DataFrame({"date": oos_idx, "y_log": y_true_log,
                        "RV": full.loc[oos_idx, "RV"].to_numpy()})
    for m in names:
        out[m] = preds[m]
    old = None
    try:
        old = pd.read_csv("data/oos_classical.csv", parse_dates=["date"])
    except FileNotFoundError:
        pass
    if old is not None:  # preserve columns other tracks added (e.g. RC_paperlike)
        extra = [c for c in old.columns if c not in out.columns]
        if extra:
            out = out.merge(old[["date"] + extra], on="date", how="left")
    out.to_csv("data/oos_classical.csv", index=False)
    print("saved data/oos_classical.csv (OOS preds for QRC comparison)")


if __name__ == "__main__":
    sys.exit(main())
