"""Classical benchmarks from Li et al. (2505.13933v2), S=1 one-step-ahead.

Models: AR1, AR3, ARMAX, HAR, HARX, RC, RCX.
(LSTM/LSTMX skipped: torch + 245 rolling refits x 100 epochs, noted below.)

Protocol (paper Sec. IV.2):
  target            log monthly RV
  rolling window    fixed 570 obs, step 1, refit every month
  OOS               Aug 1997 - Dec 2017 (245 forecasts)
  RC hyperparams    lr=0.6, sr=0.9, input_scale=0.1 (App. D);
                    RC 50 neurons / RCX 20 neurons (best per Fig. 10)
  metrics           MSE on log RV; QLIKE on exp-transformed preds (levels)

Paper Table 2 S=1 targets:
  HAR 0.1476/2.0431 | HARX 0.1508/2.2436 | AR1 0.1304/1.7279 | AR3 0.1178/1.5893
  ARMAX 0.1145/1.6196 | RC 0.1441/2.1011 | RCX 0.1089/1.6480

Usage: .venv/bin/python scripts/bench_classical.py [--seeds 10]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

EXOG = ["MKT", "SMB", "HML", "STR", "DP", "EP", "INF", "IP", "TB", "DEF"]
WIN = 570  # fixed rolling window length (Feb1950-Jul1997 = 570 obs)
OOS_START = "1997-08"
OOS_END = "2017-12"

PAPER = {  # Table 2, S=1 : (MSE, QLIKE)
    "HAR": (0.1476, 2.0431), "HARX": (0.1508, 2.2436),
    "AR1": (0.1304, 1.7279), "AR3": (0.1178, 1.5893),
    "ARMAX": (0.1145, 1.6196), "RC": (0.1441, 2.1011),
    "RCX": (0.1089, 1.6480),
}


def lags(df: pd.DataFrame, cols: list[str], k: int) -> pd.DataFrame:
    out = pd.concat({f"{c}_L{i}": df[c].shift(i) for c in cols for i in range(1, k + 1)},
                    axis=1)
    return out


def ols_fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray) -> float:
    Xtr1 = np.column_stack([np.ones(len(Xtr)), Xtr])
    Xte1 = np.concatenate([[1.0], Xte])
    beta, *_ = np.linalg.lstsq(Xtr1, ytr, rcond=None)
    return float(Xte1 @ beta)


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


def qlike(y: np.ndarray, yhat: np.ndarray) -> float:
    # Paper's code (not its printed formula): Patton SUM on abs values,
    # computed on the log-RV scale. Verified: reproduces Table 2 AR1/AR3
    # to 1e-2/1e-4 (1.7172 vs 1.7279, 1.5892 vs 1.5893).
    r = np.abs(y) / np.abs(yhat)
    return float(np.sum(r - np.log(r) - 1))


def dm_test(e1: np.ndarray, e2: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano (squared-error loss, h=1) with normal p-value."""
    from math import erf, sqrt

    d = e1**2 - e2**2
    T = len(d)
    stat = d.mean() / np.sqrt(np.mean((d - d.mean()) ** 2) / T)
    p = 1 - erf(abs(stat) / sqrt(2))
    return stat, p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--exog-lags", type=int, default=1, help="lags of X for *MAX models")
    args = ap.parse_args()

    df = pd.read_csv("data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()
    y = df["RV_log"]
    K = args.exog_lags
    XL = lags(df, EXOG, K)
    feat = pd.DataFrame(index=df.index)
    feat["y_L1"] = y.shift(1)
    feat["y_L2"] = y.shift(2)
    feat["y_L3"] = y.shift(3)
    feat["y_Q"] = y.shift(1).rolling(3).mean()  # quarterly avg of last 3 (incl. t-1)
    feat["y_A"] = y.shift(1).rolling(12).mean()  # annual avg
    feat = pd.concat([feat, XL], axis=1)

    full = pd.concat({"y": y, "RV": df["RV"]}, axis=1).join(feat).dropna()
    oos_idx = full.loc[OOS_START:OOS_END].index
    assert len(oos_idx) == 245, f"OOS len {len(oos_idx)} != 245"

    lin_models = {
        "AR1": ["y_L1"],
        "AR3": ["y_L1", "y_L2", "y_L3"],
        "ARMAX": ["y_L1", "y_L2", "y_L3"] + list(XL.columns),
        "HAR": ["y_L1", "y_Q", "y_A"],
        "HARX": ["y_L1", "y_Q", "y_A"] + list(XL.columns),
    }
    preds_log: dict[str, np.ndarray] = {m: np.zeros(245) for m in lin_models}
    rc_preds: dict[str, list] = {"RC": [], "RCX": []}

    pos = {t: full.index.get_loc(t) for t in oos_idx}
    for i, t in enumerate(oos_idx):
        j = pos[t]
        tr = full.iloc[max(0, j - WIN):j]  # first ~12 windows expand (pre-1951 lags unavailable)
        te = full.loc[t]
        ytr = tr["y"].to_numpy()
        for m, cols in lin_models.items():
            preds_log[m][i] = ols_fit_predict(tr[cols].to_numpy(), ytr, te[cols].to_numpy())
        # ESN: teacher-force through the window, forecast t
        seq_rc = tr[["y"]].to_numpy()
        seq_rcx = tr[["y"] + list(XL.columns)].to_numpy()
        for name, seq, n, std in (("RC", seq_rc, 50, False), ("RCX", seq_rcx, 20, True)):
            rc_preds[name].append(
                [esn_forecast(seq, n, s, std_in=std) for s in range(args.seeds)])
        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/245 windows", flush=True)

    y_true_log = full.loc[oos_idx, "y"].to_numpy()
    y_true = full.loc[oos_idx, "RV"].to_numpy()

    print(f"\nOOS {oos_idx[0].date()} -> {oos_idx[-1].date()} ({len(oos_idx)} obs), "
          f"exog lags 1..{K}\n")
    print(f"{'model':6s} {'MSE':>8s} {'(paper)':>8s} {'QLIKE':>8s} {'(paper)':>8s}")
    errs = {}
    for m in list(lin_models) + ["RC", "RCX"]:
        if m in preds_log:
            pl = preds_log[m]
            tag = ""
        else:
            arr = np.array(rc_preds[m])  # windows x seeds
            pl = arr.mean(axis=1)
            per_seed = np.mean((y_true_log[:, None] - arr) ** 2, axis=0)
            tag = (f"  [seeds={args.seeds}: per-seed MSE min {per_seed.min():.4f} / "
                   f"mean {per_seed.mean():.4f} / max {per_seed.max():.4f}]")
        e = y_true_log - pl
        mse = float(np.mean(e**2))
        ql = qlike(y_true_log, pl)
        errs[m] = e
        pm, pq = PAPER[m]
        print(f"{m:6s} {mse:8.4f} {pm:8.4f} {ql:8.4f} {pq:8.4f}{tag}")

    print("\nDM test (squared log-error): stat above diag vs paper Table 3 sign/magnitude")
    names = list(errs)
    hdr = "       " + "".join(f"{n:>8s}" for n in names)
    print(hdr)
    for a in names:
        row = f"{a:6s} "
        for b in names:
            s, p = dm_test(errs[a], errs[b])
            row += f"{s:8.2f}"
        print(row)

    out = pd.DataFrame({"date": oos_idx, "y_log": y_true_log, "RV": y_true})
    for m in list(lin_models):
        out[m] = preds_log[m]
    for m in ("RC", "RCX"):
        out[m] = np.array(rc_preds[m]).mean(axis=1)
    out.to_csv("data/oos_classical.csv", index=False)
    print("\nsaved data/oos_classical.csv (OOS preds for QRC comparison)")


if __name__ == "__main__":
    sys.exit(main())
