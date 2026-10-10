"""Classical RC/RCX replication of the paper's own code.

Faithful port of cell 8 in LeeQY1996/classical_reservoir.ipynb:
  - 3-step stateless ESN: reset per sample, run K=3 steps, keep LAST state
  - reservoirpy Reservoir(N, lr=0.6, sr=0.9, input_scaling=1)  [NB: paper text says 0.1]
  - Ridge(ridge=1e-7) readout refit per rolling window
  - fixed 571-obs rolling window, 245 OOS (Aug 1997 - Dec 2017)
  - RCX features: RV,MKT,diff_DP,IP,DEF,EP,SMB,diff_TB,HML,INF,STR (raw scale)
  - RC: RV only
Metrics: MSE on log-RV (raw scale, Table 2), Patton-SUM QLIKE on abs(log-RV).

Paper Table 2 S=1: RC 0.1441/2.1011 | RCX 0.1089/1.6480.
Their code sets NO seed (rpy.set_seed commented out) -> sweep seeds, report dist.

Usage: .venv/bin/python scripts/bench_rc.py [--seeds 0 1 2] [--neurons 50 20]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import reservoirpy as rpy  # noqa: F401 (kept for version pinning context)
from reservoirpy.nodes import Reservoir, Ridge

F11 = ["RV", "MKT", "diff_DP", "IP", "DEF", "EP", "SMB", "diff_TB", "HML", "INF", "STR"]
FSEL7 = ["RV", "MKT", "STR", "RVq", "EP", "INF", "DEF"]  # QR2-style selected set
L, TOTAL, K = 245, 816, 3


def load_dff(std_exog: str = "none") -> pd.DataFrame:
    df = pd.read_csv("data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()
    # dff.csv is not in their repo; reconstruct: raw log-RV + raw exog + raw diffs.
    # Extra Jan-1950 row to reach 816 rows like theirs.
    dff = pd.DataFrame(index=df.index)
    dff["RV"] = df["RV_log"]
    for c in ["MKT", "SMB", "HML", "STR", "EP", "IP", "INF", "TB", "DEF", "RVq"]:
        dff[c] = df[c]
    dff["diff_DP"] = df["DP"].diff()
    dff["diff_TB"] = df["TB"].diff()
    if std_exog == "zscore":
        for c in dff.columns:
            if c != "RV":
                dff[c] = (dff[c] - dff[c].mean()) / (dff[c].std() + 1e-12)
    elif std_exog == "minmax":
        for c in dff.columns:
            if c != "RV":
                dff[c] = 2 * (dff[c] - dff[c].min()) / (dff[c].max() - dff[c].min() + 1e-12) - 1
    dff = dff.fillna(0)
    assert len(dff) == 816  # like their Data.CSV (Jan 1950 - Dec 2017); OOS dates identical
    return dff


def run_model(dff: pd.DataFrame, features: list[str], n: int, seed: int) -> np.ndarray:
    WL = len(dff) - L
    RV = dff["RV"].to_numpy().reshape(-1, 1)
    Data = dff[features].to_numpy()
    reservoir = Reservoir(n, input_dim=len(features), lr=0.6, sr=0.9,
                          input_scaling=1, seed=seed)
    preds = np.zeros((L, 1))
    def fresh_run(X):
        # 0.4.x: state is lazily created; reset (to zeros) once it exists.
        # Faithful to their run(X, reset=True) on 0.3.x: every 3-step
        # snippet starts from a fresh zero state.
        if hasattr(reservoir, "state"):
            reservoir.reset()
        return reservoir.run(X)

    ws = 0
    while ws < L:
        train_states = np.empty((WL - K, reservoir.output_dim))
        for t in range(WL - K):
            X = Data[ws + t:ws + t + K, :].reshape(-1, len(features))
            train_states[t, :] = fresh_run(X)[-1, :]
        train_y = RV[ws + K:ws + WL]
        readout = Ridge(ridge=1e-7)
        readout = readout.fit(train_states, train_y)
        X_test = Data[ws + WL - K:ws + WL, :].reshape(-1, len(features))
        preds[ws, 0] = readout.run(fresh_run(X_test)[-1:, :])[0, 0]
        ws += 1
    return preds.ravel()


def patton_sum(y: np.ndarray, yh: np.ndarray) -> float:
    r = np.abs(y) / np.abs(yh)
    return float(np.sum(r - np.log(r) - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--neurons", type=int, nargs="+", default=[50, 20])
    ap.add_argument("--feature-set", choices=["full11", "sel7"], default="full11")
    ap.add_argument("--std-exog", choices=["none", "zscore", "minmax"], default="none")
    args = ap.parse_args()

    dff = load_dff(args.std_exog)
    feats = F11 if args.feature_set == "full11" else FSEL7
    actual = dff["RV"].to_numpy()[-L:]
    oos_dates = dff.index[-L:]
    assert str(oos_dates[0].date()) == "1997-08-31" and str(oos_dates[-1].date()) == "2017-12-31"

    for features, tag in ((["RV"], "RC"), (feats, "RCX")):
        for n in args.neurons:
            mses, qls = [], []
            for s in args.seeds:
                p = run_model(dff, features, n, s)
                mses.append(float(np.mean((p - actual) ** 2)))
                qls.append(patton_sum(actual, p))
                print(f"  {tag} N={n} seed={s}: MSE={mses[-1]:.4f} QLIKE={qls[-1]:.4f}", flush=True)
            print(f"{tag} N={n}: MSE min={min(mses):.4f} mean={np.mean(mses):.4f} "
                  f"max={max(mses):.4f} | QLIKE mean={np.mean(qls):.4f}")
    print("paper: RC 0.1441/2.1011 | RCX 0.1089/1.6480")


if __name__ == "__main__":
    sys.exit(main())
