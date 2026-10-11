"""Controls: where does the reservoir's forecasting value come from?

Exact simulation, 245 out-of-sample months. Three questions, each answered with the same readouts
and ridge selection as the experiments they check (scripts/bench_qrc.py, scripts/hybrid_qrc.py):

  alone    QR1 on its own with the entangling gates off (J = 0) and at scaled coupling strengths,
           against linear regressions on the same seven input features.
  hybrid   HARX's 13 inputs plus reservoir values, with the gates off and at scaled strengths,
           against classical transforms of the inputs (cos, absolute value, square).
  tests    Diebold-Mariano tests (HAC, 3 lags) for the comparisons the write-up cites.

With the gates off, <Z> on an input qubit is cos(pi * x) of last month's feature and the memory
qubits stay constant, so that case is a purely classical transform of the inputs.

Writes data/qr1_controls.csv.

Usage: .venv/bin/python scripts/controls_qrc.py [--seeds 20]
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

from hybrid_qrc import dm_hac, rolling
from src.harness import iter_folds, load_panel
from src.qrc.features import FEATURES
from src.qrc.reference import N_QUBITS, couplings, evolution, reservoir_features
from sweep_qrc import N_VALIDATION, RIDGES, choose_ridge, load_inputs, rolling_predictions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    panel, X, target = load_inputs()
    full, fs, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y = full["y"].to_numpy()
    y_scaled = target.scale(y)
    truth = y[oos]
    mse = lambda p: float(np.mean((p - truth) ** 2))
    H = np.column_stack([np.ones(len(full)), full[fs["HARX"]].to_numpy()])
    n_h = H.shape[1]
    val = np.arange(oos[0] - N_VALIDATION, oos[0])

    def alone(Z):  # QR1's readout: scaled target, no intercept, ridge chosen before the test period
        return target.unscale(rolling_predictions(Z, y_scaled, oos, choose_ridge(Z, y_scaled, oos[0])))

    def hybrid(E):  # HARX's columns unpenalised, ridge on the added columns
        E = np.atleast_2d(E.T).T
        F = np.column_stack([H, E])
        pen = lambda k: np.r_[np.zeros(n_h), np.full(E.shape[1], k)]
        best = min(RIDGES, key=lambda k: np.mean((rolling(F, y, val, pen(k)) - y[val]) ** 2))
        return rolling(F, y, oos, pen(best))

    def lag(k):
        L = np.full_like(X, np.nan)
        L[k:] = X[:-k]
        return L[rows]

    features = lambda J: reservoir_features(X, evolution(J))[rows]
    one = np.ones((len(full), 1))
    last = lag(1)
    records = []
    add = lambda block, name, pred, **extra: records.append({"block": block, "model": name, "mse": mse(pred), **extra})

    harx = rolling(H, y, oos, np.zeros(n_h))
    add("reference", "HARX", harx)

    # --- QR1 on its own
    Z_off = features(np.zeros((N_QUBITS, N_QUBITS)))
    assert np.allclose(Z_off[oos, :7], np.cos(np.pi * last[oos])) and np.allclose(Z_off[oos, 7:], 1)
    p_off = alone(Z_off)
    add("alone", "QR1, entangling gates off", p_off)
    add("alone", "linear regression, the 7 inputs, last month", alone(np.hstack([last, one])))
    add("alone", "linear regression, the 7 inputs, 3 months", alone(np.hstack([lag(1), lag(2), lag(3), one])))
    Zs = [features(couplings(s)) for s in range(args.seeds)]
    p_on = [alone(Z) for Z in Zs]
    m_on = np.array([mse(p) for p in p_on])
    add("alone", f"QR1 as designed, mean of {args.seeds} seeds", np.mean(p_on, axis=0), mse_seed_mean=m_on.mean(),
        mse_seed_sd=m_on.std(), seeds_beating_gates_off=int((m_on < mse(p_off)).sum()))
    for c in (0.25, 0.5, 2.0):
        ms = [mse(alone(features(c * couplings(s)))) for s in range(5)]
        records.append({"block": "alone", "model": f"QR1, coupling x{c:g}, mean of 5 seeds", "mse": float(np.mean(ms))})

    # --- hybrid with HARX
    h_off = hybrid(Z_off[:, :7])
    add("hybrid", "HARX + reservoir, entangling gates off (= cos of the 7 inputs)", h_off)
    h_on = [hybrid(Z) for Z in Zs]
    hm = np.array([mse(p) for p in h_on])
    add("hybrid", f"HARX + reservoir as designed, mean of {args.seeds} seeds", np.mean(h_on, axis=0), mse_seed_mean=hm.mean(),
        mse_seed_sd=hm.std(), seeds_beating_gates_off=int((hm < mse(h_off)).sum()), seeds_beating_harx=int((hm < mse(harx)).sum()))
    for c in (0.25, 0.5, 2.0):
        ms = [mse(hybrid(features(c * couplings(s)))) for s in range(5)]
        records.append({"block": "hybrid", "model": f"HARX + reservoir, coupling x{c:g}, mean of 5 seeds", "mse": float(np.mean(ms))})
    add("hybrid", "HARX + reservoir memory qubits only, seed 0", hybrid(Zs[0][:, 7:]))
    i_m, i_s = FEATURES.index("MKT"), FEATURES.index("STR")
    two = last[:, [i_m, i_s]]
    h_cos2 = hybrid(np.cos(np.pi * two))
    add("hybrid", "HARX + cos of MKT and STR", h_cos2)
    add("hybrid", "HARX + MKT^2 and STR^2", hybrid(two**2))
    add("hybrid", "HARX + |MKT| and |STR|", hybrid(np.abs(two)))
    add("hybrid", "HARX + cos of the other five inputs", hybrid(np.cos(np.pi * np.delete(last, [i_m, i_s], axis=1))))
    add("hybrid", "HARX + the 21 raw reservoir inputs", hybrid(np.hstack([lag(1), lag(2), lag(3)])))
    for q, name in enumerate(FEATURES):
        add("hybrid", f"HARX + cos of {name} only", hybrid(np.cos(np.pi * last[:, q])))

    # --- significance
    e = lambda p: p - truth
    qr1 = pd.read_csv(ROOT / "data/oos_qr1.csv")
    qr1_avg = qr1[[c for c in qr1 if c.startswith("QR1_s")]].mean(axis=1).to_numpy()
    for name, a, b in (("QR1 (bench_qrc, seed-averaged) vs HARX", qr1_avg, harx),
                       ("QR1 as designed (seed-averaged) vs gates off, alone", np.mean(p_on, axis=0), p_off),
                       ("hybrid as designed (seed-averaged) vs HARX", np.mean(h_on, axis=0), harx),
                       ("hybrid as designed (seed-averaged) vs hybrid gates off", np.mean(h_on, axis=0), h_off),
                       ("hybrid gates off vs HARX", h_off, harx),
                       ("HARX + cos of MKT and STR vs HARX", h_cos2, harx)):
        t, p = dm_hac(e(a), e(b))
        records.append({"block": "tests", "model": name, "mse": mse(a), "mse_other": mse(b), "dm_t": t, "dm_p": p})

    out = pd.DataFrame(records)
    out.to_csv(ROOT / "data/qr1_controls.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 220)
    print(out.fillna("").to_string(index=False))
    print("\nsaved data/qr1_controls.csv")


if __name__ == "__main__":
    main()
