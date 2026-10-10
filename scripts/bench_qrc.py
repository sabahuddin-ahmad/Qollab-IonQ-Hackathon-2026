"""QR1 (exact, noiseless) over several coupling seeds, scored through the shared harness.

Simulated locally with NumPy; nothing here is an IonQ job. See docs/qrc_spec.md.

Writes data/oos_qr1.csv: out-of-sample forecasts, one column per seed, for the comparison table.

Usage: .venv/bin/python scripts/bench_qrc.py [--seeds 20] [--fit-end 1997-07]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.harness import iter_folds, load_panel, score_both
from src.qrc.features import qr1_inputs
from src.qrc.readout import RidgeReadout
from src.qrc.reference import couplings, evolution, reservoir_features

Z_COLS = [f"z{q}" for q in range(10)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--fit-end", default=None,
                    help="last month used for feature scaling; omit for the authors' full-sample scaling")
    args = ap.parse_args()

    panel = pd.read_csv("data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()
    inputs, target = qr1_inputs(panel, fit_end=args.fit_end)
    full, _, _ = load_panel()
    folds_index = iter_folds(full)[1]
    y_true = full.loc[folds_index, "y"].to_numpy()

    out = pd.DataFrame({"date": folds_index, "y_log": y_true})
    mse = []
    for seed in range(args.seeds):
        Z = reservoir_features(inputs.to_numpy(), evolution(couplings(seed)))
        table = full.join(pd.DataFrame(Z, index=panel.index, columns=Z_COLS)).dropna(subset=Z_COLS)
        folds, _ = iter_folds(table)
        model = RidgeReadout(Z_COLS, target)
        pred = np.array([model.fit(f.X_window, f.y_window).predict(f.x_next) for f in folds])
        out[f"QR1_s{seed}"] = pred
        mse.append(score_both(y_true, pred)["corrected"]["mse"])
        print(f"seed {seed:2d}  MSE {mse[-1]:.4f}")

    mse = np.array(mse)
    print(f"\nQR1 exact, {args.seeds} seeds, scaling fit_end={args.fit_end}: "
          f"MSE mean {mse.mean():.4f}  sd {mse.std():.4f}  min {mse.min():.4f}  max {mse.max():.4f}")
    print(f"trained parameters: {len(Z_COLS)}")
    out.to_csv("data/oos_qr1.csv", index=False)
    print("saved data/oos_qr1.csv")


if __name__ == "__main__":
    main()
