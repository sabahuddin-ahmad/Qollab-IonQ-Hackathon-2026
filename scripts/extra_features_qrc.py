"""Experiment: also use the discarded input qubits' measurements as readout features.

QR1 reads <Z> on 10 qubits after the last of three months. Here the seven input qubits are also
read after each of the two earlier months, just before they are discarded: 10 + 7 + 7 = 24
features. Two reservoirs:

  restart  QR1's three-month restart. In the 24-qubit no-reset layout these are simply the 14
           qubits QR1 leaves unmeasured, so this runs on IonQ backends with every measurement
           at the end of the circuit.
  stream   the continuous stream of scripts/stream_qrc.py (hidden qubits never reset).

Exact NumPy, noiseless. Ridge chosen on the 120 months before the test period.

Usage: .venv/bin/python scripts/extra_features_qrc.py [--seeds 20]
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

from src.harness import iter_folds, load_panel
from src.qrc.reference import K_LAGS, N_HIDDEN, N_INPUT, _input_state, _z_signs, couplings, evolution, reservoir_features
from stream_qrc import stream_features
from sweep_qrc import choose_ridge, load_inputs, rolling_predictions


def restart_features_24(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Row t: [10 final <Z>, 7 input <Z> after month t-2, 7 input <Z> after month t-3]."""
    signs = _z_signs()
    out = np.full((len(X), 24), np.nan)
    for t in range(K_LAGS, len(X)):
        lags = X[t - K_LAGS : t]
        if not np.isfinite(lags).all():
            continue
        probs, hidden = np.array([1.0]), np.eye(1, 2**N_HIDDEN, dtype=complex)
        z = []
        for x in lags:
            states = np.stack([U @ np.kron(_input_state(x), h) for h in hidden])
            z.append(signs @ np.einsum("m,ms->s", probs, np.abs(states) ** 2))
            m = states.reshape(len(states), 2**N_INPUT, 2**N_HIDDEN)
            w, v = np.linalg.eigh(np.einsum("m,mih,mig->hg", probs, m, m.conj()))
            keep = w > 1e-14
            probs, hidden = w[keep], v[:, keep].T
        out[t] = np.concatenate([z[2], z[1][:N_INPUT], z[0][:N_INPUT]])
    return out


def stream_features_24(Z: np.ndarray) -> np.ndarray:
    """The stream's 10 features plus the input-qubit <Z> read one and two months earlier."""
    out = np.full((len(Z), 24), np.nan)
    out[2:] = np.hstack([Z[2:], Z[1:-1, :N_INPUT], Z[:-2, :N_INPUT]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()

    panel, X, target = load_inputs()
    full, _, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y_log = full["y"].to_numpy()
    y_scaled = target.scale(y_log)

    def score(Z):
        ridge = choose_ridge(Z, y_scaled, oos[0])
        pred = target.unscale(rolling_predictions(Z, y_scaled, oos, ridge))
        return float(np.mean((pred - y_log[oos]) ** 2)), ridge, pred

    names = ["restart_10", "restart_24", "stream_10", "stream_24"]
    preds = {n: [] for n in names}
    table = []
    for seed in range(args.seeds):
        U = evolution(couplings(seed))
        r24 = restart_features_24(X, U)
        assert np.nanmax(np.abs(r24[:, :10] - reservoir_features(X, U))) < 1e-10
        s10 = stream_features(X, U)
        feats = {"restart_10": r24[:, :10], "restart_24": r24, "stream_10": s10, "stream_24": stream_features_24(s10)}
        row, ridges = [], []
        for n in names:
            m, k, p = score(feats[n][rows])
            row.append(m); ridges.append(k); preds[n].append(p)
        table.append(row)
        print(f"seed {seed:2d}  " + "   ".join(f"{n} {m:.4f} (ridge {k:g})" for n, m, k in zip(names, row, ridges)), flush=True)
    t = np.array(table)
    print()
    for i, n in enumerate(names):
        ens = np.mean((np.mean(preds[n], axis=0) - y_log[oos]) ** 2)
        print(f"{n:11s} mean {t[:, i].mean():.4f}  sd {t[:, i].std():.4f}  best {t[:, i].min():.4f}  worst {t[:, i].max():.4f}  "
              f"beats HARX 0.1030 on {(t[:, i] < 0.1030).sum()}/{len(t)}, ARMAX 0.1070 on {(t[:, i] < 0.1070).sum()}/{len(t)}  "
              f"seed-averaged forecast {ens:.4f}")
    print(f"restart_24 better than restart_10 on {(t[:, 1] < t[:, 0]).sum()}/{len(t)} seeds; "
          f"stream_24 better than stream_10 on {(t[:, 3] < t[:, 2]).sum()}/{len(t)}")
    pd.DataFrame(t, columns=names).rename_axis("seed").to_csv(ROOT / "data/qr1_extra_features.csv", float_format="%.6f")
    print("saved data/qr1_extra_features.csv")


if __name__ == "__main__":
    main()
