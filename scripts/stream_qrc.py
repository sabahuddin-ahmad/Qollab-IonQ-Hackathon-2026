"""One experiment: QR1 as a continuous stream instead of a three-month restart.

Same reservoir as QR1 (same couplings, tau = 1, RY encoding, <Z> readout on 10 qubits, ridge
readout). The only change: the three memory qubits are never re-initialised. Each month the
seven input qubits are reset and loaded with that month's features, all ten evolve, and the
memory qubits carry on to the next month. Resetting the inputs is the dissipation that makes
old months fade; QR1 is this stream cut off after three months.

Exact NumPy, noiseless. One configuration, fixed before looking at the test months.

Usage: .venv/bin/python scripts/stream_qrc.py [--seeds 20]
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
from src.qrc.reference import N_HIDDEN, N_INPUT, N_QUBITS, _input_state, _z_signs, couplings, evolution, reservoir_features
from sweep_qrc import choose_ridge, load_inputs, rolling_predictions


def stream_features(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Row t holds <Z> after feeding every month up to t-1 in order; memory qubits are never reset."""
    out = np.full((len(X), N_QUBITS), np.nan)
    probs, hidden = np.array([1.0]), np.eye(1, 2**N_HIDDEN, dtype=complex)  # |000>
    signs = _z_signs()
    for t in range(1, len(X)):
        x = X[t - 1]
        if not np.isfinite(x).all():
            continue
        psi = _input_state(x)
        states = np.stack([U @ np.kron(psi, h) for h in hidden])
        out[t] = signs @ np.einsum("m,ms->s", probs, np.abs(states) ** 2)
        m = states.reshape(len(states), 2**N_INPUT, 2**N_HIDDEN)
        w, v = np.linalg.eigh(np.einsum("m,mih,mig->hg", probs, m, m.conj()))
        keep = w > 1e-14
        probs, hidden = w[keep], v[:, keep].T
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

    out = pd.DataFrame({"y_log": y_log[oos]}, index=pd.Index(oos_index, name="date"))
    records = []
    for seed in range(args.seeds):
        U = evolution(couplings(seed))
        restart, k_r, _ = score(reservoir_features(X, U)[rows])
        stream, k_s, pred = score(stream_features(X, U)[rows])
        out[f"stream_s{seed}"] = pred
        records.append((restart, stream))
        print(f"seed {seed:2d}  QR1 restart {restart:.4f} (ridge {k_r:g})   stream {stream:.4f} (ridge {k_s:g})   diff {stream - restart:+.4f}", flush=True)
    r = np.array(records)
    print(f"\nmean over {args.seeds} seeds: QR1 restart {r[:, 0].mean():.4f} (sd {r[:, 0].std():.4f})   "
          f"stream {r[:, 1].mean():.4f} (sd {r[:, 1].std():.4f})")
    print(f"stream better on {(r[:, 1] < r[:, 0]).sum()}/{args.seeds} seeds; beats HARX 0.1030 on {(r[:, 1] < 0.1030).sum()}, "
          f"ARMAX 0.1070 on {(r[:, 1] < 0.1070).sum()}; best {r[:, 1].min():.4f}, worst {r[:, 1].max():.4f}")
    ens = out[[c for c in out if c.startswith("stream_s")]].mean(axis=1)
    print(f"forecast averaged over seeds: stream {np.mean((ens - out['y_log']) ** 2):.4f}")
    out.to_csv(ROOT / "data/oos_qr1_stream.csv", float_format="%.6f")
    print("saved data/oos_qr1_stream.csv")


if __name__ == "__main__":
    main()
