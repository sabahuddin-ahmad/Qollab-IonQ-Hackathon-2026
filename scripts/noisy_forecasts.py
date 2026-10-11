"""Month-by-month QR1 forecasts under noise, for the actual-against-forecast figure.

Reads the cached outcome probabilities written by ``sweep_qrc.py simulate`` (no circuits run
here), draws one set of shots per case with a fixed seed, and refits the readout each month.
Coupling seed 0, 2 Trotter steps. Each case is a single draw, so its MSE differs slightly from
the 10-draw mean in data/qr1_noise_shots.csv.

Usage: .venv/bin/python scripts/noisy_forecasts.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.harness import iter_folds, load_panel
from src.qrc.circuit import z_from_probabilities
from sweep_qrc import cache_path, choose_ridge, load_inputs, rolling_predictions, sample

SEED, TROTTER_STEPS = 0, 2
# column -> (gate error, shots, readout trained on the same noisy features?)
CASES = {
    "exact_p0": (0.0, None, True),
    "matched_p0.016_s1024": (0.016, 1024, True),
    "matched_p0.016_s256": (0.016, 256, True),
    "matched_p0.03_s1024": (0.03, 1024, True),
    "noiseless_trained_p0.016_s1024": (0.016, 1024, False),
}


def main() -> None:
    panel, _, target = load_inputs()
    full, _, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y_log = full["y"].to_numpy()
    y_scaled = target.scale(y_log)

    Z_ideal = z_from_probabilities(np.load(cache_path(SEED, "gate", 0.0, TROTTER_STEPS))[rows])
    ridge_ideal = choose_ridge(Z_ideal, y_scaled, oos[0])
    rng = np.random.default_rng(2026)
    out = pd.DataFrame({"y_log": y_log[oos]}, index=pd.Index(oos_index, name="date"))
    for name, (level, shots, matched) in CASES.items():
        Z = sample(np.load(cache_path(SEED, "gate", level, TROTTER_STEPS))[rows], shots, rng)
        if matched:
            pred = rolling_predictions(Z, y_scaled, oos, choose_ridge(Z, y_scaled, oos[0]))
        else:
            pred = rolling_predictions(Z, y_scaled, oos, ridge_ideal, Z_train=Z_ideal)
        out[name] = target.unscale(pred)
        print(f"{name:34s} MSE {np.mean((out[name] - out['y_log']) ** 2):.4f}")
    out.to_csv(ROOT / "data/oos_qr1_noisy.csv", float_format="%.6f")
    print("saved data/oos_qr1_noisy.csv")


if __name__ == "__main__":
    main()
