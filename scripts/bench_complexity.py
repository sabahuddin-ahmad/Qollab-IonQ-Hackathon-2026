"""What HARX and QR1 each need to produce one monthly forecast: sizes and wall-clock time.

Sizes are read from the code. Times are measured here on this machine, except the IonQ simulator
runs, which are copied from data/ionq_runs.json. Writes data/complexity.csv.

Usage: .venv/bin/python scripts/bench_complexity.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.harness import WIN, load_panel
from src.qrc.circuit import outcome_probabilities, qr1_circuit
from src.qrc.noise import gate_noise_model
from src.qrc.reference import K_LAGS, N_INPUT, N_QUBITS, couplings, evolution, month_features
from sweep_qrc import load_inputs


def timed(fn, repeats: int) -> float:
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - t0) / repeats


def main() -> None:
    _, X, _ = load_inputs()
    _, feature_sets, _ = load_panel()
    harx_inputs = len(feature_sets["HARX"])
    lags = X[-K_LAGS:]
    J = couplings(0)
    U = evolution(J)
    rng = np.random.default_rng(0)
    A, y = rng.standard_normal((WIN, N_QUBITS)), rng.standard_normal(WIN)

    eff = pd.read_csv(ROOT / "data/efficiency.csv").set_index("model")
    ionq = {r["shots"]: r["execution_seconds"] for r in json.loads((ROOT / "data/ionq_runs.json").read_text())["runs"]}
    noisy = qr1_circuit(lags, J, r=2)
    rows = [
        ("HARX", "size", "trained weights", int(eff.loc["HARX", "n_params"]), "count"),
        ("QR1", "size", "trained weights", N_QUBITS, "count"),
        ("HARX", "size", "input values per forecast", harx_inputs, "count"),
        ("QR1", "size", "input values per forecast", K_LAGS * N_INPUT, "count"),
        ("HARX", "size", "fixed random parameters", 0, "count"),
        ("QR1", "size", "fixed random parameters", N_QUBITS * (N_QUBITS - 1) // 2, "count"),
        ("HARX", "size", "qubits on IonQ", 0, "count"),
        ("QR1", "size", "qubits on IonQ", K_LAGS * N_INPUT + N_QUBITS - N_INPUT, "count"),
        ("HARX", "size", "two-qubit gates", 0, "count"),
        ("QR1", "size", "two-qubit gates", noisy.count_ops()["rxx"], "count"),
        ("HARX", "time", "fit and forecast", float(eff.loc["HARX", "refit_seconds_mean"]), "seconds"),
        ("QR1", "time", "fit the 10 readout weights", timed(lambda: np.linalg.solve(A.T @ A + 1e-8 * np.eye(N_QUBITS), A.T @ y), 2000), "seconds"),
        ("QR1", "time", "one circuit, exact (NumPy)", timed(lambda: month_features(lags, U), 20), "seconds"),
        ("QR1", "time", "one circuit, noisy (Aer density matrix)", timed(lambda: outcome_probabilities(noisy, noise_model=gate_noise_model(0.016)), 3), "seconds"),
        ("QR1", "time", "one circuit, IonQ simulator, 100 shots", ionq[100], "seconds"),
        ("QR1", "time", "one circuit, IonQ simulator, 500 shots", ionq[500], "seconds"),
    ]
    out = pd.DataFrame(rows, columns=["model", "kind", "item", "value", "unit"])
    out.to_csv(ROOT / "data/complexity.csv", index=False)
    print(out.to_string(index=False))
    print(f"\nHARX inputs: {feature_sets['HARX']}")


if __name__ == "__main__":
    main()
