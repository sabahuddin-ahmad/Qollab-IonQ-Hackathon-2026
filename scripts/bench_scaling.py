"""Measured local simulation time against qubit count, for a QR1-shaped circuit of any size.

Same structure as QR1 (3 months, input qubits reset between months, 2 second-order Trotter steps,
RXX on every pair, 3 memory qubits) with random inputs and couplings. Simulated on Aer's density
matrix with 1.6% depolarizing error per RXX, the way the noise study was. Writes data/scaling_qubits.csv.

Usage: .venv/bin/python scripts/bench_scaling.py [--max-qubits 13]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from src.qrc.noise import gate_noise_model

MONTHS, STEPS, MEMORY = 3, 2, 3


def circuit(n: int, rng: np.random.Generator) -> QuantumCircuit:
    inputs = list(range(n - MEMORY))
    J = rng.random((n, n))
    qc = QuantumCircuit(n)
    for k in range(MONTHS):
        if k:
            qc.reset(inputs)
        for q in inputs:
            qc.ry(np.pi * rng.uniform(-1, 1), q)
        for _ in range(STEPS):
            qc.rz(1 / STEPS, range(n))
            for a in range(n):
                for b in range(a + 1, n):
                    qc.rxx(2 * J[a, b] / STEPS, a, b)
            qc.rz(1 / STEPS, range(n))
    qc.save_probabilities()
    return qc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-qubits", type=int, default=13)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    sim = AerSimulator(method="density_matrix", noise_model=gate_noise_model(0.016))
    rows = []
    for n in range(4, args.max_qubits + 1):
        qc = circuit(n, rng)
        best = np.inf
        for _ in range(3 if n <= 11 else 1):
            t0 = time.perf_counter()
            sim.run(qc, shots=1).result()
            best = min(best, time.perf_counter() - t0)
        rows.append({"qubits": n, "rxx_gates": qc.count_ops()["rxx"], "seconds": best, "density_matrix_bytes": 16 * 4**n})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "data/scaling_qubits.csv", index=False)
    print("saved data/scaling_qubits.csv")


if __name__ == "__main__":
    main()
