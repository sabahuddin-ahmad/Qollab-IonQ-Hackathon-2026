"""Transpile QR1 to universal gates with explicit Hadamard and estimate Big-O cost.

Native QR1 uses RY (encoding) + RZ + RXX (Ising evolution) + reset/measure
(src/qrc/circuit.py:qr1_circuit). This script recompiles the exact demo
circuits to the universal set {H, RZ, CX} (+ reset/measure, passed through)
so the Hadamard cost is explicit, then scales one-circuit counts to a full
training job (M months x shots).

Decomposition verified here (opt level 0):
  1x RXX -> 4x H + 2x CX + 1x RZ
  1x RY  -> 2x H + 3x RZ
  1x RZ  -> 1x RZ

Usage: .venv/bin/python scripts/transpile_qr1.py [--r 2] [--shots 1024] [--opt 0]
   Train/test months (M_train/M_test) come from src/harness.py so the
   Big-O below is for the paper's full rolling run, not a guess.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from qiskit.compiler import transpile

from src.qrc.circuit import qr1_circuit
from src.qrc.reference import K_LAGS, N_HIDDEN, N_INPUT, N_QUBITS, couplings

BASIS = ["h", "rz", "cx", "reset", "measure"]


def transpiled_counts(r: int, unrolled: bool, opt: int = 0):
    J = couplings(0)
    rng = np.random.default_rng(0)
    x = rng.random((K_LAGS, N_INPUT))
    native = qr1_circuit(x, J, r=r, unrolled=unrolled, finish="measure")
    t = transpile(native, basis_gates=BASIS, optimization_level=opt)
    return native, t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, default=2, help="Trotter steps (demo uses 2)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--opt", type=int, default=0, help="transpile optimization_level")
    args = ap.parse_args()

    for unrolled in (False, True):
        native, t = transpiled_counts(args.r, unrolled, args.opt)
        ops, tops = dict(native.count_ops()), dict(t.count_ops())
        tag = "24q unrolled (IonQ, no reset)" if unrolled else "10q reset"
        print(f"{tag} r={args.r} native: {ops} depth {native.depth()}")
        print(f"{tag} r={args.r} -> {BASIS[:3]} opt={args.opt}: {tops} depth {t.depth()}")

    # Train/test sizes from the harness (WIN=570 rolling window, F=245 OOS months).
    from src.harness import WIN, iter_folds, load_panel

    full, _, _ = load_panel()
    folds, oos_idx = iter_folds(full)
    train_union: set = set()
    for f in folds:
        train_union.update(f.X_window.index.tolist())
    m_train, m_test, f = len(train_union), len(oos_idx), len(folds)
    m_all = len(set(train_union) | set(oos_idx.tolist()))
    assert (m_train, m_test, f, len(full)) == (803, 245, 245, 804)

    # Big-O estimate from gate algebra (matches the table above to the gate):
    # per month: RXX = K*r*N(N-1)/2 = 135*r, RZ_native = K*r*2N = 60*r, RY = K*N_input = 21
    r = args.r
    n, k, ni = N_QUBITS, K_LAGS, N_INPUT
    d = N_QUBITS  # readout features = 10 <Z> values
    rxx = k * r * n * (n - 1) // 2
    rz_nat = k * r * 2 * n
    ry = k * ni
    cx = 2 * rxx
    h = 4 * rxx + 2 * ry
    rz = rxx + rz_nat + 3 * ry
    print(f"\nPer-month formulas (N={n}, K={k}): RXX=135*r={rxx}, "
          f"CX=2*RXX={cx}, H=4*RXX+2*RY={h}, RZ=RXX+{rz_nat}+3*RY={rz}")
    print(f"Big-O per circuit: O(K*r*N^2) two-qubit (CX={cx}), O(K*r*N^2) single-qubit (H={h}, RZ={rz})")
    s = args.shots
    print(f"Train quantum (M_train={m_train} unique window rows x S={s} shots): "
          f"O(M_train*S*K*r*N^2) = CX~{cx*m_train*s:.2e}, H~{h*m_train*s:.2e}, RZ~{rz*m_train*s:.2e} gate-shots")
    print(f"Test quantum  (M_test={m_test} OOS months x S={s} shots): "
          f"O(M_test*S*K*r*N^2) = CX~{cx*m_test*s:.2e}, H~{h*m_test*s:.2e}, RZ~{rz*m_test*s:.2e} gate-shots")
    print(f"Full unique job ({m_all} circuits, test overlaps train windows): O((M_train+M_test-F+1)*S*K*r*N^2)")
    print(f"Train classical: O(P) scaling + O(M_train*2^N*N) counts-><Z> + "
          f"O(F*WIN*d^2+F*d^3) ridge fits ({f}*{WIN}*{d}^2+{f}*{d}^3)")
    print(f"Test classical: O(M_test*2^N*N) counts-><Z> + O(M_test*d) predict + O(M_test) score")
    print("Note: IonQ native is RXX (M-S); H/CX form is for cost accounting only, do not submit it.")


if __name__ == "__main__":
    main()
