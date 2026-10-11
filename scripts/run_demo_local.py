"""Run main.py the way Qollab does, on this machine: define ``backend``, run the file, call main().

Default backend is Qiskit's BasicSimulator, the class behind Qollab's built-in simulator.
``--as-ionq`` exercises the 24-qubit no-reset path that IonQ backends get, simulated locally
on Aer under an IonQ-like name. It checks the code path only; it is not IonQ's simulator.

Usage: .venv/bin/python scripts/run_demo_local.py [--shots 1024] [--month 2017-12] [--all-three-months] [--as-ionq] [--mid-circuit]
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _AerAsIonq:
    name = "ionq_simulator (emulated locally on Aer, ideal)"

    def run(self, circuits, shots):
        from qiskit_aer import AerSimulator

        if isinstance(circuits, (list, tuple)):  # what Qollab's IonQ backends do with a list
            raise RuntimeError("Only job type ionq.circuit.v1 is supported")

        return AerSimulator(method="statevector").run(circuits, shots=shots)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--month", default="2017-12")
    ap.add_argument("--all-three-months", action="store_true")
    ap.add_argument("--as-ionq", action="store_true")
    ap.add_argument("--mid-circuit", action="store_true")
    args = ap.parse_args()

    if args.as_ionq:
        backend = _AerAsIonq()
    else:
        from qiskit.providers.basic_provider import BasicSimulator

        backend = BasicSimulator()

    os.chdir(ROOT)  # main.py reads demo/qr1_demo.json relative to the project root
    sys.path.insert(0, str(ROOT))  # and imports src.* from it, as on Qollab
    scope = {"backend": backend, "__name__": "qollab_main"}
    exec(compile((ROOT / "main.py").read_text(), "main.py", "exec"), scope)
    scope["main"](shots=args.shots, month=args.month, all_three_months=args.all_three_months, mid_circuit=args.mid_circuit)


if __name__ == "__main__":
    main()
