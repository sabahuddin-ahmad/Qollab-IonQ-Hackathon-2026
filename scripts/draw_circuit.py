"""Draw the QR1 circuit for the project page (docs/qrc_spec.md, sections 4 and 6).

Writes three figures to docs/figures/:
  qr1_circuit.png       the whole circuit for one forecast month, each evolution as one block
  qr1_circuit_24q.png   the same forecast without reset (24 qubits), the form IonQ backends get
  qr1_trotter_step.png  the gates inside one block: a single second-order Trotter step

Black-and-white style so nothing depends on colour.

Usage: uv run --with matplotlib --with pylatexenc python scripts/draw_circuit.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter

from src.qrc.features import FEATURES
from src.qrc.reference import K_LAGS, N_HIDDEN, N_INPUT, N_QUBITS

OUT = ROOT / "docs" / "figures"


def overview() -> QuantumCircuit:
    """One forecast month t: three months of features in, ten <Z> values out."""
    inp, hid = QuantumRegister(N_INPUT, "input"), QuantumRegister(N_HIDDEN, "hidden")
    out = ClassicalRegister(N_QUBITS, "z")
    qc = QuantumCircuit(inp, hid, out)
    evolve = Gate("evolve", N_QUBITS, [], label="exp(-i H τ)")
    for k in range(K_LAGS, 0, -1):  # oldest month first
        if k < K_LAGS:
            qc.reset(inp)
        for q, name in enumerate(FEATURES):
            qc.ry(Parameter(f"π·{name}(t-{k})"), inp[q])
        qc.append(evolve, [*inp, *hid])
        qc.barrier()
    qc.measure([*inp, *hid], out)
    return qc


def overview_unrolled() -> QuantumCircuit:
    """The same forecast without reset, as sent to IonQ backends: each month gets its own input qubits."""
    months = [QuantumRegister(N_INPUT, f"input(t-{k})") for k in range(K_LAGS, 0, -1)]  # oldest first
    hid = QuantumRegister(N_HIDDEN, "hidden")
    out = ClassicalRegister(N_QUBITS, "z")
    qc = QuantumCircuit(*months, hid, out)
    evolve = Gate("evolve", N_QUBITS, [], label="exp(-i H τ)")
    for k, inp in zip(range(K_LAGS, 0, -1), months):
        for q, name in enumerate(FEATURES):
            qc.ry(Parameter(f"π·{name}(t-{k})"), inp[q])
        qc.append(evolve, [*inp, *hid])
        qc.barrier()
    qc.measure([*months[-1], *hid], out)  # earlier months' input qubits are left unmeasured
    return qc


def overview_stream() -> QuantumCircuit:
    """The continuous-stream variant (scripts/stream_qrc.py): the hidden qubits are never reset.

    Shown for the last three months of the stream. After each month the input qubits are measured
    and reset; measuring them does not disturb the hidden qubits' state, so it is the same as
    discarding them. Not runnable on Qollab's IonQ backend, which rejects reset.
    """
    inp, hid = QuantumRegister(N_INPUT, "input"), QuantumRegister(N_HIDDEN, "hidden")
    mids = [ClassicalRegister(N_INPUT, f"mid(t-{k})") for k in range(K_LAGS, 1, -1)]
    out = ClassicalRegister(N_QUBITS, "z")
    qc = QuantumCircuit(inp, hid, *mids, out)
    qc.append(Gate("memory", N_HIDDEN, [], label="state left by\nall earlier months"), hid)
    evolve = Gate("evolve", N_QUBITS, [], label="exp(-i H τ)")
    for i, k in enumerate(range(K_LAGS, 0, -1)):
        for q, name in enumerate(FEATURES):
            qc.ry(Parameter(f"π·{name}(t-{k})"), inp[q])
        qc.append(evolve, [*inp, *hid])
        qc.barrier()
        if k > 1:
            qc.measure(inp, mids[i])
            qc.reset(inp)
            qc.barrier()
    qc.measure([*inp, *hid], out)
    return qc


def trotter_step() -> QuantumCircuit:
    """One second-order Trotter step of exp(-i H τ/r): RZ on all, RXX on all 45 pairs, RZ on all.

    RXX angles are labelled θab, meaning 2·J_ab·τ/r.
    """
    q = QuantumRegister(N_QUBITS, "q")
    qc = QuantumCircuit(q)
    half = Parameter("τ/r")
    for i in range(N_QUBITS):
        qc.rz(half, q[i])
    qc.barrier()
    for a in range(N_QUBITS):
        for b in range(a + 1, N_QUBITS):
            qc.rxx(Parameter(f"θ{a}{b}"), q[a], q[b])  # θab = 2·J_ab·τ/r
    qc.barrier()
    for i in range(N_QUBITS):
        qc.rz(half, q[i])
    return qc


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig = overview().draw("mpl", style="bw", fold=-1, scale=0.8, plot_barriers=True)
    fig.savefig(OUT / "qr1_circuit.png", dpi=170, bbox_inches="tight", facecolor="white")
    fig = overview_unrolled().draw("mpl", style="bw", fold=-1, scale=0.7, plot_barriers=True)
    fig.savefig(OUT / "qr1_circuit_24q.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig = overview_stream().draw("mpl", style="bw", fold=-1, scale=0.7, plot_barriers=True)
    fig.savefig(OUT / "qr1_circuit_stream.png", dpi=150, bbox_inches="tight", facecolor="white")
    fig = trotter_step().draw("mpl", style="bw", fold=26, scale=0.7)
    fig.savefig(OUT / "qr1_trotter_step.png", dpi=150, bbox_inches="tight", facecolor="white")
    print("saved qr1_circuit.png, qr1_circuit_24q.png and qr1_trotter_step.png in", OUT)


if __name__ == "__main__":
    main()
