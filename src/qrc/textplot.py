"""Plain-text circuit diagram and charts for the Qollab console, which may not show images.

Only ASCII characters in the charts, so they line up in any monospaced output panel.
"""

from __future__ import annotations

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate

from .reference import K_LAGS, N_HIDDEN, N_INPUT, N_QUBITS

# Same order as src/qrc/features.py, repeated here so the Qollab run does not need pandas.
FEATURES = ["RV", "MKT", "DP", "IP", "RV_q", "STR", "DEF"]


def circuit_diagram(unrolled: bool, steps: int, mid_circuit: bool = False, width: int = 150) -> str:
    """The forecast circuit with each evolution drawn as one block, as text.

    The real circuit has the same structure; every "evolve" block stands for ``steps`` Trotter steps,
    each an RZ on all ten qubits, an RXX on all 45 pairs and another RZ.
    """
    evolve = Gate("evolve", N_QUBITS, [], label="evolve")
    load = lambda name: Gate("ry", 1, [], label=f"RY {name}")
    mem = QuantumRegister(N_HIDDEN, "mem")
    if unrolled:
        months = [QuantumRegister(N_INPUT, f"t-{k}") for k in range(K_LAGS, 0, -1)]
        creg = ClassicalRegister(N_QUBITS + (K_LAGS - 1) * N_INPUT if mid_circuit else N_QUBITS, "c")
        qc = QuantumCircuit(*months, mem, creg)
    else:
        months = [QuantumRegister(N_INPUT, "in")] * K_LAGS
        creg = ClassicalRegister(N_QUBITS, "c")
        qc = QuantumCircuit(months[0], mem, creg)
    for k, inp in enumerate(months):
        if k > 0 and not unrolled:
            qc.reset(inp)
        for q, name in enumerate(FEATURES):
            qc.append(load(name), [inp[q]])
        qc.append(evolve, [*inp, *mem])
        if unrolled and mid_circuit and k < K_LAGS - 1:
            first = N_QUBITS + (K_LAGS - 2 - k) * N_INPUT
            qc.measure(inp, creg[first : first + N_INPUT])
        qc.barrier()
    qc.measure([*months[-1], *mem], creg[:N_QUBITS])
    return str(qc.draw("text", fold=width, plot_barriers=False, cregbundle=True, vertical_compression="high"))


def z_chart(labels: list[str], series: dict[str, np.ndarray], half: int = 24) -> str:
    """One row per qubit on a -1..+1 axis. ``series`` maps a one-character marker to ten values;
    later entries overwrite earlier ones where they coincide."""
    lines = [f"{'':9s}-1{'':{half - 2}s}0{'':{half - 2}s}+1"]
    for i, label in enumerate(labels):
        row = [" "] * (2 * half + 1)
        row[half] = "|"
        for mark, values in series.items():
            row[int(round((float(np.clip(values[i], -1, 1)) + 1) * half))] = mark
        lines.append(f"{label:8s} " + "".join(row))
    return "\n".join(lines)


def bar_chart(rows: list[tuple[str, float]], lo: float, hi: float, width: int = 40, fmt: str = "{:.4f}", mark: str = "#") -> str:
    """Horizontal bars from ``lo`` to each value, on a shared scale ending at ``hi``."""
    name_width = max(len(name) for name, _ in rows)
    out = []
    for name, value in rows:
        n = int(round((min(max(value, lo), hi) - lo) / (hi - lo) * width))
        out.append(f"  {name:{name_width}s} {mark * n:{width}s} {fmt.format(value)}")
    return "\n".join(out)
