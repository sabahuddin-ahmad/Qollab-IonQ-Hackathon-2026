"""QR1 as a Qiskit circuit (docs/qrc_spec.md, sections 4, 6 and 6a).

Building circuits needs only Qiskit. Local simulation needs qiskit-aer, imported lazily so the
builder also works where Aer is unavailable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from .reference import K_LAGS, N_HIDDEN, N_INPUT, N_QUBITS, TAU

LABEL = "probabilities"


def _evolve(qc: QuantumCircuit, qubits: list[int], J: np.ndarray, r: int, tau: float, step_channel) -> None:
    """Second-order Trotter step, r times. ``qubits[q]`` is the physical qubit carrying logical qubit q."""
    for _ in range(r):
        for q in qubits:
            qc.rz(tau / r, q)
        for a in range(N_QUBITS):
            for b in range(a + 1, N_QUBITS):
                qc.rxx(2 * J[a, b] * tau / r, qubits[a], qubits[b])
        for q in qubits:
            qc.rz(tau / r, q)
        if step_channel is not None:
            for q in qubits:
                qc.append(step_channel, [q])


def qr1_circuit(
    x_lags: np.ndarray,
    J: np.ndarray,
    r: int = 2,
    tau: float = TAU,
    step_channel=None,
    unrolled: bool = False,
    finish: Literal["probabilities", "measure", "none"] = "probabilities",
) -> QuantumCircuit:
    """Circuit for one forecast month. ``x_lags`` is (3, 7): scaled features, oldest month first.

    Default layout is 10 qubits with a reset of the input qubits between months. ``unrolled=True``
    gives every month its own input qubits instead (24 qubits, no reset, same output distribution).
    ``step_channel`` is an optional one-qubit noise instruction applied after every Trotter step.
    ``finish="probabilities"`` saves the exact outcome distribution (Aer); ``"measure"`` adds
    measurements into 10 classical bits, bit q holding logical qubit q.
    """
    x_lags = np.asarray(x_lags, dtype=float)
    if x_lags.shape != (K_LAGS, N_INPUT):
        raise ValueError(f"x_lags must be ({K_LAGS}, {N_INPUT}), got {x_lags.shape}")
    hidden = list(range(K_LAGS * N_INPUT, K_LAGS * N_INPUT + N_HIDDEN)) if unrolled else list(range(N_INPUT, N_QUBITS))
    qc = QuantumCircuit(K_LAGS * N_INPUT + N_HIDDEN if unrolled else N_QUBITS)
    for k, x in enumerate(x_lags):
        inputs = list(range(k * N_INPUT, (k + 1) * N_INPUT)) if unrolled else list(range(N_INPUT))
        if k > 0 and not unrolled:
            qc.reset(inputs)
        for q, value in zip(inputs, x):
            qc.ry(np.pi * value, q)
        _evolve(qc, inputs + hidden, J, r, tau, step_channel)
    out = inputs + hidden  # logical qubits 0..9 after the last month
    if finish == "probabilities":
        import qiskit_aer  # noqa: F401  (registers QuantumCircuit.save_probabilities)

        qc.save_probabilities(out, label=LABEL)
    elif finish == "measure":
        creg = ClassicalRegister(N_QUBITS, "z")
        qc.add_register(creg)
        qc.measure(out, creg)
    return qc


def outcome_probabilities(circuits, noise_model=None, method: str = "density_matrix", **sim_options) -> np.ndarray:
    """Exact outcome distributions, one row of 1024 per circuit, simulated locally on Aer.

    Circuits are run as built (rxx, rz, ry, reset): no transpilation, so gate noise attaches to
    the gates the spec names. Use ``method="statevector"`` for noiseless unrolled circuits.
    Extra keyword arguments go to ``AerSimulator`` (e.g. ``max_parallel_experiments=0`` for batches).
    """
    from qiskit_aer import AerSimulator

    circuits = [circuits] if isinstance(circuits, QuantumCircuit) else list(circuits)
    sim = AerSimulator(method=method, noise_model=noise_model, **sim_options)
    result = sim.run(circuits, shots=1).result()
    return np.array([result.data(i)[LABEL] for i in range(len(circuits))])


def _signs() -> np.ndarray:
    """(10, 1024) of +-1: Z on logical qubit q for each outcome index (bit q of the index)."""
    idx = np.arange(2**N_QUBITS)
    return np.array([1 - 2 * ((idx >> q) & 1) for q in range(N_QUBITS)])


def z_from_probabilities(p: np.ndarray) -> np.ndarray:
    """<Z_q> features from outcome probabilities; works on one distribution or a stack of them."""
    return np.asarray(p) @ _signs().T


def sample_features(p: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Finite-shot estimate of the features for one cached distribution."""
    p = np.clip(np.asarray(p, dtype=float), 0.0, None)
    return z_from_probabilities(rng.multinomial(shots, p / p.sum()) / shots)


def z_from_counts(counts: dict[str, int]) -> np.ndarray:
    """Features from a backend's counts for a ``finish="measure"`` circuit (bit q is the q-th from the right)."""
    shots = sum(counts.values())
    z = np.zeros(N_QUBITS)
    for bits, n in counts.items():
        bits = bits.replace(" ", "")
        z += n * np.array([1 - 2 * int(bits[-1 - q]) for q in range(N_QUBITS)])
    return z / shots
