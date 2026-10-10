"""Noise for local Aer simulation of QR1 (docs/qrc_spec.md, section 6a).

Generic, uniform, memoryless noise. This is not IonQ's device noise model.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    amplitude_damping_error,
    depolarizing_error,
    phase_damping_error,
)

from .reference import TAU


def gate_noise_model(p2: float, p1: float | None = None) -> NoiseModel | None:
    """Depolarizing error ``p2`` on every RXX and ``p1`` (default p2/10) on every RY; RZ is noiseless."""
    if p2 == 0:
        return None
    model = NoiseModel(basis_gates=["rxx", "rz", "ry", "reset"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rxx"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2 / 10 if p1 is None else p1, 1), ["ry"])
    return model


def step_channel(
    kind: Literal["amplitude_damping", "depolarizing", "phase_damping"],
    gamma: float,
    r: int,
    tau: float = TAU,
):
    """One-qubit channel to apply after each Trotter step, for dissipation rate ``gamma``.

    Strength per step is ``1 - exp(-2 * gamma * tau / r)``, so the total over a month's evolution
    does not depend on r. Phase damping commutes with the Z readout and has almost no effect.
    """
    if gamma == 0:
        return None
    strength = 1.0 - np.exp(-2.0 * gamma * tau / r)
    if kind == "amplitude_damping":
        error = amplitude_damping_error(strength)
    elif kind == "depolarizing":
        error = depolarizing_error(strength, 1)
    elif kind == "phase_damping":
        error = phase_damping_error(strength)
    else:
        raise ValueError(f"unknown channel kind: {kind}")
    return error.to_instruction()
