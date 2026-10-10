"""Exact QR1 reservoir: the ground truth the circuit is checked against (docs/qrc_spec.md, sections 3-4).

Pure NumPy. Qubits 0-6 are inputs, 7-9 hidden; qubit 0 is the first tensor factor.
"""

from __future__ import annotations

import numpy as np

N_QUBITS, N_INPUT, K_LAGS, TAU = 10, 7, 3, 1.0
N_HIDDEN = N_QUBITS - N_INPUT

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_I = np.eye(2, dtype=complex)


def couplings(seed: int, n: int = N_QUBITS) -> np.ndarray:
    """Random symmetric couplings with zero diagonal, scaled so the largest eigenvalue is 1."""
    m = np.random.default_rng(seed).random((n, n))
    m = (m + m.T) / 2
    np.fill_diagonal(m, 0.0)
    return m / np.linalg.eigvalsh(m).max()


def _on_qubit(single: np.ndarray, q: int) -> np.ndarray:
    out = np.array([[1]], dtype=complex)
    for k in range(N_QUBITS):
        out = np.kron(out, single if k == q else _I)
    return out


def _z_signs() -> np.ndarray:
    """(10, 1024) array of +-1: the diagonal of Z on each qubit."""
    idx = np.arange(2**N_QUBITS)
    return np.array([1 - 2 * ((idx >> (N_QUBITS - 1 - q)) & 1) for q in range(N_QUBITS)])


def _xx_part(J: np.ndarray) -> np.ndarray:
    xs = [_on_qubit(_X, q) for q in range(N_QUBITS)]
    return sum(J[i, j] * xs[i] @ xs[j] for i in range(N_QUBITS) for j in range(i + 1, N_QUBITS))


def _exp(h: np.ndarray, t: float) -> np.ndarray:
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * t * w)) @ v.conj().T


def evolution(J: np.ndarray, tau: float = TAU) -> np.ndarray:
    """exp(-i H tau) for H = sum_{i<j} J_ij X_i X_j + sum_i Z_i."""
    return _exp(_xx_part(J) + np.diag(_z_signs().sum(axis=0)).astype(complex), tau)


def trotter_evolution(J: np.ndarray, r: int, tau: float = TAU) -> np.ndarray:
    """Second-order Trotter approximation with r steps: the unitary the gate circuit implements."""
    half_z = np.diag(np.exp(-1j * tau / (2 * r) * _z_signs().sum(axis=0)))
    return np.linalg.matrix_power(half_z @ _exp(_xx_part(J), tau / r) @ half_z, r)


def _input_state(x: np.ndarray) -> np.ndarray:
    psi = np.array([1.0 + 0j])
    for v in x:
        psi = np.kron(psi, np.array([np.cos(np.pi * v / 2), np.sin(np.pi * v / 2)], dtype=complex))
    return psi


def month_features(x_lags: np.ndarray, U: np.ndarray) -> np.ndarray:
    """<Z_q> on all 10 qubits after feeding three months of features, oldest first. ``x_lags`` is (3, 7)."""
    probs = np.array([1.0])
    hidden = np.zeros((1, 2**N_HIDDEN), dtype=complex)
    hidden[0, 0] = 1.0  # |000>
    for k, x in enumerate(x_lags):
        states = np.stack([U @ np.kron(_input_state(x), h) for h in hidden])
        if k == len(x_lags) - 1:
            return _z_signs() @ np.einsum("m,ms->s", probs, np.abs(states) ** 2)
        # discard the inputs: reduced state of the hidden qubits, kept as an eigen-ensemble
        m = states.reshape(len(states), 2**N_INPUT, 2**N_HIDDEN)
        w, v = np.linalg.eigh(np.einsum("m,mih,mig->hg", probs, m, m.conj()))
        keep = w > 1e-14
        probs, hidden = w[keep], v[:, keep].T


def reservoir_features(X: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Features for every month. Row t uses rows t-3..t-1 of ``X`` only; rows without full lags are NaN."""
    out = np.full((len(X), N_QUBITS), np.nan)
    for t in range(K_LAGS, len(X)):
        lags = X[t - K_LAGS : t]
        if np.isfinite(lags).all():
            out[t] = month_features(lags, U)
    return out
