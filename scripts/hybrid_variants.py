"""Experiment: four ways to get more out of the reservoir, each tested in a hybrid with HARX.

  correlators   also read <Z_i Z_j> on all 45 pairs (free on a device: same measured bitstrings)
  multiplex     read <Z> at tau/2 and at tau after the last month (temporal multiplexing, 20 values)
  bases         also read <X> and <Y> on every qubit (30 values; two more circuits per forecast)
  hamiltonian   Hamiltonian encoding: no RY and no reset; each month's features set the Z field on the
                seven input qubits (h_q = pi * x_q), three months applied in turn from |0...0>, <Z> read

Each feature set is scored two ways on 245 months, exact simulation: added to HARX's 13 inputs in one
readout (ridge on the reservoir columns only), and on its own with QR1's readout. Ridge is chosen on
the 120 months before the test period. Every variant is also run with the entangling gates off (J = 0),
which leaves only classical functions of the inputs: the control for "is the quantum dynamics helping".
With J = 0 the Hamiltonian encoding returns constants, so it has no such control.

Writes data/qr1_hybrid_variants.csv.

Usage: .venv/bin/python scripts/hybrid_variants.py [--seeds 10] [--workers 10]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

from hybrid_qrc import rolling
from src.harness import iter_folds, load_panel
from src.qrc.reference import K_LAGS, N_HIDDEN, N_INPUT, N_QUBITS, _input_state, _z_signs, couplings, evolution
from sweep_qrc import N_VALIDATION, RIDGES, choose_ridge, load_inputs, rolling_predictions

SIGNS = _z_signs()
PAIRS = [(a, b) for a in range(N_QUBITS) for b in range(a + 1, N_QUBITS)]
ZZ = np.array([SIGNS[a] * SIGNS[b] for a, b in PAIRS])


def _xy(probs: np.ndarray, states: np.ndarray) -> np.ndarray:
    out = []
    for q in range(N_QUBITS):
        s = states.reshape(len(states), 2**q, 2, 2 ** (N_QUBITS - 1 - q))
        c = np.einsum("m,mah,mah->", probs, s[:, :, 0].conj(), s[:, :, 1])
        out.append((2 * c.real, 2 * c.imag))
    return np.array(out).T.ravel()  # ten <X>, then ten <Y>


def reset_reservoir(X: np.ndarray, J: np.ndarray) -> dict[str, np.ndarray]:
    """QR1's three-month restart, returning every observable set for each month."""
    half = evolution(J, 0.5)
    U = half @ half
    names = {"z": 10, "zz": 45, "z_half": 10, "xy": 20}
    out = {k: np.full((len(X), n), np.nan) for k, n in names.items()}
    for t in range(K_LAGS, len(X)):
        lags = X[t - K_LAGS : t]
        if not np.isfinite(lags).all():
            continue
        probs, hidden = np.array([1.0]), np.eye(1, 2**N_HIDDEN, dtype=complex)
        for k, x in enumerate(lags):
            start = np.stack([np.kron(_input_state(x), h) for h in hidden])
            if k == K_LAGS - 1:
                mid = start @ half.T
                states = mid @ half.T
                p = np.einsum("m,ms->s", probs, np.abs(states) ** 2)
                out["z"][t], out["zz"][t] = SIGNS @ p, ZZ @ p
                out["z_half"][t] = SIGNS @ np.einsum("m,ms->s", probs, np.abs(mid) ** 2)
                out["xy"][t] = _xy(probs, states)
                break
            states = start @ U.T
            m = states.reshape(len(states), 2**N_INPUT, 2**N_HIDDEN)
            w, v = np.linalg.eigh(np.einsum("m,mih,mig->hg", probs, m, m.conj()))
            keep = w > 1e-14
            probs, hidden = w[keep], v[:, keep].T
    return out


def hamiltonian_reservoir(X: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Inputs enter as Z-field strengths; the state stays pure and is rebuilt from |0...0> each month."""
    x_op = sp.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    eye = sp.identity(2, dtype=complex, format="csr")

    def on(q):
        m = sp.identity(1, dtype=complex, format="csr")
        for k in range(N_QUBITS):
            m = sp.kron(m, x_op if k == q else eye, format="csr")
        return m

    xs = [on(q) for q in range(N_QUBITS)]
    xx = sum(J[a, b] * xs[a] @ xs[b] for a, b in PAIRS)
    out = np.full((len(X), N_QUBITS), np.nan)
    zero = np.zeros(2**N_QUBITS, dtype=complex)
    zero[0] = 1.0
    for t in range(K_LAGS, len(X)):
        lags = X[t - K_LAGS : t]
        if not np.isfinite(lags).all():
            continue
        psi = zero
        for x in lags:
            field = np.r_[np.pi * x, np.ones(N_HIDDEN)] @ SIGNS
            psi = expm_multiply(-1j * (xx + sp.diags(field)), psi)
        out[t] = SIGNS @ np.abs(psi) ** 2
    return out


def feature_sets(X: np.ndarray, J: np.ndarray, with_hamiltonian: bool) -> dict[str, np.ndarray]:
    r = reset_reservoir(X, J)
    sets = {"baseline (10 <Z>)": r["z"], "correlators (10 <Z> + 45 <ZZ>)": np.hstack([r["z"], r["zz"]]),
            "correlators only (45 <ZZ>)": r["zz"], "multiplex (<Z> at tau/2 and tau)": np.hstack([r["z_half"], r["z"]]),
            "bases (<Z>, <X>, <Y>)": np.hstack([r["z"], r["xy"]])}
    if with_hamiltonian:
        sets["hamiltonian encoding (10 <Z>)"] = hamiltonian_reservoir(X, J)
    return sets


CTX: dict = {}


def setup() -> None:
    panel, X, target = load_inputs()
    full, fs, _ = load_panel()
    _, oos_index = iter_folds(full)
    y = full["y"].to_numpy()
    oos = full.index.get_indexer(oos_index)
    CTX.update(X=X, target=target, rows=panel.index.get_indexer(full.index), oos=oos, y=y, y_scaled=target.scale(y),
               H=np.column_stack([np.ones(len(full)), full[fs["HARX"]].to_numpy()]), val=np.arange(oos[0] - N_VALIDATION, oos[0]))


def score(E: np.ndarray) -> tuple[float, float]:
    """(hybrid with HARX, stand-alone) out-of-sample MSE for one feature matrix."""
    c = CTX
    E = E[:, E[c["oos"]].std(axis=0) > 1e-9]  # constant columns duplicate the intercept
    n_h = c["H"].shape[1]
    F = np.column_stack([c["H"], np.nan_to_num(E)])
    pen = lambda k: np.r_[np.zeros(n_h), np.full(E.shape[1], k)]
    best = min(RIDGES, key=lambda k: np.mean((rolling(F, c["y"], c["val"], pen(k)) - c["y"][c["val"]]) ** 2))
    hybrid = np.mean((rolling(F, c["y"], c["oos"], pen(best)) - c["y"][c["oos"]]) ** 2)
    S = np.column_stack([np.nan_to_num(E), np.ones(len(E))])
    k = choose_ridge(S, c["y_scaled"], c["oos"][0])
    alone = np.mean((c["target"].unscale(rolling_predictions(S, c["y_scaled"], c["oos"], k)) - c["y"][c["oos"]]) ** 2)
    return float(hybrid), float(alone)


def work(seed: int) -> list[dict]:
    """seed -1 is the gates-off control."""
    if not CTX:
        setup()
    J = np.zeros((N_QUBITS, N_QUBITS)) if seed < 0 else couplings(seed)
    sets = feature_sets(CTX["X"], J, with_hamiltonian=seed >= 0)
    rows = []
    for name, E in sets.items():
        hybrid, alone = score(E[CTX["rows"]])
        rows.append({"seed": seed, "variant": name, "n_features": E.shape[1], "hybrid_mse": hybrid, "alone_mse": alone})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()
    with Pool(args.workers) as pool:
        out = pd.DataFrame([r for rows in pool.map(work, [-1, *range(args.seeds)]) for r in rows])
    out.to_csv(ROOT / "data/qr1_hybrid_variants.csv", index=False, float_format="%.6f")
    on, off = out[out.seed >= 0], out[out.seed < 0].set_index("variant")
    order = list(dict.fromkeys(out.variant))
    print(f"HARX alone 0.1030; HARX + cos(pi x) of the seven inputs 0.0999; QR1 alone about 0.111.  {args.seeds} seeds.\n")
    print(f"{'variant':36s} {'n':>3s} | {'HYBRID mean':>11s} {'best':>7s} {'worst':>7s} {'gates off':>9s} {'<HARX':>6s} {'<off':>5s} | {'ALONE mean':>10s} {'gates off':>9s}")
    for v in order:
        g = on[on.variant == v]
        o = off.loc[v] if v in off.index else None
        h_off = f"{o.hybrid_mse:.4f}" if o is not None else "n/a"
        a_off = f"{o.alone_mse:.4f}" if o is not None else "n/a"
        beat_off = f"{(g.hybrid_mse < o.hybrid_mse).sum()}/{len(g)}" if o is not None else "n/a"
        print(f"{v:36s} {int(g.n_features.iloc[0]):3d} | {g.hybrid_mse.mean():11.4f} {g.hybrid_mse.min():7.4f} {g.hybrid_mse.max():7.4f} {h_off:>9s} "
              f"{f'{(g.hybrid_mse < 0.1030).sum()}/{len(g)}':>6s} {beat_off:>5s} | {g.alone_mse.mean():10.4f} {a_off:>9s}")
    print("\nsaved data/qr1_hybrid_variants.csv")


if __name__ == "__main__":
    main()
