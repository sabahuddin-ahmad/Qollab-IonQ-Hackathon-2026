"""Univariate comparison: every model sees past realized volatility only, no macro variables.

Classical side (from data/oos_classical.csv): AR1, AR3, HAR and the classical reservoir RC. Without
macro variables HARX is HAR and ARMAX is AR3, so those two are the univariate HARX and ARMAX.

Quantum side: QR1's reservoir (same Hamiltonian, RY encoding, reset between months, <Z> readout)
fed volatility only. Four designs, fixed before looking at the test months:

  A  1 input qubit (monthly log RV) + 3 memory qubits, 3 months fed in       4 qubits
  B  as A with 12 months fed in                                              4 qubits
  C  3 input qubits (monthly, quarterly, annual log RV: HAR's inputs) + 3 memory, 3 months   6 qubits
  D  as C with 7 memory qubits                                               10 qubits

Inputs are min-max scaled to [-1, 0] over the whole sample, as QR1 scales RV. The readout is a
ridge regression on the <Z> values plus an intercept, penalty chosen on the 120 months before the
test period. Each design is also run with the entangling gates off (J = 0), where it reduces to
cos(pi x) of last month's inputs: a classical transform.

With --lags the comparison is matched lag for lag instead: for K = 1, 2, 3, 6, 12, an autoregression
on the last K months of log RV against a reservoir given exactly those K values and nothing else,
in two layouts: sequential (1 input qubit + 3 memory, the K months fed in turn) and parallel
(K input qubits, one per lag, + 3 memory, loaded at once; K <= 6).

Writes data/univariate.csv and data/oos_univariate.csv, or data/univariate_lags.csv with --lags.

Usage: .venv/bin/python scripts/bench_univariate.py [--seeds 20] [--lags]
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from hybrid_qrc import dm_hac
from src.harness import iter_folds, load_panel
from src.qrc.features import TargetScale
from src.qrc.reference import couplings
from sweep_qrc import choose_ridge, rolling_predictions

DESIGNS = {"A": (["m"], 3, 3), "B": (["m"], 3, 12), "C": (["m", "q", "a"], 3, 3), "D": (["m", "q", "a"], 7, 3)}  # inputs, memory, months
LABEL = {"A": "1 input + 3 memory, 3 months", "B": "1 input + 3 memory, 12 months",
         "C": "3 inputs + 3 memory, 3 months", "D": "3 inputs + 7 memory, 3 months"}


def unitary(J: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """exp(-i H tau), H = sum_{i<j} J_ij X_i X_j + sum_i Z_i, for any number of qubits."""
    n = len(J)
    x, eye = np.array([[0, 1], [1, 0]], dtype=complex), np.eye(2, dtype=complex)

    def on(q):
        m = np.array([[1]], dtype=complex)
        for k in range(n):
            m = np.kron(m, x if k == q else eye)
        return m

    xs = [on(q) for q in range(n)]
    h = sum(J[a, b] * xs[a] @ xs[b] for a in range(n) for b in range(a + 1, n)) + np.diag(signs(n).sum(axis=0)).astype(complex)
    w, v = np.linalg.eigh(h)
    return (v * np.exp(-1j * tau * w)) @ v.conj().T


def signs(n: int) -> np.ndarray:
    idx = np.arange(2**n)
    return np.array([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])


def reservoir(X: np.ndarray, U: np.ndarray, n_in: int, months: int) -> np.ndarray:
    """Row t: <Z> on every qubit after feeding rows t-months..t-1 of X, oldest first, inputs reset in between."""
    n = int(np.log2(len(U)))
    n_mem, s = n - n_in, signs(n)
    out = np.full((len(X), n), np.nan)
    for t in range(months, len(X)):
        lags = X[t - months : t]
        if not np.isfinite(lags).all():
            continue
        probs, hidden = np.array([1.0]), np.eye(1, 2**n_mem, dtype=complex)
        for k, x in enumerate(lags):
            psi = np.array([1.0 + 0j])
            for v in x:
                psi = np.kron(psi, [np.cos(np.pi * v / 2), np.sin(np.pi * v / 2)])
            states = np.stack([np.kron(psi, h) for h in hidden]) @ U.T
            if k == months - 1:
                out[t] = s @ np.einsum("m,ms->s", probs, np.abs(states) ** 2)
                break
            m = states.reshape(len(states), 2**n_in, 2**n_mem)
            w, v = np.linalg.eigh(np.einsum("m,mih,mig->hg", probs, m, m.conj()))
            keep = w > 1e-14
            probs, hidden = w[keep], v[:, keep].T
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--lags", action="store_true", help="lag-for-lag comparison against AR(K)")
    args = ap.parse_args()

    panel = pd.read_csv(ROOT / "data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()
    full, _, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y = full["y"].to_numpy()
    rv = panel["RV_log"]
    target = TargetScale(float(rv.min()), float(rv.max()))
    y_scaled = target.scale(y)
    scale01 = lambda s: (s - s.min()) / (s.max() - s.min()) - 1.0
    inputs = pd.DataFrame({"m": scale01(rv), "q": scale01(rv.rolling(3).mean()), "a": scale01(rv.rolling(12).mean())})
    truth = y[oos]
    mse = lambda p: float(np.mean((p - truth) ** 2))

    def forecast(Z):
        Z = Z[rows]
        Z = Z[:, np.nanstd(Z[oos], axis=0) > 1e-9]  # constant columns duplicate the intercept
        S = np.column_stack([np.nan_to_num(Z), np.ones(len(Z))])
        assert np.isfinite(Z[oos[0] - 240 :]).all()
        k = choose_ridge(S, y_scaled, oos[0])
        return target.unscale(rolling_predictions(S, y_scaled, oos, k)), S.shape[1]

    classical = pd.read_csv(ROOT / "data/oos_classical.csv")
    assert np.allclose(classical["y_log"], truth)
    if args.lags:
        return lag_comparison(args.seeds, panel, full, rows, oos, y, inputs["m"].to_numpy(), forecast, mse, truth)
    out = pd.DataFrame({"y_log": truth}, index=pd.Index(oos_index, name="date"))
    records = []
    for name, weights, note in (("AR1", 2, ""), ("AR3", 4, "univariate ARMAX"), ("HAR", 4, "univariate HARX"), ("RC", 51, "classical reservoir, forecast averaged over 8 seeds")):
        records.append({"model": name, "kind": "classical", "weights": weights, "mse": mse(classical[name].to_numpy()), "note": note})
    persistence = y[oos - 1]
    records.append({"model": "Persistence", "kind": "classical", "weights": 0, "mse": mse(persistence), "note": "last month's value"})

    har = classical["HAR"].to_numpy()
    for key, (cols, n_mem, months) in DESIGNS.items():
        X = inputs[cols].to_numpy()
        n = len(cols) + n_mem
        off, k_off = forecast(reservoir(X, unitary(np.zeros((n, n))), len(cols), months))
        preds, k_on = [], 0
        for seed in range(args.seeds):
            p, k_on = forecast(reservoir(X, unitary(couplings(seed, n)), len(cols), months))
            preds.append(p)
            out[f"{key}_s{seed}"] = p
        per_seed = np.array([mse(p) for p in preds])
        avg = np.mean(preds, axis=0)
        t, pv = dm_hac(avg - truth, har - truth)
        records.append({"model": f"Quantum reservoir {key}: {LABEL[key]}", "kind": "quantum", "weights": k_on, "mse": per_seed.mean(),
                        "mse_sd": per_seed.std(), "mse_best": per_seed.min(), "mse_worst": per_seed.max(), "mse_seed_averaged": mse(avg),
                        "mse_gates_off": mse(off), "weights_gates_off": k_off, "seeds_beating_gates_off": int((per_seed < mse(off)).sum()),
                        "seeds_beating_har": int((per_seed < mse(har)).sum()), "seeds_beating_rc": int((per_seed < mse(classical["RC"].to_numpy())).sum()),
                        "dm_p_vs_har": pv, "note": f"{n} qubits"})
        print(f"design {key} done", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(ROOT / "data/univariate.csv", index=False, float_format="%.6f")
    out.to_csv(ROOT / "data/oos_univariate.csv", float_format="%.6f")
    pd.set_option("display.width", 250)
    print("\n" + res[res.kind == "classical"][["model", "weights", "mse", "note"]].to_string(index=False))
    q = res[res.kind == "quantum"]
    print("\n" + q[["model", "weights", "mse", "mse_sd", "mse_best", "mse_worst", "mse_seed_averaged", "mse_gates_off", "seeds_beating_gates_off",
                    "seeds_beating_har", "seeds_beating_rc", "dm_p_vs_har"]].round(4).to_string(index=False))
    print("\nsaved data/univariate.csv and data/oos_univariate.csv")


def lag_comparison(n_seeds, panel, full, rows, oos, y, x, forecast, mse, truth) -> None:
    from hybrid_qrc import rolling

    rv = panel["RV_log"]
    records = []
    for K in (1, 2, 3, 6, 12):
        L = np.column_stack([rv.shift(k).to_numpy() for k in range(1, K + 1)])[rows]
        F = np.column_stack([np.ones(len(full)), L])
        ar = rolling(F, y, oos, np.zeros(K + 1))
        row = {"lags": K, "ar_weights": K + 1, "ar_mse": mse(ar)}
        layouts = {"seq": (x[:, None], 1, K)}
        if K <= 6:  # row s holds months s, s-1, ..., s-K+1, so one block sees lags 1..K
            layouts["par"] = (np.column_stack([np.r_[np.full(k, np.nan), x[: len(x) - k]] for k in range(K)]), K, 1)
        for name, (X, n_in, months) in layouts.items():
            n = n_in + 3
            off, _ = forecast(reservoir(X, unitary(np.zeros((n, n))), n_in, months))
            preds, w = [], 0
            for seed in range(n_seeds):
                pr, w = forecast(reservoir(X, unitary(couplings(seed, n)), n_in, months))
                preds.append(pr)
            per = np.array([mse(q) for q in preds])
            avg = np.mean(preds, axis=0)
            _, pv = dm_hac(avg - truth, ar - truth)
            row.update({f"{name}_qubits": n, f"{name}_weights": w, f"{name}_mse": per.mean(), f"{name}_sd": per.std(), f"{name}_best": per.min(),
                        f"{name}_worst": per.max(), f"{name}_seed_avg": mse(avg), f"{name}_gates_off": mse(off),
                        f"{name}_seeds_beating_ar": int((per < mse(ar)).sum()), f"{name}_dm_p_vs_ar": pv})
        records.append(row)
        print(f"K={K} done", flush=True)
    res = pd.DataFrame(records)
    res.to_csv(ROOT / "data/univariate_lags.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    for name, title in (("seq", "sequential: 1 input qubit + 3 memory, K months fed in turn"), ("par", "parallel: K input qubits + 3 memory, loaded at once")):
        cols = ["lags", "ar_weights", "ar_mse"] + [c for c in res if c.startswith(name + "_")]
        print(f"\n{title}\n" + res[cols].dropna().round(4).to_string(index=False))
    print("\nsaved data/univariate_lags.csv")


if __name__ == "__main__":
    main()
