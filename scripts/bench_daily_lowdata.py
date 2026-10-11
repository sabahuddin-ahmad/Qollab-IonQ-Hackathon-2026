"""Low-data control on daily volatility: is the reservoir's small edge over AR(K) quantum?

Setting where the 4-qubit reservoir was marginally ahead of a matched autoregression
(scripts/bench_daily.py --corner): training windows of 125 and 250 days, K = 1, 2, 5 days fed in.
Every model sees the same K days of log volatility and gets the same readout (ridge tuned on the
500 days before the test period, intercept unpenalised). Test: 2,013 days, 2010-2017.

  AR(K)             linear in the K days
  AR(K) + squares   adds the square of each day
  cos/sin           cos(pi x) and sin(pi x) of each day: what the RY encoding gives without any dynamics
  ESN-n             classical tanh reservoir with n units (n = 4, 6, 8), fed the K days in turn from a
                    zero state, like the quantum one; readout on its n final values
  QRC-n             quantum reservoir with n qubits (1 input + n-1 memory), readout on its n <Z> values

Seeds: 20, except 5 for QRC-8. Writes data/daily_lowdata.csv.

Usage: .venv/bin/python scripts/bench_daily_lowdata.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from bench_daily import N_VAL, RIDGES, TEST_END, TEST_START, load_daily
from bench_univariate import unitary
from hybrid_qrc import dm_hac
from src.qrc.reference import couplings

WINDOWS, LAGS, SIZES = (125, 250), (1, 2, 5), (4, 6, 8)
SEEDS = {4: 20, 6: 20, 8: 5}


def qrc(x: np.ndarray, U: np.ndarray, days: int, first: int) -> np.ndarray:
    """<Z> on all qubits after feeding days t-days..t-1 (one input qubit, reset between days), for rows >= first."""
    n = int(np.log2(len(U)))
    d = 2 ** (n - 1)
    idx = np.arange(2**n)
    signs = np.array([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)]).reshape(n, 2, d)
    Ur = U.reshape(2, d, 2, d)
    c, s = np.cos(np.pi * x / 2), np.sin(np.pi * x / 2)
    out = np.full((len(x), n), np.nan)
    for t in range(max(first, days), len(x)):
        rho = np.zeros((d, d), dtype=complex)
        rho[0, 0] = 1.0
        for k in range(t - days, t):
            V = Ur[:, :, 0, :] * c[k] + Ur[:, :, 1, :] * s[k]      # (2, d, d): memory in -> input (x) memory out
            A = V @ rho
            if k == t - 1:
                out[t] = np.einsum("qim,im->q", signs, np.einsum("img,img->im", A, V.conj()).real)
            else:
                rho = np.einsum("img,ing->mn", A, V.conj())
    return out


def esn(x: np.ndarray, n: int, days: int, seed: int) -> np.ndarray:
    """Final state of an n-unit tanh reservoir fed the same days from a zero state (spectral radius 0.9)."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    W *= 0.9 / max(abs(np.linalg.eigvals(W)))
    w_in, b = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    h = np.zeros((len(x), n))
    for k in range(days, 0, -1):  # oldest first
        xk = np.r_[np.full(k, np.nan), x[:-k]]
        h = np.tanh(h @ W.T + np.outer(xk, w_in) + b)
    return h


def main() -> None:
    d = load_daily()
    y = d["log_vol"].to_numpy()
    test = np.flatnonzero((d.index >= TEST_START) & (d.index <= TEST_END))
    val = np.arange(test[0] - N_VAL, test[0])
    pre = slice(0, test[0])
    first = val[0] - max(WINDOWS)
    truth = y[test]
    mse = lambda p: float(np.mean((p - truth) ** 2))
    x = np.clip((y - y[pre].min()) / (y[pre].max() - y[pre].min()), 0, 1) - 1.0  # [-1, 0], pre-test scaling
    lag = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]

    def roll(F, positions, penalty, win):
        out = np.empty(len(positions))
        for i, j in enumerate(positions):
            A, b = F[j - win : j], y[j - win : j]
            out[i] = F[j] @ np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ b)
        return out

    def fit(E, win):
        E = E[:, np.nanstd(E[test], axis=0) > 1e-9]
        E = (E - np.nanmean(E[first:test[0]], axis=0)) / np.nanstd(E[first:test[0]], axis=0)
        F = np.column_stack([np.nan_to_num(E), np.ones(len(E))])
        pen = lambda k: np.r_[np.full(E.shape[1], k), 0.0]
        k = min(RIDGES, key=lambda r: np.mean((roll(F, val, pen(r), win) - y[val]) ** 2))
        return roll(F, test, pen(k), win)

    feats = {}
    for K in LAGS:
        for n in SIZES:
            feats[("QRC", n, K)] = [qrc(x, unitary(couplings(s, n)), K, first) for s in range(SEEDS[n])]
            feats[("ESN", n, K)] = [esn(x, n, K, s) for s in range(20)]
        print(f"features for K={K} ready", flush=True)

    records = []
    for win in WINDOWS:
        for K in LAGS:
            L = np.column_stack([lag(y, k) for k in range(1, K + 1)])
            Lx = np.column_stack([lag(x, k) for k in range(1, K + 1)])
            ar = fit(L, win)
            base = {"window": win, "days_fed_in": K}
            records.append({**base, "model": "AR(K)", "weights": K + 1, "mse": mse(ar)})
            records.append({**base, "model": "AR(K) + squares", "weights": 2 * K + 1, "mse": mse(fit(np.column_stack([L, L**2]), win))})
            records.append({**base, "model": "cos/sin of each day", "weights": 2 * K + 1,
                            "mse": mse(fit(np.column_stack([np.cos(np.pi * Lx), np.sin(np.pi * Lx)]), win))})
            avg = {}
            for kind in ("ESN", "QRC"):
                for n in SIZES:
                    ps = [fit(Z, win) for Z in feats[(kind, n, K)]]
                    per = np.array([mse(p) for p in ps])
                    avg[(kind, n)] = np.mean(ps, axis=0)
                    t, pv = dm_hac(avg[(kind, n)] - truth, ar - truth, lag=10)
                    records.append({**base, "model": f"{kind}-{n}", "weights": n + 1, "mse": per.mean(), "mse_sd": per.std(), "mse_best": per.min(),
                                    "mse_worst": per.max(), "n_seeds": len(per), "seeds_beating_ar": int((per < mse(ar)).sum()),
                                    "mse_seed_averaged": mse(avg[(kind, n)]), "dm_p_vs_ar": pv})
            for n in SIZES:
                t, pv = dm_hac(avg[("QRC", n)] - truth, avg[("ESN", n)] - truth, lag=10)
                records[-1 - (len(SIZES) - 1 - SIZES.index(n))]["dm_t_vs_esn"] = t
                records[-1 - (len(SIZES) - 1 - SIZES.index(n))]["dm_p_vs_esn"] = pv
            print(f"window {win}, K={K} done", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(ROOT / "data/daily_lowdata.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    order = ["AR(K)", "AR(K) + squares", "cos/sin of each day"] + [f"{k}-{n}" for n in SIZES for k in ("ESN", "QRC")]
    table = res.pivot(index="model", columns=["window", "days_fed_in"], values="mse").loc[order]
    print("\nmean MSE (columns: training window, days fed in)\n" + table.round(4).to_string())
    q = res[res.model.str.startswith(("QRC", "ESN"))]
    print("\nseeds beating AR(K) / seeds:\n" + (q.seeds_beating_ar.astype(int).astype(str) + "/" + q.n_seeds.astype(int).astype(str)).to_frame("v")
          .join(q[["model", "window", "days_fed_in"]]).pivot(index="model", columns=["window", "days_fed_in"], values="v").to_string())
    qq = res[res.model.str.startswith("QRC")]
    print("\nDM p-value, QRC-n vs ESN-n (seed-averaged forecasts; t < 0 means QRC better):\n"
          + (qq.dm_t_vs_esn.round(2).astype(str) + " (p=" + qq.dm_p_vs_esn.round(3).astype(str) + ")").to_frame("v")
          .join(qq[["model", "window", "days_fed_in"]]).pivot(index="model", columns=["window", "days_fed_in"], values="v").to_string())
    print("\nsaved data/daily_lowdata.csv")


if __name__ == "__main__":
    main()
