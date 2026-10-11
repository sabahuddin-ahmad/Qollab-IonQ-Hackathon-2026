"""Daily forecast: does the quantum reservoir do better on next-day volatility than on monthly?

Data: S&P 500 daily open/high/low/close from the Yahoo Finance chart API, 2000-2017 (cached in
data/spx_daily.csv). Daily volatility is the Garman-Klass range estimator; the target is its log.
Protocol, fixed before looking at results: test days 2010-01-01 to 2017-12-31, one-day-ahead forecasts,
rolling window of 1,000 days refitted every day. Input scaling uses days before the test period only,
and the ridge penalty is chosen on the 500 days before it.

Classical: AR1, AR5, AR22, HAR (1, 5 and 22-day averages), HAR + yesterday's return and its negative
part (leverage). Each with OLS and with a tuned ridge.
Quantum (QR1's Hamiltonian, RY encoding, reset between days, <Z> readout + intercept, tuned ridge):
  Q1  1 input qubit (log vol) + 3 memory, 5 days fed in
  Q2  1 input qubit (log vol) + 3 memory, 22 days fed in
  Q3  3 input qubits (HAR's three averages) + 3 memory, 3 days fed in
  Q4  2 input qubits (log vol, return) + 3 memory, 22 days fed in
Each also with the entangling gates off. 20 coupling seeds, exact simulation.

With --sweeps, two further checks on the same test days (default 10 seeds there):
  lags     the sequential reservoir (1 input + 3 memory) fed K = 1, 2, 5, 10, 22, 44 days, against AR(K)
  windows  the main models with training windows of 250, 500, 1,000 and 2,000 days

Writes data/daily.csv, or data/daily_lags.csv and data/daily_windows.csv with --sweeps.

Usage: .venv/bin/python scripts/bench_daily.py [--seeds 20] [--sweeps]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from bench_univariate import unitary
from hybrid_qrc import dm_hac
from src.qrc.reference import couplings

CACHE = ROOT / "data/spx_daily.csv"
WIN, N_VAL = 1000, 500
TEST_START, TEST_END = "2010-01-01", "2017-12-31"
RIDGES = [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
DESIGNS = {"Q1": (["v"], 5), "Q2": (["v"], 22), "Q3": (["v", "w", "m"], 3), "Q4": (["v", "r"], 22)}
LABEL = {"Q1": "1 input + 3 memory, 5 days", "Q2": "1 input + 3 memory, 22 days",
         "Q3": "3 inputs (HAR's) + 3 memory, 3 days", "Q4": "2 inputs (vol, return) + 3 memory, 22 days"}


def load_daily() -> pd.DataFrame:
    if not CACHE.exists():
        import requests

        p1 = int(datetime(1999, 10, 1, tzinfo=timezone.utc).timestamp())
        p2 = int(datetime(2018, 1, 5, tzinfo=timezone.utc).timestamp())
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?period1={p1}&period2={p2}&interval=1d"
        j = requests.get(url, headers={"User-Agent": "Mozilla/5.0 qollab-hackathon-research"}, timeout=60).json()["chart"]["result"][0]
        q = j["indicators"]["quote"][0]
        idx = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(None).floor("D")
        raw = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")}, index=idx).dropna()
        o, h, l, c = (np.log(raw[k]) for k in ("open", "high", "low", "close"))
        gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2  # Garman-Klass daily variance
        out = pd.DataFrame({"log_vol": 0.5 * np.log(gk.clip(lower=1e-10)), "ret": c.diff()}).dropna()
        out.index.name = "date"
        out.to_csv(CACHE, float_format="%.8f")
    return pd.read_csv(CACHE, parse_dates=["date"]).set_index("date")


def reservoir(X: np.ndarray, U: np.ndarray, n_in: int, days: int) -> np.ndarray:
    """Row t: <Z> on every qubit after feeding rows t-days..t-1, inputs reset between days (density-matrix form)."""
    n = int(np.log2(len(U)))
    d_in, d_mem = 2**n_in, 2 ** (n - n_in)
    idx = np.arange(2**n)
    signs = np.array([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])
    half = np.pi * X / 2
    kets = np.stack([np.cos(half), np.sin(half)], axis=-1)  # (T, n_in, 2)
    psi = kets[:, 0]
    for q in range(1, n_in):
        psi = np.einsum("ta,tb->tab", psi, kets[:, q]).reshape(len(X), -1)
    Ur = U.reshape(d_in, d_mem, d_in, d_mem)
    out = np.full((len(X), n), np.nan)
    rho0 = np.zeros((d_mem, d_mem), dtype=complex)
    rho0[0, 0] = 1.0
    for t in range(days, len(X)):
        rho = rho0
        for s in range(t - days, t):
            V = np.einsum("imjh,j->imh", Ur, psi[s])            # acts on memory, outputs input (x) memory
            full = np.einsum("imh,hg,jng->imjn", V, rho, V.conj())
            if s == t - 1:
                out[t] = signs @ np.einsum("imim->im", full).real.ravel()
            else:
                rho = np.einsum("imin->mn", full)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--sweeps", action="store_true")
    ap.add_argument("--corner", action="store_true", help="few days fed in combined with short training windows")
    args = ap.parse_args()
    args.seeds = args.seeds or (10 if args.sweeps else 20)

    d = load_daily()
    y = d["log_vol"].to_numpy()
    test = np.flatnonzero((d.index >= TEST_START) & (d.index <= TEST_END))
    val = np.arange(test[0] - N_VAL, test[0])
    pre = slice(0, test[0])
    truth = y[test]
    mse = lambda p: float(np.mean((p - truth) ** 2))
    print(f"{len(d)} days {d.index[0]:%Y-%m-%d} to {d.index[-1]:%Y-%m-%d}; test {len(test)} days from {d.index[test[0]]:%Y-%m-%d}; "
          f"log-vol sd {y[test].std():.3f}")

    lag = lambda a, k: np.r_[np.full(k, np.nan), a[:-k]]
    avg = lambda k: pd.Series(y).rolling(k).mean().shift(1).to_numpy()
    r1 = lag(d["ret"].to_numpy(), 1)
    sets = {"AR1": [lag(y, 1)], "AR5": [lag(y, k) for k in range(1, 6)], "AR22": [lag(y, k) for k in range(1, 23)],
            "HAR": [avg(1), avg(5), avg(22)], "HAR + return (leverage)": [avg(1), avg(5), avg(22), r1, np.minimum(r1, 0)]}

    def roll(F, positions, penalty, win=WIN):
        out = np.empty(len(positions))
        for i, j in enumerate(positions):
            A, b = F[j - win : j], y[j - win : j]
            out[i] = F[j] @ np.linalg.solve(A.T @ A + np.diag(penalty), A.T @ b)
        return out

    def fit(E, tuned, win=WIN):
        E = np.atleast_2d(E.T).T
        E = E[:, np.nanstd(E[test], axis=0) > 1e-9]
        E = (E - np.nanmean(E[pre], axis=0)) / np.nanstd(E[pre], axis=0)
        F = np.column_stack([np.nan_to_num(E), np.ones(len(E))])
        pen = lambda k: np.r_[np.full(E.shape[1], k), 0.0]
        k = min(RIDGES, key=lambda r: np.mean((roll(F, val, pen(r), win) - y[val]) ** 2)) if tuned else 0.0
        return roll(F, test, pen(k), win), F.shape[1]

    def scale(a, lo, hi):  # min-max on pre-test data, clipped
        a0, a1 = np.nanmin(a[pre]), np.nanmax(a[pre])
        return np.clip((a - a0) / (a1 - a0), 0, 1) * (hi - lo) + lo

    ins = {"v": scale(y, -1, 0), "w": scale(pd.Series(y).rolling(5).mean().to_numpy(), -1, 0),
           "m": scale(pd.Series(y).rolling(22).mean().to_numpy(), -1, 0)}
    r = d["ret"].to_numpy()
    ins["r"] = np.clip(r / np.abs(r[pre]).max(), -1, 1)
    if args.sweeps:
        return sweeps(args.seeds, sets, ins, fit, mse, truth, lag, y)
    if args.corner:
        return corner(args.seeds, sets, ins, fit, mse, truth, lag, y)

    records, preds = [], {}
    for name, cols in sets.items():
        E = np.column_stack(cols)
        p_ols, w = fit(E, False)
        p_ridge, _ = fit(E, True)
        preds[name] = p_ridge
        records.append({"model": name, "kind": "classical", "weights": w, "mse_ols": mse(p_ols), "mse": mse(p_ridge)})
        print(f"{name:26s} {w:3d} weights  OLS {mse(p_ols):.4f}  ridge {mse(p_ridge):.4f}", flush=True)
    records.append({"model": "Persistence", "kind": "classical", "weights": 0, "mse": mse(y[test - 1])})

    for key, (cols, days) in DESIGNS.items():
        X = np.column_stack([ins[c] for c in cols])
        n = len(cols) + 3
        base = "HAR + return (leverage)" if "r" in cols else "HAR"
        p_off, _ = fit(reservoir(X, unitary(np.zeros((n, n))), len(cols), days), True)
        ps, w = [], 0
        for seed in range(args.seeds):
            p, w = fit(reservoir(X, unitary(couplings(seed, n)), len(cols), days), True)
            ps.append(p)
        per = np.array([mse(p) for p in ps])
        ens = np.mean(ps, axis=0)
        t, pv = dm_hac(ens - truth, preds[base] - truth, lag=10)
        records.append({"model": f"{key}: {LABEL[key]}", "kind": "quantum", "weights": w, "mse": per.mean(), "mse_sd": per.std(),
                        "mse_best": per.min(), "mse_worst": per.max(), "mse_seed_averaged": mse(ens), "mse_gates_off": mse(p_off),
                        "seeds_beating_gates_off": int((per < mse(p_off)).sum()), "compared_with": base, "mse_compared": mse(preds[base]),
                        "seeds_beating_it": int((per < mse(preds[base])).sum()), "dm_t": t, "dm_p": pv})
        print(f"{key} done: mean {per.mean():.4f}", flush=True)

    res = pd.DataFrame(records)
    res.to_csv(ROOT / "data/daily.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    q = res[res.kind == "quantum"]
    print("\n" + q[["model", "weights", "mse", "mse_sd", "mse_best", "mse_worst", "mse_seed_averaged", "mse_gates_off", "seeds_beating_gates_off",
                    "compared_with", "mse_compared", "seeds_beating_it", "dm_t", "dm_p"]].round(4).to_string(index=False))
    print("\nsaved data/daily.csv")


def corner(n_seeds, sets, ins, fit, mse, truth, lag, y) -> None:
    """The sequential reservoir against AR(K) with K = 1, 2, 5 days fed in and training windows of 125 to 1,000 days."""
    x = ins["v"][:, None]
    feats = {K: [reservoir(x, unitary(couplings(s, 4)), 1, K) for s in range(n_seeds)] for K in (1, 2, 5)}
    rows = []
    for win in (125, 250, 500, 1000):
        har = mse(fit(np.column_stack(sets["HAR"]), True, win)[0])
        lev = mse(fit(np.column_stack(sets["HAR + return (leverage)"]), True, win)[0])
        for K in (1, 2, 5):
            ar = fit(np.column_stack([lag(y, k) for k in range(1, K + 1)]), True, win)[0]
            ps = [fit(Z, True, win)[0] for Z in feats[K]]
            per = np.array([mse(p) for p in ps])
            t, pv = dm_hac(np.mean(ps, axis=0) - truth, ar - truth, lag=10)
            rows.append({"window": win, "days_fed_in": K, "ar_mse": mse(ar), "reservoir_mse": per.mean(), "reservoir_sd": per.std(),
                         "seeds_beating_ar": int((per < mse(ar)).sum()), "gain_pct": 100 * (1 - per.mean() / mse(ar)), "dm_t": t, "dm_p": pv,
                         "har_mse": har, "har_return_mse": lev})
        print(f"window {win} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "data/daily_corner.csv", index=False, float_format="%.6f")
    pd.set_option("display.width", 250)
    print("\n" + res.round(4).to_string(index=False) + f"\n\n{n_seeds} seeds. saved data/daily_corner.csv")


def sweeps(n_seeds, sets, ins, fit, mse, truth, lag, y) -> None:
    pd.set_option("display.width", 250)
    # --- days fed in, lag for lag
    x = ins["v"][:, None]
    off, _ = fit(reservoir(x, unitary(np.zeros((4, 4))), 1, 1), True)
    rows = []
    for K in (1, 2, 5, 10, 22, 44):
        ar, w = fit(np.column_stack([lag(y, k) for k in range(1, K + 1)]), True)
        ps = [fit(reservoir(x, unitary(couplings(s, 4)), 1, K), True)[0] for s in range(n_seeds)]
        per = np.array([mse(p) for p in ps])
        t, pv = dm_hac(np.mean(ps, axis=0) - truth, ar - truth, lag=10)
        rows.append({"days_fed_in": K, "ar_weights": w, "ar_mse": mse(ar), "reservoir_weights": 5, "reservoir_mse": per.mean(), "reservoir_sd": per.std(),
                     "reservoir_best": per.min(), "seeds_beating_ar": int((per < mse(ar)).sum()), "gates_off_mse": mse(off), "dm_p_vs_ar": pv})
        print(f"lags K={K} done", flush=True)
    lags = pd.DataFrame(rows)
    lags.to_csv(ROOT / "data/daily_lags.csv", index=False, float_format="%.6f")
    print("\n" + lags.round(4).to_string(index=False) + "\n")

    # --- training window
    feats = {key: [reservoir(np.column_stack([ins[c] for c in cols]), unitary(couplings(s, len(cols) + 3)), len(cols), days) for s in range(n_seeds)]
             for key, (cols, days) in DESIGNS.items()}
    print("reservoir features ready", flush=True)
    rows = []
    for win in (250, 500, 1000, 2000):
        ref = {}
        for name in ("AR5", "AR22", "HAR", "HAR + return (leverage)"):
            E = np.column_stack(sets[name])
            ref[name] = mse(fit(E, True, win)[0])
            rows.append({"window": win, "model": name, "mse": ref[name], "mse_ols": mse(fit(E, False, win)[0])})
        for key, zs in feats.items():
            per = np.array([mse(fit(Z, True, win)[0]) for Z in zs])
            base = "HAR + return (leverage)" if key == "Q4" else "HAR"
            rows.append({"window": win, "model": f"{key}: {LABEL[key]}", "mse": per.mean(), "mse_sd": per.std(), "mse_best": per.min(),
                         "seeds_beating_linear": int((per < ref[base]).sum()), "compared_with": base})
        print(f"window {win} done", flush=True)
    wins = pd.DataFrame(rows)
    wins.to_csv(ROOT / "data/daily_windows.csv", index=False, float_format="%.6f")
    print("\nridge-tuned MSE by training window (days):\n" + wins.pivot(index="model", columns="window", values="mse").round(4).to_string())
    print("\nOLS, linear models:\n" + wins.dropna(subset=["mse_ols"]).pivot(index="model", columns="window", values="mse_ols").round(4).to_string())
    q = wins.dropna(subset=["seeds_beating_linear"])
    print("\nseeds beating the linear counterpart (of %d):\n" % n_seeds + q.pivot(index="model", columns="window", values="seeds_beating_linear").astype(int).to_string())
    print("\nsaved data/daily_lags.csv and data/daily_windows.csv")


if __name__ == "__main__":
    main()
