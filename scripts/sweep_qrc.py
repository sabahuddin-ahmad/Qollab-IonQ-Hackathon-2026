"""Shots and noise study for QR1, simulated locally on Aer (nothing here is an IonQ job).

Two stages, both resumable:

  simulate  For each noise setting, run every month's circuit once with the density-matrix
            method and cache the exact outcome probabilities under cache/qr1_probs/ (not in git).
  analyze   Sample shots from the cache, fit the readout, score through the harness window,
            and write data/qr1_noise_shots.csv.

Noise is generic and uniform (depolarizing on RXX/RY, or amplitude damping after each Trotter
step). It is not IonQ's device noise model. See docs/qrc_spec.md, section 6a.

Usage: .venv/bin/python scripts/sweep_qrc.py simulate [--workers 12] [--limit N]
       .venv/bin/python scripts/sweep_qrc.py analyze  [--reps 10]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.harness import WIN, iter_folds, load_panel
from src.qrc.circuit import outcome_probabilities, qr1_circuit, z_from_probabilities
from src.qrc.features import qr1_inputs
from src.qrc.noise import gate_noise_model, step_channel
from src.qrc.readout import RidgeReadout
from src.qrc.reference import couplings, evolution, reservoir_features

CACHE = ROOT / "cache" / "qr1_probs"
GATE_LEVELS = [0.0, 0.001, 0.004, 0.01, 0.016, 0.02, 0.03]  # 0.016 and 0.02 bracket what IonQ's simulator showed
DAMPING_LEVELS = [0.03, 0.1, 0.3, 1.0]
SHOTS = [256, 1024, 4096, None]  # None = exact expectation values
RIDGES = [1e-8, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
N_VALIDATION = 120  # months before the out-of-sample period used to choose the ridge penalty


def settings() -> list[tuple[int, str, float, int]]:
    """(seed, kind, level, trotter steps), most important first."""
    out = [(0, "gate", p, 2) for p in GATE_LEVELS]
    out += [(0, "damping", g, 2) for g in DAMPING_LEVELS]
    out += [(0, "gate", p, r) for r in (1, 4) for p in (0.0, 0.004)]  # depth against noise
    out += [(s, "gate", p, 2) for s in (1, 2) for p in (0.0, 0.001, 0.004, 0.01, 0.03)]
    return out


def cache_path(seed: int, kind: str, level: float, r: int) -> pathlib.Path:
    return CACHE / f"seed{seed}_{kind}_{level:g}_r{r}.npy"


def load_inputs() -> tuple[pd.DataFrame, np.ndarray, object]:
    panel = pd.read_csv(ROOT / "data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()
    inputs, target = qr1_inputs(panel)
    return panel, inputs.to_numpy(), target


def _simulate_chunk(seed: int, kind: str, level: float, r: int, lags: np.ndarray) -> np.ndarray:
    J = couplings(seed)
    channel = step_channel("amplitude_damping", level, r=r) if kind == "damping" else None
    noise = gate_noise_model(level) if kind == "gate" else None
    circuits = [qr1_circuit(x, J, r=r, step_channel=channel) for x in lags]
    return outcome_probabilities(circuits, noise_model=noise, max_parallel_threads=1, precision="single").astype(np.float32)


def simulate(workers: int, chunk: int, limit: int | None = None) -> None:
    """Simulate missing settings in priority order; ``limit`` stops after that many new settings."""
    CACHE.mkdir(parents=True, exist_ok=True)
    done = 0
    _, X, _ = load_inputs()
    months = [t for t in range(3, len(X)) if np.isfinite(X[t - 3 : t]).all()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for seed, kind, level, r in settings():
            path = cache_path(seed, kind, level, r)
            if path.exists():
                continue
            if limit is not None and done >= limit:
                break
            done += 1
            t0 = time.perf_counter()
            jobs = {pool.submit(_simulate_chunk, seed, kind, level, r, np.stack([X[t - 3 : t] for t in part])): part
                    for part in (months[i : i + chunk] for i in range(0, len(months), chunk))}
            probs = np.full((len(X), 1024), np.nan, dtype=np.float32)
            for job in as_completed(jobs):
                probs[jobs[job]] = job.result()
            np.save(path, probs)
            print(f"{path.name}: {len(months)} months in {time.perf_counter() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- analysis
def rolling_predictions(Z: np.ndarray, y_scaled: np.ndarray, positions: np.ndarray, ridge: float,
                        Z_train: np.ndarray | None = None) -> np.ndarray:
    """Scaled forecasts for the given row positions. Trains on the previous WIN rows of ``Z_train``
    (default: the same features), so passing ideal features there gives the transfer protocol."""
    Zt = Z if Z_train is None else Z_train
    eye = ridge * np.eye(Z.shape[1])
    out = np.empty(len(positions))
    for i, j in enumerate(positions):
        A = Zt[max(0, j - WIN) : j]
        out[i] = Z[j] @ np.linalg.solve(A.T @ A + eye, A.T @ y_scaled[max(0, j - WIN) : j])
    return out


def choose_ridge(Z: np.ndarray, y_scaled: np.ndarray, first_oos: int, Z_train: np.ndarray | None = None) -> float:
    """Penalty with the lowest error on the months just before the out-of-sample period."""
    val = np.arange(first_oos - N_VALIDATION, first_oos)
    err = [np.mean((rolling_predictions(Z, y_scaled, val, k, Z_train) - y_scaled[val]) ** 2) for k in RIDGES]
    return RIDGES[int(np.argmin(err))]


def sample(P: np.ndarray, shots: int | None, rng: np.random.Generator) -> np.ndarray:
    if shots is None:
        return z_from_probabilities(P)
    P = np.clip(P.astype(np.float64), 0.0, None)
    return z_from_probabilities(rng.multinomial(shots, P / P.sum(axis=1, keepdims=True)) / shots)


def analyze(reps: int) -> None:
    panel, X, target = load_inputs()
    full, _, _ = load_panel()
    folds, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)          # panel row of each harness row
    oos = full.index.get_indexer(oos_index)             # harness positions of the 245 test months
    y_log = full["y"].to_numpy()
    y_scaled = target.scale(y_log)
    mse = lambda pred: float(np.mean((target.unscale(pred) - y_log[oos]) ** 2))

    # the fast rolling fit must agree with the harness-driven readout used by bench_qrc.py
    Z_exact = reservoir_features(X, evolution(couplings(0)))[rows]
    table = full.join(pd.DataFrame(Z_exact, index=full.index, columns=[f"z{q}" for q in range(10)]))
    model = RidgeReadout([f"z{q}" for q in range(10)], target)
    via_harness = np.array([model.fit(f.X_window, f.y_window).predict(f.x_next) for f in iter_folds(table)[0]])
    fast = target.unscale(rolling_predictions(Z_exact, y_scaled, oos, 1e-8))
    assert np.abs(via_harness - fast).max() < 1e-8, "fast rolling fit disagrees with the harness"
    print(f"check passed: fast rolling fit equals the harness (seed 0 exact MSE {mse(target.scale(fast)):.4f})")

    records = []
    for seed, kind, level, r in settings():
        path = cache_path(seed, kind, level, r)
        ideal_path = cache_path(seed, "gate", 0.0, r)
        if not path.exists() or not ideal_path.exists():
            continue
        P, P_ideal = np.load(path)[rows], np.load(ideal_path)[rows]
        Z_ideal = z_from_probabilities(P_ideal)
        z_noisy = z_from_probabilities(P)
        shrink = float(np.sum(z_noisy[oos] * Z_ideal[oos]) / np.sum(Z_ideal[oos] ** 2))
        ridge_ideal = choose_ridge(Z_ideal, y_scaled, oos[0])
        rng = np.random.default_rng(1000 * seed + 17)
        for shots in SHOTS:
            n = 1 if shots is None else reps
            for protocol in ("consistent", "transfer"):
                scores, ridges = [], []
                for _ in range(n):
                    Z = sample(P, shots, rng)
                    if protocol == "consistent":
                        k = choose_ridge(Z, y_scaled, oos[0])
                        pred = rolling_predictions(Z, y_scaled, oos, k)
                    else:  # weights learned on ideal exact features, applied to noisy sampled ones
                        k = ridge_ideal
                        pred = rolling_predictions(Z, y_scaled, oos, k, Z_train=Z_ideal)
                    scores.append(mse(pred)); ridges.append(k)
                records.append({"seed": seed, "noise": kind, "level": level, "trotter_steps": r,
                                "shots": "exact" if shots is None else shots, "protocol": protocol,
                                "mse_mean": np.mean(scores), "mse_sd": np.std(scores), "n_rep": n,
                                "ridge_median": float(np.median(ridges)), "feature_shrink": shrink})
        print(f"analyzed {path.name}", flush=True)

    out = pd.DataFrame(records)
    out.to_csv(ROOT / "data/qr1_noise_shots.csv", index=False)
    print(f"saved data/qr1_noise_shots.csv ({len(out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["simulate", "analyze"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None, help="simulate at most this many new settings")
    args = ap.parse_args()
    simulate(args.workers, args.chunk, args.limit) if args.stage == "simulate" else analyze(args.reps)
