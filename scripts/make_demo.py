"""Build demo/qr1_demo.json, the only data main.py reads on Qollab.

Holds, for the last few out-of-sample months: the scaled inputs, the realized value, frozen
readout weights, the features a run should return, and the classical forecasts. The weights
are trained locally on features from the same circuit the demo runs (2 Trotter steps), sampled
at each supported shot count so training matches what the demo measures.

Two sets of weights: noiseless, for ideal simulators, and noise-matched, for IonQ backends,
whose simulator applies a device noise model. The matched set is trained under our generic
depolarizing model at the cached level closest to what IonQ's simulator returned.

Needs the caches from: scripts/sweep_qrc.py simulate --limit 5

Usage: .venv/bin/python scripts/make_demo.py
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from src.harness import WIN, iter_folds, load_panel
from src.qrc.circuit import z_from_probabilities
from src.qrc.reference import couplings
from hybrid_qrc import ridge_fit, rolling
from sweep_qrc import N_VALIDATION, RIDGES, cache_path, choose_ridge, load_inputs, sample

SEED, TROTTER_STEPS, N_MONTHS = 0, 2, 3
SHOT_LEVELS = [128, 256, 512, 1024, 4096]
# Calibrated on the IonQ Forte 1 simulator, 2026-10-10, forecast month 2017-12, 24-qubit layout:
# a 500-shot run kept 0.65 of the ideal signal (inputs 0.75, hidden qubits 0.38). Our generic
# depolarizing model at 1.6% per RXX gives 0.68 (0.78, 0.43) and agrees with the ten measured
# values within shot noise (chi-squared 11.1); 1% and 3% do not (73 and 89). One month, one run.
# 2026-10-11: five more 500-shot runs over three months fit 1.6% and 2% about equally (chi-squared 64 and 54 on
# 50 values). Kept at 1.6%, the level set before those runs, so they remain an out-of-sample check.
IONQ_MATCH_P2 = 0.016
BASELINES = ["HARX", "ARMAX", "HAR", "AR3", "RC", "RCX"]
MACRO_MODELS = {"HARX", "ARMAX", "RCX"}  # given macro variables as well as past volatility, like QR1


def frozen_weights(Z: np.ndarray, y_scaled: np.ndarray, j: int, ridge: float) -> np.ndarray:
    A = Z[max(0, j - WIN) : j]
    return np.linalg.solve(A.T @ A + ridge * np.eye(Z.shape[1]), A.T @ y_scaled[max(0, j - WIN) : j])


def main() -> None:
    panel, X, target = load_inputs()
    full, feature_sets, _ = load_panel()
    _, oos_index = iter_folds(full)
    rows = panel.index.get_indexer(full.index)
    oos = full.index.get_indexer(oos_index)
    y_log = full["y"].to_numpy()
    y_scaled = target.scale(y_log)
    # Hybrid readout: intercept and HARX's 13 inputs (unpenalised) plus the ten reservoir values (ridge), in log space.
    harx_inputs = np.column_stack([np.ones(len(full)), full[feature_sets["HARX"]].to_numpy()])
    n_harx = harx_inputs.shape[1]
    validation = np.arange(oos[0] - N_VALIDATION, oos[0])

    def hybrid_weights(Z: np.ndarray, j: int) -> list[float]:
        F = np.column_stack([harx_inputs, Z])
        penalty = lambda k: np.r_[np.zeros(n_harx), np.full(Z.shape[1], k)]
        best = min(RIDGES, key=lambda k: np.mean((rolling(F, y_log, validation, penalty(k)) - y_log[validation]) ** 2))
        return ridge_fit(F[max(0, j - WIN) : j], y_log[max(0, j - WIN) : j], penalty(best)).tolist()

    path = cache_path(SEED, "gate", 0.0, TROTTER_STEPS)
    if not path.exists():
        raise SystemExit(f"missing {path}; run: scripts/sweep_qrc.py simulate --limit 1")
    noisy_path = cache_path(SEED, "gate", IONQ_MATCH_P2, TROTTER_STEPS)
    if not noisy_path.exists():
        raise SystemExit(f"missing {noisy_path}; run: scripts/sweep_qrc.py simulate --limit 5")
    P, P_noisy = np.load(path)[rows], np.load(noisy_path)[rows]
    Z_exact, Z_noisy = z_from_probabilities(P), z_from_probabilities(P_noisy)

    rng = np.random.default_rng(2026)
    sampled = {shots: sample(P, shots, rng) for shots in SHOT_LEVELS}
    ridges = {shots: choose_ridge(Z, y_scaled, oos[0]) for shots, Z in sampled.items()}
    sampled_noisy = {shots: sample(P_noisy, shots, rng) for shots in SHOT_LEVELS}
    ridges_noisy = {shots: choose_ridge(Z, y_scaled, oos[0]) for shots, Z in sampled_noisy.items()}

    classical = pd.read_csv(ROOT / "data/oos_classical.csv", parse_dates=["date"]).set_index("date")
    qr1 = pd.read_csv(ROOT / "data/oos_qr1.csv", parse_dates=["date"]).set_index("date")
    months = []
    for j in oos[-N_MONTHS:]:
        date = full.index[j]
        t = rows[j]
        months.append({
            "date": date.strftime("%Y-%m"),
            "inputs": X[t - 3 : t].tolist(),  # months t-3, t-2, t-1; seven scaled features each
            "actual_log_rv": float(full["y"].iloc[j]),
            "ideal_features": Z_exact[j].tolist(),
            "weights": {str(s): frozen_weights(sampled[s], y_scaled, j, ridges[s]).tolist() for s in SHOT_LEVELS},
            "noise_matched_features": Z_noisy[j].tolist(),
            "noise_matched_weights": {str(s): frozen_weights(sampled_noisy[s], y_scaled, j, ridges_noisy[s]).tolist()
                                      for s in SHOT_LEVELS},
            "harx_inputs": harx_inputs[j].tolist(),  # intercept, then HARX's 13 inputs for this month
            "hybrid_weights": {str(s): hybrid_weights(sampled[s], j) for s in SHOT_LEVELS},
            "noise_matched_hybrid_weights": {str(s): hybrid_weights(sampled_noisy[s], j) for s in SHOT_LEVELS},
            "classical_forecasts": {m: float(classical.loc[date, m]) for m in BASELINES},
        })

    y = classical["y_log"].to_numpy()
    mse = lambda col: float(np.mean((y - col) ** 2))
    seeds = np.array([mse(qr1[c].to_numpy()) for c in qr1.columns if c.startswith("QR1_s")])
    efficiency = pd.read_csv(ROOT / "data/efficiency.csv").set_index("model")["n_params"]
    inputs = lambda m: "volatility + macro" if m in MACRO_MODELS else "volatility only"
    table = [{"model": m, "inputs": inputs(m), "mse": mse(classical[m].to_numpy()), "trained_parameters": int(efficiency[m])} for m in BASELINES]
    table.append({"model": "QR1, quantum (mean of 20 seeds)", "inputs": "volatility + macro", "mse": float(seeds.mean()), "trained_parameters": 10})

    # Headline numbers for the story main.py prints, read from the committed result files.
    controls = pd.read_csv(ROOT / "data/qr1_controls.csv").set_index("model")
    noise = pd.read_csv(ROOT / "data/qr1_noise_shots.csv")
    noise["shots"] = noise["shots"].astype(str)
    at = lambda level, shots, protocol: float(noise[(noise.noise == "gate") & (noise.trotter_steps == 2) & (noise.level == level)
                                                    & (noise.shots == shots) & (noise.protocol == protocol)]["mse_mean"].mean())
    hybrid = pd.read_csv(ROOT / "data/qr1_hybrid.csv")
    hybrid_oos = pd.read_csv(ROOT / "data/oos_qr1_hybrid.csv")
    series = {"QR1, quantum reservoir (seed 0)": qr1["QR1_s0"].to_numpy(), "HARX, linear": classical["HARX"].to_numpy(),
              "Hybrid: HARX + quantum reservoir (seed 0)": hybrid_oos["augmented_s0"].to_numpy()}
    history = {"note": "Out-of-sample forecasts, exact simulation, computed locally.",
               "dates": [d.strftime("%Y-%m") for d in classical.index], "realized": np.round(y, 4).tolist(),
               "models": {name: {"mse": mse(v), "forecast": np.round(v, 4).tolist()} for name, v in series.items()}}
    findings = {
        "qr1_mse": float(seeds.mean()), "qr1_best_seed": float(seeds.min()), "qr1_worst_seed": float(seeds.max()),
        "harx_mse": mse(classical["HARX"].to_numpy()), "armax_mse": mse(classical["ARMAX"].to_numpy()), "rcx_mse": mse(classical["RCX"].to_numpy()),
        "gates_off_mse": float(controls.loc["QR1, entangling gates off", "mse"]),
        "gates_on_mse": float(controls.loc["QR1 as designed, mean of 20 seeds", "mse_seed_mean"]),
        "gates_p_value": float(controls.loc["QR1 as designed (seed-averaged) vs gates off, alone", "dm_p"]),
        "hybrid_gates_off_mse": float(controls.loc["HARX + reservoir, entangling gates off (= cos of the 7 inputs)", "mse"]),
        "hybrid_gates_on_mse": float(controls.loc["HARX + reservoir as designed, mean of 20 seeds", "mse_seed_mean"]),
        "hybrid_mse": float(hybrid["augmented"].mean()), "hybrid_seeds_beating_harx": int((hybrid["augmented"] < mse(classical["HARX"].to_numpy())).sum()),
        "hybrid_seeds": int(len(hybrid)),
        "no_noise_mse": at(0.0, "exact", "consistent"),
        "noise_1pct_matched_mse": at(0.01, "1024", "consistent"), "noise_1pct_unmatched_mse": at(0.01, "1024", "transfer"),
    }

    demo = {
        "note": "Generated by scripts/make_demo.py. Weights and the table were computed locally (NumPy and Aer), not on Qollab.",
        "coupling_seed": SEED,
        "couplings": couplings(SEED).tolist(),
        "trotter_steps": TROTTER_STEPS,
        "target_lo": target.lo,
        "target_hi": target.hi,
        "shot_levels": SHOT_LEVELS,
        "ridge_by_shots": {str(s): ridges[s] for s in SHOT_LEVELS},
        "noise_match_p2": IONQ_MATCH_P2,
        "findings": findings,
        "history": history,
        "months": months,
        "table": sorted(table, key=lambda r: (r["inputs"] != "volatility + macro", r["mse"])),  # like for like groups
    }
    out = ROOT / "demo" / "qr1_demo.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(demo, indent=1))
    print(f"saved {out} ({out.stat().st_size / 1024:.1f} KB); months {[m['date'] for m in months]}; ridge by shots {ridges}")


if __name__ == "__main__":
    main()
