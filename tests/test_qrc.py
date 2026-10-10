"""Acceptance checks from docs/qrc_spec.md, section 7. Run: uv run --with pytest pytest tests -q"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.qrc.circuit import (outcome_probabilities, qr1_circuit, sample_features,
                             z_from_counts, z_from_probabilities)
from src.qrc.features import FEATURES, qr1_inputs
from src.qrc.noise import gate_noise_model, step_channel
from src.qrc.reference import (couplings, evolution, month_features, reservoir_features,
                               trotter_evolution)

ROOT = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(ROOT / "data/rv_monthly.csv", parse_dates=["date"]).set_index("date").sort_index()


@pytest.fixture(scope="module")
def inputs(panel):
    return qr1_inputs(panel)[0].to_numpy()


J = couplings(0)
MONTHS = [200, 480, 700]  # rows of the panel used as forecast months


def test_feature_scaling_ranges(panel):
    X, target = qr1_inputs(panel)
    assert list(X.columns) == FEATURES
    for name, (lo, hi) in {"RV": (-1, 0), "MKT": (-1, 1), "DP": (0, 1), "STR": (-1, 1), "DEF": (0, 1)}.items():
        assert X[name].min() == pytest.approx(lo) and X[name].max() == pytest.approx(hi)
    assert X["RV_q"].min() == pytest.approx(-1) and X["RV_q"].max() < 0
    assert X["IP"].abs().max() < 0.1
    y = panel["RV_log"].to_numpy()
    assert np.allclose(target.unscale(target.scale(y)), y)


def test_clean_scaling_uses_only_training_period(panel):
    full = qr1_inputs(panel)[0]
    clean = qr1_inputs(panel, fit_end="1997-07")[0]
    changed = panel.copy()
    changed.loc["1997-08":, ["STR", "DEF", "DP", "MKT"]] *= 3.0
    clean_changed = qr1_inputs(changed, fit_end="1997-07")[0]
    assert np.allclose(clean.loc[:"1997-07"], clean_changed.loc[:"1997-07"], equal_nan=True)
    assert not np.allclose(full["DEF"], clean["DEF"])  # DEF's maximum is in the test period


def test_features_do_not_look_ahead(inputs):
    U = evolution(J)
    t = 480
    base = reservoir_features(inputs[: t + 1], U)[t]
    tampered = inputs.copy()
    tampered[t:] = 0.123
    assert np.array_equal(reservoir_features(tampered[: t + 1], U)[t], base)
    assert np.isnan(reservoir_features(inputs[:10], U)[:3]).all()


def test_circuit_matches_reference_with_same_trotter_formula(inputs):
    r = 2
    lags = inputs[MONTHS[0] - 3 : MONTHS[0]]
    z = z_from_probabilities(outcome_probabilities(qr1_circuit(lags, J, r=r)))[0]
    assert np.abs(z - month_features(lags, trotter_evolution(J, r))).max() < 1e-10


def test_circuit_r4_is_close_to_exact_evolution(inputs):
    U = evolution(J)
    circuits = [qr1_circuit(inputs[t - 3 : t], J, r=4) for t in MONTHS]
    z = z_from_probabilities(outcome_probabilities(circuits))
    exact = np.array([month_features(inputs[t - 3 : t], U) for t in MONTHS])
    assert np.abs(z - exact).max() < 0.02


def test_unrolled_24_qubit_circuit_equals_reset_circuit(inputs):
    lags = inputs[MONTHS[1] - 3 : MONTHS[1]]
    with_reset = outcome_probabilities(qr1_circuit(lags, J, r=1))[0]
    unrolled = qr1_circuit(lags, J, r=1, unrolled=True)
    assert unrolled.num_qubits == 24 and "reset" not in unrolled.count_ops()
    assert np.abs(outcome_probabilities(unrolled, method="statevector")[0] - with_reset).max() < 1e-9


def test_gate_noise_shrinks_features_and_zero_noise_is_ideal(inputs):
    lags = inputs[MONTHS[2] - 3 : MONTHS[2]]
    qc = qr1_circuit(lags, J, r=2)
    ideal = outcome_probabilities(qc)[0]
    assert gate_noise_model(0.0) is None
    noisy = outcome_probabilities(qc, noise_model=gate_noise_model(0.01))[0]
    assert noisy.sum() == pytest.approx(1.0)
    z_ideal, z_noisy = z_from_probabilities(ideal), z_from_probabilities(noisy)
    assert np.linalg.norm(z_noisy) < 0.9 * np.linalg.norm(z_ideal)
    more = z_from_probabilities(outcome_probabilities(qc, noise_model=gate_noise_model(0.03))[0])
    assert np.linalg.norm(more) < np.linalg.norm(z_noisy)


def test_step_channel_changes_output_and_zero_rate_is_none(inputs):
    lags = inputs[MONTHS[2] - 3 : MONTHS[2]]
    assert step_channel("amplitude_damping", 0.0, r=2) is None
    ideal = outcome_probabilities(qr1_circuit(lags, J, r=2))[0]
    damped = outcome_probabilities(qr1_circuit(lags, J, r=2, step_channel=step_channel("amplitude_damping", 0.2, r=2)))[0]
    assert damped.sum() == pytest.approx(1.0)
    assert np.abs(z_from_probabilities(damped) - z_from_probabilities(ideal)).max() > 0.01


def test_shot_sampling_converges_to_exact_features(inputs):
    lags = inputs[MONTHS[0] - 3 : MONTHS[0]]
    p = outcome_probabilities(qr1_circuit(lags, J, r=2))[0]
    exact = z_from_probabilities(p)
    rng = np.random.default_rng(1)
    err = {s: np.mean([np.abs(sample_features(p, s, rng) - exact).mean() for _ in range(50)]) for s in (256, 4096)}
    assert err[4096] < err[256] < 0.1
    assert err[4096] < 0.02


def test_counts_decoding_matches_probabilities(inputs):
    lags = inputs[MONTHS[0] - 3 : MONTHS[0]]
    p = outcome_probabilities(qr1_circuit(lags, J, r=2))[0]
    counts = {format(i, "010b"): int(n) for i, n in enumerate(np.random.default_rng(2).multinomial(200_000, p / p.sum())) if n}
    assert np.abs(z_from_counts(counts) - z_from_probabilities(p)).max() < 0.01
    measured = qr1_circuit(lags, J, r=2, finish="measure")
    assert measured.num_clbits == 10 and measured.count_ops()["measure"] == 10
